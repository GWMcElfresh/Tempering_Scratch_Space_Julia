#!/usr/bin/env julia

# Generate a realistic mock donor-level count table compatible with
# scripts/run_efdm.jl and config/model_config.toml.

using Random
using CSV
using DataFrames
using Distributions

function logistic(x)
    return 1.0 / (1.0 + exp(-x))
end

function simplex_from_scores(scores::Vector{Float64})
    m = maximum(scores)
    w = exp.(scores .- m)
    return w ./ sum(w)
end

function make_mock_df(n::Int; seed::Int=20260602)
    rng = MersenneTwister(seed)

    donor_id = ["donor_$(lpad(string(i), 3, '0'))" for i in 1:n]

    # Binary treatment encoded as numeric 0/1 to match runner expectations.
    treatment = rand(rng, Bernoulli(0.5), n)

    # Ages roughly centered in an adult cohort.
    age = clamp.(round.(Int, rand(rng, Normal(48, 11), n)), 20, 80)

    # Generate realistic library sizes with moderate spread.
    total_counts = clamp.(round.(Int, rand(rng, LogNormal(log(1300), 0.30), n)), 450, 4200)

    counts = Matrix{Int}(undef, n, 6)
    age_z = (age .- mean(age)) ./ std(age)

    for i in 1:n
        tr = Float64(treatment[i])
        az = age_z[i]

        # Covariate-informed latent composition signal.
        scores = [
            -0.5 + 0.7 * tr - 0.2 * az,
            0.2 - 0.3 * tr + 0.4 * az,
            0.0 + 0.2 * tr + 0.1 * az,
            -0.2 + 0.1 * tr - 0.3 * az,
            -1.1 + 0.8 * tr + 0.5 * az,
            -2.2 - 0.2 * tr - 0.1 * az,
        ]

        mu = simplex_from_scores(scores)

        # Add donor-level overdispersion with a baseline mixture component.
        p0 = [0.28, 0.23, 0.17, 0.16, 0.11, 0.05]
        w = 0.15 .+ 0.20 .* logistic(0.8 * tr - 0.5 * az)
        blended = (1.0 - w) .* mu .+ w .* p0
        alpha = 35.0 .* blended

        q = rand(rng, Dirichlet(alpha))
        y = rand(rng, Multinomial(total_counts[i], q))
        counts[i, :] = y
    end

    return DataFrame(
        donor_id=donor_id,
        CT0=counts[:, 1],
        CT1=counts[:, 2],
        CT2=counts[:, 3],
        CT3=counts[:, 4],
        CT4=counts[:, 5],
        CT5=counts[:, 6],
        treatment=Int.(treatment),
        age=age,
    )
end

function main()
    out_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "data", "mock_counts.csv")
    n = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 80
    seed = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20260602

    mkpath(dirname(out_path))
    df = make_mock_df(n; seed=seed)
    CSV.write(out_path, df)

    println("Wrote mock dataset: $(abspath(out_path))")
    println("Rows: $(nrow(df))")
    println("Columns: $(join(names(df), ", "))")
end

main()