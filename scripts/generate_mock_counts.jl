#!/usr/bin/env julia

# Generate a realistic mock donor-level count table compatible with
# scripts/run_efdm.jl and config/model_config.toml.
#
# DATA GENERATING PROCESS
# ========================
# The mock data follows the EFDM (Extended Flexible Dirichlet Multinomial)
# mixture model. For each donor i:
#
#   1. Covariate-informed composition:
#        μ_i = softmax(X_i · β)
#      where β[j,:] = score_coef[j,:] - score_coef[D,:] (reference category D)
#
#   2. Per-category mixture (fixed w, not covariate-dependent):
#        For each category j: w̃_j = w_j × min(1, μ_ij / p0_j)
#        temp2 = Σ_j p0_j × w̃_j
#        α_base,j = aplus × max(μ_ij - p0_j × w̃_j, 0) / (1 - temp2)
#
#   3. Latent category assignment:
#        z ~ Categorical(p0)
#        τ_z = aplus × w̃_z / max(1 - w̃_z, ε)
#        α = α_base + τ_z × e_z   (boost concentration for category z)
#
#   4. Dirichlet-Multinomial draw:
#        q ~ Dirichlet(α)
#        y ~ Multinomial(total_i, q)
#
# CRITICAL: This matches the EFDM model specification exactly. Earlier versions
# used a simple Dirichlet-Multinomial (alpha = aplus * blended) which is NOT
# the same as the EFDM mixture and caused severe model-data mismatch.
#
# GENERATING PARAMETERS (ground truth for model validation)
# ==========================================================
# These are the true parameter values used to generate the data.
#
# Default: D=6 categories (CT0..CT5). Example score coefficients:
#   CT0:  intercept=-0.5,  treatment= 0.7, age_z=-0.2
#   CT1:  intercept= 0.2,  treatment=-0.3, age_z= 0.4
#   CT2:  intercept= 0.0,  treatment= 0.2, age_z= 0.1
#   CT3:  intercept=-0.2,  treatment= 0.1, age_z=-0.3
#   CT4:  intercept=-1.1,  treatment= 0.8, age_z= 0.5
#   CT5:  intercept=-2.2,  treatment=-0.2, age_z=-0.1   (EFDM reference)
#
# For D=3 (convergence test mode), only CT0, CT1, CT2 are used with
# their corresponding first-3-rows of score_coef, p0, and w arrays.
# The EFDM reference is CT2 when D=3.
#
# The EFDM parameterization uses the last category as the reference. With D=6,
# the reference is CT5; with D=3, the reference is CT2.
#   β[j,:] = score_coef[j,:] - score_coef[D,:]   for j in 1:(D-1)
#
# Baseline mixture proportions (6-element simplex):
#   p0 = [0.28, 0.23, 0.17, 0.16, 0.11, 0.05]
#
# Per-category component weights (D=6 independent w_j values):
#   w_base      = [0.15, 0.20, 0.18, 0.25, 0.30, 0.35]
#   w_tr_shift  = [0.10, -0.05, 0.08, 0.12, 0.15, 0.06]
#   w_age_shift = [-0.08, 0.06, -0.04, 0.10, 0.12, -0.03]
#
# Overdispersion:
#   aplus = 35.0
#
# For model validation, compute the covariate-driven β from score_coef,
# then compare posterior means to these ground-truth values.
# The w_j posteriors should cluster near each category's donor-averaged w.

using Random
using CSV
using DataFrames
using Distributions

"""
    generating_params(; D::Int=6, signal_scale::Float64=1.0) -> NamedTuple

Return the ground-truth generating parameters for the mock data.
Useful for model validation: compare posterior means to these values.

Supported values for D: 3, 4, or 6 (default 6).

`signal_scale` multiplies the score coefficients. Use signal_scale=2.0
for the convergence test to ensure stronger signal-to-noise ratio
and faster MCMC mixing.

For D=3 (convergence test mode), only CT0, CT1, CT2 are generated.
For D=4, categories CT0..CT3 (CT3 is the EFDM reference).

Returns
  score_coef — D×3 matrix (categories x [intercept, treatment, age_z])
  p0         — D-element baseline mixture simplex
  w_base     — D-element per-category baseline component weights
  w_tr_shift — D-element treatment-induced shift per category
  w_age_shift — D-element age_z-induced shift per category
  aplus      — scalar overdispersion
  D          — number of response categories
  K          — number of covariates (inc. intercept)
"""
function generating_params(; D::Int=6, signal_scale::Float64=1.0)
    # Base score coefficients (signal_scale=1.0 gives the standard values)
    if D == 6
        base_coef = [
            -0.5   0.7  -0.2   # CT0
             0.2  -0.3   0.4   # CT1
             0.0   0.2   0.1   # CT2
            -0.2   0.1  -0.3   # CT3
            -1.1   0.8   0.5   # CT4
            -2.2  -0.2  -0.1   # CT5 (reference)
        ]
        p0         = [0.28, 0.23, 0.17, 0.16, 0.11, 0.05]
        w_base     = [0.15, 0.20, 0.18, 0.25, 0.30, 0.35]
        w_tr_shift = [0.10, -0.05, 0.08, 0.12, 0.15, 0.06]
        w_age_shift = [-0.08, 0.06, -0.04, 0.10, 0.12, -0.03]
    elseif D == 4
        base_coef = [
            -0.5   0.7  -0.2   # CT0
             0.2  -0.3   0.4   # CT1
             0.0   0.2   0.1   # CT2
            -0.2   0.1  -0.3   # CT3 (reference)
        ]
        p0         = [0.30, 0.28, 0.22, 0.20]
        w_base     = [0.15, 0.20, 0.18, 0.25]
        w_tr_shift = [0.10, -0.05, 0.08, 0.12]
        w_age_shift = [-0.08, 0.06, -0.04, 0.10]
    elseif D == 3
        base_coef = [
            -0.5   0.7  -0.2   # CT0
             0.2  -0.3   0.4   # CT1
             0.0   0.2   0.1   # CT2 (reference)
        ]
        p0         = [0.40, 0.35, 0.25]
        w_base     = [0.15, 0.20, 0.18]
        w_tr_shift = [0.10, -0.05, 0.08]
        w_age_shift = [-0.08, 0.06, -0.04]
    else
        error("D=$D not supported. Use D=3, 4, or 6.")
    end
    score_coef = base_coef .* signal_scale
    aplus = 35.0
    K = 3
    return (; score_coef, p0, w_base, w_tr_shift, w_age_shift, aplus, D, K)
end

function logistic(x)
    return 1.0 / (1.0 + exp(-x))
end

function simplex_from_scores(scores::AbstractVector{Float64})
    m = maximum(scores)
    w = exp.(scores .- m)
    return w ./ sum(w)
end

function make_mock_df(n::Int; seed::Int=20260602, n_categories::Int=6, signal_scale::Float64=1.0)
    rng = MersenneTwister(seed)
    gp = generating_params(D=n_categories, signal_scale=signal_scale)
    D = gp.D
    K = 3

    donor_id = ["donor_$(lpad(string(i), 3, '0'))" for i in 1:n]
    treatment = rand(rng, Bernoulli(0.5), n)
    age = clamp.(round.(Int, rand(rng, Normal(48, 11), n)), 20, 80)
    total_counts = clamp.(round.(Int, rand(rng, LogNormal(log(1300), 0.30), n)), 450, 4200)

    # Convert generating params to EFDM parameters.
    # β[j,:] = score_coef[j,:] - score_coef[D,:],  β[D,:] = 0 (reference)
    beta_true = similar(gp.score_coef)
    for j in 1:D
        for k in 1:K
            beta_true[j, k] = gp.score_coef[j, k] - gp.score_coef[D, k]
        end
    end

    # Fixed EFDM parameters (not donor-dependent):
    #   p0       = gp.p0         baseline mixture weights (simplex)
    #   w_norm   = gp.w_base      component weights (in (0,1))
    #   aplus    = gp.aplus      overdispersion
    #
    # The w_base is used directly (no covariate shifts) because the EFDM
    # model has fixed w parameters shared across all donors.
    p0 = gp.p0
    w_norm = gp.w_base
    aplus_val = gp.aplus

    age_z = (age .- mean(age)) ./ std(age)
    X = hcat(ones(n), Float64.(treatment), age_z)

    counts = Matrix{Int}(undef, n, D)
    eps_sim = 1e-8

    for i in 1:n
        # 1. Regression-driven proportions from softmax(Xβ)
        scores = gp.score_coef * X[i, :]
        # softmax (numerically stable)
        m = maximum(scores)
        z = exp.(scores .- m)
        mu_i = z ./ sum(z)

        # 2. Per-category clamping: w_j × min(1, mu_j / p0_j)
        w_clamped = [clamp(w_norm[j] * min(1.0, mu_i[j] / p0[j]), eps_sim, 1.0 - eps_sim) for j in 1:D]
        temp2 = sum(p0[j] * w_clamped[j] for j in 1:D)
        temp2 = clamp(temp2, eps_sim, 1.0 - eps_sim)
        inv_scale = inv(1.0 - temp2)

        # 3. Base concentration (α) shared by all mixture components
        alpha_base = [max(aplus_val * max(mu_i[j] - p0[j] * w_clamped[j], 0.0) * inv_scale, eps_sim) for j in 1:D]

        # 4. Draw latent category assignment z ~ Categorical(p0)
        z = rand(rng, Categorical(p0))

        # 5. Add τ boost for the selected category
        w_z = w_clamped[z]
        tau_z = max(aplus_val * w_z / max(1.0 - w_z, eps_sim), eps_sim)
        alpha = copy(alpha_base)
        alpha[z] += tau_z

        # 6. Draw q from Dirichlet(α) and y from Multinomial
        q = rand(rng, Dirichlet(alpha))
        y = rand(rng, Multinomial(total_counts[i], q))
        counts[i, :] = y
    end

    col_names = ["CT$i" for i in 0:(D - 1)]
    result_df = DataFrame(
        donor_id=donor_id,
        treatment=Int.(treatment),
        age=age,
    )
    for j in 1:D
        result_df[!, col_names[j]] = counts[:, j]
    end
    return result_df
end

function main()
    out_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "data", "mock_counts.csv")
    n = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 80
    seed = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20260602
    n_cat = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 6
    signal = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1.0

    gp = generating_params(D=n_cat, signal_scale=signal)

    mkpath(dirname(out_path))
    df = make_mock_df(n; seed=seed, n_categories=n_cat, signal_scale=signal)
    CSV.write(out_path, df)

    println("Wrote mock dataset: $(abspath(out_path))")
    println("Rows: $(nrow(df))")
    println("Columns: $(join(names(df), ", "))")
    println()
    println("Generating parameters (D=$(gp.D), K=$(gp.K), aplus=$(gp.aplus), signal_scale=$(signal)):")
    println("  score_coef: $(size(gp.score_coef, 1)) categories x $(size(gp.score_coef, 2)) covariates")
    println("  p0:         $(join(round.(gp.p0, digits=3), ", "))")
    println("  w_base:     $(join(round.(gp.w_base, digits=3), ", "))")
    println("  aplus:      $(gp.aplus)")
    println("  seed:       $(seed)")
    println()
    println("For model validation, compare posterior estimates to generating_params().")
end

main()