# =============================================================================
# EFDM Auto-Tune — Covariance, Shrinkage, and Empirical Bayes
# =============================================================================
# This module provides:
#   1. Extraction of the variational posterior (mean + precision) from the
#      GaussianReference fitted during Pigeons.jl PT.
#   2. Shrinkage diagnostics: how much does each parameter block shrink
#      relative to its prior? This reveals which categories need more or
#      less regularization.
#   3. Posterior covariance matrices in the constrained space:
#      - logit(w):  D×D, which categories share similar mixture behavior
#      - ALR(p):   (D-1)×(D-1), substitution patterns in baseline composition
#      - mu:        D×D, compositional substitution in expected proportions
#   4. An empirical Bayes auto-tune loop that iteratively tunes per-category
#      prior SDs to match the posterior variance, adapting to the data.
# =============================================================================

using LinearAlgebra

# ─── Variational posterior extraction ─────────────────────────────────────────

"""
    extract_variational_parameters(pt)

Extract the mean vector and precision matrix from the GaussianReference
distribution fitted during Pigeons.jl PT.

Returns: (mean, precision) or (nothing, nothing) if unavailable.

Note: Access paths depend on the Pigeons.jl version. Known paths are tried
in order of likelihood.
"""
function extract_variational_parameters(pt)
    tempering = pt.shared.tempering
    if !hasproperty(tempering, :variational_leg)
        @warn "No variational leg found — did you use GaussianReference?"
        return nothing, nothing
    end
    var_leg = tempering.variational_leg
    log_pots = var_leg.log_potentials
    n_var = length(log_pots)
    if n_var == 0
        @warn "Variational leg is empty."
        return nothing, nothing
    end
    lp = log_pots[n_var]

    # Try common field names across Pigeons versions
    mean_candidate = nothing
    prec_candidate = nothing
    for field in [:mean, :μ, :mu, :location]
        if hasproperty(lp, field)
            mean_candidate = getproperty(lp, field)
            break
        end
    end
    for field in [:precision, :P, :prec, :inv_cov]
        if hasproperty(lp, field)
            prec_candidate = getproperty(lp, field)
            break
        end
    end
    for field in [:covariance, :Σ, :sigma, :cov]
        if hasproperty(lp, field) && prec_candidate === nothing
            prec_candidate = inv(Matrix(getproperty(lp, field)))
            break
        end
    end

    if mean_candidate === nothing || prec_candidate === nothing
        @warn "Could not extract variational parameters from GaussianReference. " *
              "Known fields: $(propertynames(lp))"
        return nothing, nothing
    end

    return vec(mean_candidate), Matrix(prec_candidate)
end

"""
    sample_variational_posterior(pt, n_draws=1000)

Draw samples from the variational posterior (Gaussian approximation).
Returns an (n_draws × n_params) matrix in unconstrained θ space.

Falls back to actual PT samples if variational parameters are unavailable.
"""
function sample_variational_posterior(pt, n_draws::Int=1000)
    μ, P = extract_variational_parameters(pt)
    if μ === nothing || P === nothing
        @warn "Variational parameters unavailable. Using PT posterior samples instead."
        samples = sample_array(pt)
        tci = extract_target_chain_indices(pt)
        n_iter = size(samples, 1)
        n_chain = length(tci)
        all_draws = reshape(samples[:, :, tci], n_iter * n_chain, :)
        idxs = rand(1:size(all_draws, 1), n_draws)
        return all_draws[idxs, :]
    end

    n_params = length(μ)
    # Precision → covariance
   Σ = inv(Symmetric(P))
    # Draw from multivariate normal
    rng = MersenneTwister(42)
    raw = randn(rng, n_params, n_draws)
    L = cholesky(Σ).L
    return hcat(μ) .+ L * raw  # (n_params × n_draws) → transpose for (n_draws × n_params)
end

# ─── Shrinkage ─────────────────────────────────────────────────────────────────

"""
    compute_shrinkage(posterior_variance, prior_variance)

Compute the shrinkage factor: 1 - posterior_variance / prior_variance.
- shrinkage → 1:  data dominates (posterior much tighter than prior)
- shrinkage → 0:  prior dominates (posterior ≈ prior width)
- shrinkage < 0:  posterior wider than prior (unusual, suggests prior-data conflict)

Can be scalar or element-wise vector.
"""
function compute_shrinkage(post_var::Real, prior_var::Real)
    return 1.0 - post_var / prior_var
end
compute_shrinkage(post_var::AbstractVector, prior_var::AbstractVector) =
    [compute_shrinkage(pv, pv_prior) for (pv, pv_prior) in zip(post_var, prior_var)]

"""
    shrinkage_report(pt, D, K, beta_sd, aplus_log_sd, p_alr_sd, w_logit_sd;
                     n_draws=2000)

Compute per-element shrinkage factors for every parameter block.

Returns a NamedTuple with fields:
  beta      — vector of length (D-1)*K
  aplus     — scalar
  p_alr     — vector of length D-1
  w_logit   — vector of length D

Each value is the shrinkage factor (0 = prior dominant, 1 = data dominant).
"""
function shrinkage_report(pt, D::Int, K::Int,
                          beta_sd::Real, aplus_log_sd::Real,
                          p_alr_sd::AbstractVector{<:Real},
                          w_logit_sd::AbstractVector{<:Real};
                          n_draws::Int=2000)

    # Get posterior draws (prefer variational for speed, fall back to PT)
    θ_draws = sample_variational_posterior(pt, n_draws)

    # Use model-parameter count derived from D, K rather than from the sample
    # array, which may include Pigeons auxiliary columns.
    n_params = efdm_n_params(D, K)

    # Posterior variance in unconstrained space (only model parameters)
    post_var = [var(@view(θ_draws[:, p])) for p in 1:n_params]

    # Prior variances
    n_beta = (D - 1) * K
    prior_var_beta = fill(beta_sd^2, n_beta)
    prior_var_aplus = fill(aplus_log_sd^2, 1)
    p_sd_vec = p_alr_sd isa AbstractVector ? Float64.(p_alr_sd) : fill(Float64(p_alr_sd), D - 1)
    w_sd_vec = w_logit_sd isa AbstractVector ? Float64.(w_logit_sd) : fill(Float64(w_logit_sd), D)

    prior_var = vcat(prior_var_beta, prior_var_aplus,
                     p_sd_vec.^2, w_sd_vec.^2)

    @assert length(prior_var) == n_params "Mismatch: $(length(prior_var)) priors vs $n_params params"

    shrinkage = compute_shrinkage(post_var, prior_var)

    # Split into blocks
    idx_end = n_beta
    shr_beta = shrinkage[1:n_beta]
    shr_aplus = shrinkage[n_beta + 1]
    shr_p = shrinkage[n_beta + 2 : n_beta + D]
    shr_w = shrinkage[n_beta + D + 1 : n_beta + 2 * D]

    return (
        beta=shr_beta,
        aplus=shr_aplus,
        p_alr=shr_p,
        w_logit=shr_w,
        post_sd=sqrt.(post_var)
    )
end

"""
    shrinkage_report_str(report; D, K)

Format a shrinkage report as a readable string.
"""
function shrinkage_report_str(report; D::Int, K::Int)
    lines = String[]
    push!(lines, "="^60)
    push!(lines, "SHRINKAGE ANALYSIS (per-category regularization)")
    push!(lines, "  shrinkage = 1 - posterior_var / prior_var")
    push!(lines, "  → 1 means data dominates, → 0 means prior dominates")
    push!(lines, "="^60)

    push!(lines, "\nβ (regression coefficients):")
    push!(lines, @sprintf("  mean shrinkage: %.3f  range: [%.3f, %.3f]",
                          mean(report.beta), minimum(report.beta), maximum(report.beta)))

    push!(lines, @sprintf("\naplus (overdispersion):  %.3f", report.aplus))

    push!(lines, "\np (baseline weights, ALR scale):")
    for j in 1:(D - 1)
        push!(lines, @sprintf("  p[%d]:  %.3f", j, report.p_alr[j]))
    end

    push!(lines, "\nw (component weights, logit scale):")
    for j in 1:D
        push!(lines, @sprintf("  w[%d]:  %.3f", j, report.w_logit[j]))
    end

    low_shrinkage = findall(x -> x < 0.3, report.w_logit)
    if length(low_shrinkage) > 0
        push!(lines, "\nNOTE: w[$low_shrinkage] have shrinkage < 0.3 — prior is dominating.")
        push!(lines, "  Consider loosening w_sd for these categories (auto_tune will handle this).")
    end

    high_shrinkage = findall(x -> x > 0.99, report.w_logit)
    if length(high_shrinkage) > 0
        push!(lines, "\nNOTE: w[$high_shrinkage] have shrinkage > 0.99 — data dominates entirely.")
        push!(lines, "  Consider tightening w_sd if these estimates seem unstable.")
    end

    push!(lines, "="^60)
    return join(lines, "\n")
end

# ─── Posterior covariance in constrained space ─────────────────────────────────

"""
    posterior_logit_w_covariance(samples, D, K, tci)

Compute the posterior covariance matrix of the logit-transformed
component weights w, i.e. the D×D covariance of θ_w.

Returns: (cov_matrix, labels) where cov_matrix is D×D and labels are strings.
"""
function posterior_logit_w_covariance(samples, D::Int, K::Int, tci)
    n_beta = (D - 1) * K
    base_w = n_beta + 2 + (D - 1)
    n_iter = size(samples, 1)

    # Pool all cold-chain draws for w parameters
    w_draws = Matrix{Float64}(undef, n_iter * length(tci), D)
    off = 0
    for ci in tci
        for j in 1:D
            w_draws[off + 1 : off + n_iter, j] = samples[:, base_w + j - 1, ci]
        end
        off += n_iter
    end

    Σ = cov(w_draws)
    labels = ["w[$j]" for j in 1:D]
    return Σ, labels
end

"""
    posterior_alr_p_covariance(samples, D, K, tci)

Compute the posterior covariance matrix of the ALR-transformed
baseline mixture weights. Returns ((D-1) × (D-1)) covariance of
ALR(p[1..D-1]).
"""
function posterior_alr_p_covariance(samples, D::Int, K::Int, tci)
    n_beta = (D - 1) * K
    base_p = n_beta + 2
    n_iter = size(samples, 1)

    p_draws = Matrix{Float64}(undef, n_iter * length(tci), D - 1)
    off = 0
    for ci in tci
        for j in 1:(D - 1)
            p_draws[off + 1 : off + n_iter, j] = samples[:, base_p + j - 1, ci]
        end
        off += n_iter
    end

    Σ = cov(p_draws)
    labels = ["p[$j]" for j in 1:(D - 1)]
    return Σ, labels
end

"""
    posterior_mu_covariance(samples, X, D, K, tci; n_draws=500)

Compute the posterior covariance of the expected proportions μ = softmax(Xβ)
across categories. This is a D×D covariance representing the compositional
relationships: "when category j goes up, category k goes down."

To avoid computing this for every posterior draw (potentially very large),
a random subset of `n_draws` parameter vectors is used.

Returns: (cov_matrix_DxD, labels)
"""
function posterior_mu_covariance(samples, X::Matrix{Float64},
                                 D::Int, K::Int, tci;
                                 n_draws::Int=500)
    n_iter = size(samples, 1)
    n_chains = length(tci)
    n_total = n_iter * n_chains
    N_obs = size(X, 1)

    # Subsample draws
    idxs = rand(1:n_total, n_draws)
    N_all = Matrix{Float64}(undef, n_draws * N_obs, D)
    buf = Vector{Float64}(undef, D)
    off = 0
    for (di, global_idx) in enumerate(idxs)
        # Map global_idx → (chain_idx_in_tci, iteration)
        chain_idx_in_list = div(global_idx - 1, n_iter) + 1
        local_it = mod1(global_idx, n_iter)
        actual_chain = tci[chain_idx_in_list]
        θ = @view(samples[local_it, :, actual_chain])
        beta_mat = unpack_efdm(θ, D, K)[1]
        for i in 1:N_obs
            softmax_linear_row!(buf, @view(X[i, :]), beta_mat)
            N_all[off + 1, :] = buf'
            off += 1
        end
    end
    Σ = cov(N_all[1:off, :])
    labels = ["CT$(j-1)" for j in 1:D]
    return Σ, labels
end

"""
    posterior_mu_covariance_cond(samples, X, D, K, tci;
                                 n_draws=500, cond_on="mean")

Conditional version: compute mu covariance only at specific covariate values.
- "mean" — hold all covariates at their column means (average composition)
- "all"  — average over all observations (default)

Otherwise, provide a 1×K row vector of covariate values.
"""
function posterior_mu_covariance_cond(samples, X::Matrix{Float64},
                                       D::Int, K::Int, tci;
                                       n_draws::Int=500,
                                       cond_on="mean")
    if cond_on == "mean"
        x_row = vec(mean(X, dims=1))'
    elseif cond_on == "all"
        return posterior_mu_covariance(samples, X, D, K, tci; n_draws=n_draws)
    else
        x_row = cond_on
    end

    n_iter = size(samples, 1)
    n_total = n_iter * length(tci)
    idxs = rand(1:n_total, n_draws)

    mu_draws = Matrix{Float64}(undef, n_draws, D)
    buf = Vector{Float64}(undef, D)
    for (di, global_idx) in enumerate(idxs)
        chain_idx_in_list = div(global_idx - 1, n_iter) + 1
        local_it = mod1(global_idx, n_iter)
        actual_chain = tci[chain_idx_in_list]
        θ = @view(samples[local_it, :, actual_chain])
        beta_mat = unpack_efdm(θ, D, K)[1]
        softmax_linear_row!(buf, vec(x_row), beta_mat)
        mu_draws[di, :] = buf
    end

    Σ = cov(mu_draws)
    labels = ["CT$(j-1)" for j in 1:D]
    return Σ, labels
end

# ─── Empirical Bayes auto-tune ────────────────────────────────────────────────

"""
    auto_tune_round(pt, D, K; inflation=1.5)

Extract posterior marginal SDs from a completed PT run and compute
updated per-category prior SDs as:

    new_w_sd[j] = posterior_sd(logit(w[j])) × inflation
    new_p_sd[j] = posterior_sd(ALR(p[j])) × inflation

The inflation factor prevents overfitting to the current posterior
(which would collapse the prior to the posterior and remove regularization).

Returns: (new_p_sd, new_w_sd) — each a Float64 vector.
"""
function auto_tune_round(pt, D::Int, K::Int; inflation::Real=1.5)
    samples = sample_array(pt)
    tci = extract_target_chain_indices(pt)
    n_beta = (D - 1) * K
    n_iter = size(samples, 1)

    # Posterior SD of logit(w[j])
    base_w = n_beta + 2 + (D - 1)
    w_sd = zeros(D)
    for j in 1:D
        draws = Float64[]
        for ci in tci
            append!(draws, samples[:, base_w + j - 1, ci])
        end
        w_sd[j] = std(draws) * inflation
    end

    # Posterior SD of ALR(p[j])
    base_p = n_beta + 2
    p_sd = zeros(D - 1)
    for j in 1:(D - 1)
        draws = Float64[]
        for ci in tci
            append!(draws, samples[:, base_p + j - 1, ci])
        end
        p_sd[j] = std(draws) * inflation
    end

    # Floor: never go below 0.1 (maintain some regularization)
    clamp!(p_sd, 0.1, Inf)
    clamp!(w_sd, 0.1, Inf)

    return p_sd, w_sd
end

"""
    auto_tune_report_str(round, p_sd, w_sd)

Format the results of an auto-tune round.
"""
function auto_tune_report_str(round::Int, p_sd::Vector{Float64}, w_sd::Vector{Float64})
    lines = String[]
    push!(lines, "─"^60)
    push!(lines, "Auto-tune round $round — new per-category prior SDs")
    push!(lines, "─"^60)
    push!(lines, "  ALR(p) sd:")
    for j in eachindex(p_sd)
        push!(lines, @sprintf("    p[%d]:  %.3f", j, p_sd[j]))
    end
    push!(lines, "  logit(w) sd:")
    for j in eachindex(w_sd)
        push!(lines, @sprintf("    w[%d]:  %.3f", j, w_sd[j]))
    end
    push!(lines, "─"^60)
    return join(lines, "\n")
end

"""
    run_auto_tune(target_fn, reference_fn, sampler_kwargs;
                  D, K, n_iter=3, inflation=1.5, output_dir="efdm_output",
                  X=Matrix{Float64}(undef,0,0), Y=Matrix{Float64}(undef,0,0),
                  covariate_names=String[])

Run the full empirical Bayes auto-tune loop:

  1. Run PT with uniform priors (default scalar SDs)
  2. Extract posterior marginal SD of logit(w[j]) and ALR(p[j])
  3. Set new SDs = posterior_SD × inflation
  4. Re-run PT with per-category SDs
  5. Repeat until SDs stabilize

`target_fn(p_alr_sd=p_sd, w_logit_sd=w_sd)` and `reference_fn(...)` should be
closures that return (target, reference) given prior SD vectors.

`sampler_kwargs` is a NamedTuple of Pigeons.jl sampler settings.

`X`, `Y`, `covariate_names` are optional — if provided, diagnostic plots are
generated for each round.

Returns: (pt_final, p_sd_final, w_sd_final, history)
"""
function run_auto_tune(target_fn, reference_fn, sampler_kwargs;
                       D::Int, K::Int,
                       n_iter::Int=3,
                       inflation::Real=1.5,
                       output_dir::String="efdm_output",
                       X::Matrix{Float64}=Matrix{Float64}(undef,0,0),
                       Y::Matrix{Float64}=Matrix{Float64}(undef,0,0),
                       covariate_names::AbstractVector{<:AbstractString}=String[])

    mkpath(output_dir)
    history_rows = []

    # Round 0: start with uniform priors
    p_sd = fill(2.0, D - 1)
    w_sd = fill(2.0, D)

    for round in 1:n_iter
        println("\n" * "="^70)
        println("AUTO-TUNE ROUND $round / $n_iter")
        println("="^70)

        # Build target and reference with current priors
        target, reference = target_fn(p_alr_sd=p_sd, w_logit_sd=w_sd)

        # Run PT
        pt = pigeons(; target=target, reference=reference,
                      sampler_kwargs...)

        # Print convergence diagnostics
        summary = convergence_summary(pt)
        println(convergence_report_str(summary))

        # Print shrinkage
        shr = shrinkage_report(pt, D, K, 1.0, 1.0, p_sd, w_sd)
        println(shrinkage_report_str(shr; D=D, K=K))

        # Compute next-iteration prior SDs
        new_p_sd, new_w_sd = auto_tune_round(pt, D, K; inflation=inflation)
        println(auto_tune_report_str(round, new_p_sd, new_w_sd))

        # Log history
        push!(history_rows, (
            round=round,
            p_sd_mean=mean(p_sd), p_sd_min=minimum(p_sd), p_sd_max=maximum(p_sd),
            w_sd_mean=mean(w_sd), w_sd_min=minimum(w_sd), w_sd_max=maximum(w_sd),
            Λ=summary.Λ, restarts=summary.restarts, max_rhat=summary.max_rhat,
        ))

        p_sd, w_sd = new_p_sd, new_w_sd

        # Save intermediate results if we have plotting data
        if size(X, 1) > 0
            round_dir = joinpath(output_dir, "round_$round")
            mkpath(round_dir)
            round_plots_dir = joinpath(round_dir, "plots")
            mkpath(round_plots_dir)
            plot_all_diagnostics(pt, D, K, X, Y,
                                 covariate_names; output_dir=round_plots_dir)
        end
    end

    # N+1: final run with tuned priors
    println("\n" * "="^70)
    println("AUTO-TUNE FINAL RUN")
    println("="^70)

    target, reference = target_fn(p_alr_sd=p_sd, w_logit_sd=w_sd)
    pt = pigeons(; target=target, reference=reference, sampler_kwargs...)

    history_df = DataFrame(history_rows)
    history_csv = joinpath(output_dir, "auto_tune_history.csv")
    CSV.write(history_csv, history_df)
    @info "Auto-tune history saved: $history_csv"

    return pt, p_sd, w_sd, history_df
end
