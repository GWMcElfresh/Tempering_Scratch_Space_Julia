# =============================================================================
# EFDM Diagnostics — Post-Sampling Convergence Analysis
# =============================================================================
# Functions for extracting and summarizing MCMC samples from a Pigeons.jl PT
# run, computing convergence metrics, and producing structured reports.
# =============================================================================

"""
    ConvergenceSummary

Structured result from `convergence_summary()` containing the key convergence
diagnostics for an EFDM PT run.
"""
struct ConvergenceSummary
    Λ::Float64
    Λ_var::Union{Float64, Nothing}
    restarts::Int
    round_trips::Int
    max_rhat::Float64
    min_ess::Float64
    n_params::Int
    n_iterations::Int
    n_cold_chains::Int
end

"""
    extract_target_chain_indices(pt)

Find the chain indices corresponding to the cold (target-distributed) chains
in a Pigeons.jl PT output. Handles both standard and variational PT ladders.

Returns a sorted vector of integer indices.
"""
function extract_target_chain_indices(pt)
    tempering = pt.shared.tempering

    if hasproperty(tempering, :indexer) &&
       hasproperty(tempering, :fixed_leg) &&
       hasproperty(tempering, :variational_leg)

        n_var = length(tempering.variational_leg.log_potentials)
        n_fixed = length(tempering.fixed_leg.log_potentials)
        indices = Int[]

        for (idx, tup) in pairs(tempering.indexer.i2t)
            if (tup.leg == :variational && tup.chain == n_var) ||
               (tup.leg == :fixed && tup.chain == n_fixed)
                push!(indices, idx)
            end
        end
        sort!(indices)
        return indices
    end

    # Fallback: no variational leg — chain 1 is the cold chain
    return [1]
end

"""
    extract_parameter_draws(samples, param_idx, chain_indices)

Pool all MCMC draws for a single parameter across multiple cold chains.
Returns a flat Float64 vector of combined draws.

`samples` — the 3D array from `sample_array(pt)` (iter × params × chains)
`param_idx` — which parameter column to extract (1-indexed)
`chain_indices` — from `extract_target_chain_indices(pt)`
"""
function extract_parameter_draws(samples::AbstractArray{<:Real, 3},
                                 param_idx::Int,
                                 chain_indices::AbstractVector{<:Integer})
    n_iter = size(samples, 1)
    n_chains = length(chain_indices)
    out = Vector{eltype(samples)}(undef, n_iter * n_chains)
    offset = 0
    for ci in chain_indices
        @views out[(offset + 1):(offset + n_iter)] .= samples[:, param_idx, ci]
        offset += n_iter
    end
    return out
end

"""
    convergence_summary(pt; chains=nothing)

Compute key convergence diagnostics from a Pigeons.jl PT result.

Returns a `ConvergenceSummary` struct with fields:
  Λ          — global barrier (full ladder)
  Λ_var      — variational barrier (if variational PT, else nothing)
  restarts   — number of tempered restarts
  round_trips — number of round trips
  max_rhat   — maximum Rhat across all parameters
  min_ess    — minimum ESS across all parameters
  n_params   — number of model parameters
  n_iterations — number of MCMC iterations per chain
  n_cold_chains — number of cold (target) chains

`chains` — optional pre-computed MCMCChains.Chains object; will be computed
            from pt if not provided.
"""
function convergence_summary(pt; chains=nothing)
    Λ = Pigeons.global_barrier(pt)
    restarts = Pigeons.n_tempered_restarts(pt)
    round_trips = Pigeons.n_round_trips(pt)

    # Try to get Λ_var from the variational leg
    Λ_var = nothing
    try
        Λ_var = Pigeons.global_barrier(pt, variant=:variational)
    catch
        # Variational leg may not exist
    end

    samples = sample_array(pt)
    n_iter = size(samples, 1)
    n_params = size(samples, 2)

    tci = extract_target_chain_indices(pt)

    if chains === nothing
        chain_obj = Chains(pt)
    else
        chain_obj = chains
    end

    rhat_vals = rhat(chain_obj)
    ess_vals = ess(chain_obj)

    # Handle DataFrame or vector output from MCMCChains
    rhat_vec = if rhat_vals isa DataFrame
        Float64[skipmissing(rhat_vals.nt.rhat)...]
    else
        Float64[skipmissing(rhat_vals)...]
    end
    ess_vec = if ess_vals isa DataFrame
        Float64[skipmissing(ess_vals.nt.ess)...]
    else
        Float64[skipmissing(ess_vals)...]
    end

    max_rhat = length(rhat_vec) > 0 ? maximum(rhat_vec) : 1.0
    min_ess = length(ess_vec) > 0 ? minimum(ess_vec) : 0.0

    return ConvergenceSummary(
        Λ, Λ_var, restarts, round_trips,
        max_rhat, min_ess,
        n_params, n_iter, length(tci)
    )
end

"""
    convergence_assessment(summary)

Return a Symbol classifying convergence quality: `:converged`, `:partial`,
or `:failed`.

Thresholds (from lessons_learned.txt, section 2D):
  Converged:   Λ ≤ 3, rst ≥ 10, max_Rhat < 1.05
  Partial:     rst > 0 and Λ ≤ 5
  Failed:      rst == 0 or Λ > 5 or (Λ_var !== nothing && Λ_var > 5)
"""
function convergence_assessment(summary::ConvergenceSummary)
    Λ = summary.Λ_var !== nothing ? summary.Λ_var : summary.Λ

    if Λ ≤ 3 && summary.restarts ≥ 10 && summary.max_rhat ≤ 1.05
        return :converged
    elseif summary.restarts > 0 && Λ ≤ 5
        return :partial
    else
        return :failed
    end
end

"""
    convergence_report_str(summary)

Return a formatted string with the "three numbers" (Λ, restarts, Rhat)
and convergence assessment, suitable for printing to stdout.
"""
function convergence_report_str(summary::ConvergenceSummary)
    lines = String[]
    push!(lines, "="^60)
    push!(lines, "CONVERGENCE DIAGNOSTICS")
    push!(lines, "="^60)

    @Λ_str = if summary.Λ_var !== nothing
        @sprintf "  Λ (full ladder):           %.3f\n  Λ_var (variational leg):  %.3f" summary.Λ summary.Λ_var
    else
        @sprintf "  Λ (global barrier):        %.3f" summary.Λ
    end
    push!(lines, @Λ_str)

    push!(lines, @sprintf "  Tempered restarts:        %d" summary.restarts)
    push!(lines, @sprintf "  Round trips:              %d" summary.round_trips)
    push!(lines, @sprintf "  Max Rhat:                 %.4f" summary.max_rhat)
    push!(lines, @sprintf "  Min ESS:                  %.1f" summary.min_ess)

    push!(lines, "")
    assessment = convergence_assessment(summary)
    if assessment == :converged
        push!(lines, "  Result: CONVERGED  (Λ≤3, rst≥10, Rhat<1.05)")
    elseif assessment == :partial
        push!(lines, "  Result: PARTIAL convergence  (rst>0, Λ≤5)")
        push!(lines, "  Recommend: increase n_rounds or n_chains")
    else
        push!(lines, "  Result: FAILED  (no restarts or barrier too high)")
        push!(lines, "  Recommend: add/check variational GaussianReference,")
        push!(lines, "            tighten priors (beta_sd ≤ 1.0), or increase chains")
    end

    push!(lines, "="^60)
    push!(lines, "  Parameters: $(summary.n_params)  Iterations: $(summary.n_iterations)")
    push!(lines, "  Cold chains: $(summary.n_cold_chains)")
    push!(lines, "="^60)

    return join(lines, "\n")
end

"""
    posterior_summary_table(samples, param_names, chain_indices)

Build a DataFrame with posterior mean, SD, and quantiles for every parameter.

`samples` — 3D array from `sample_array(pt)`
`param_names` — vector of parameter name strings (one per column)
`chain_indices` — from `extract_target_chain_indices(pt)`
"""
function posterior_summary_table(samples::AbstractArray{<:Real, 3},
                                 param_names::AbstractVector{<:AbstractString},
                                 chain_indices::AbstractVector{<:Integer})
    n_params = size(samples, 2)
    @assert length(param_names) == n_params "param_names length must match samples columns"

    rows = []
    for p in 1:n_params
        draws = extract_parameter_draws(samples, p, chain_indices)
        q = quantile(draws, [0.025, 0.25, 0.5, 0.75, 0.975])
        push!(rows, (
            param=param_names[p], index=p,
            mean=mean(draws), std=std(draws),
            q2_5=q[1], q25=q[2], q50=q[3], q75=q[4], q97_5=q[5]
        ))
    end

    return DataFrame(rows)
end

"""
    swap_acceptance_matrix(pt)

Extract swap acceptance probabilities between adjacent temperature rungs.
Returns a vector of acceptance rates.

NOTE: Pigeons does not expose this directly in a standard API. This function
attempts multiple extraction strategies, falling back to an empty vector.
"""
function swap_acceptance_matrix(pt)
    # Try to extract swap information from extended traces
    swaps = Float64[]
    try
        # Pigeons sometimes stores swap info in pt.recorder or pt.shared
        if hasproperty(pt, :recorder) && hasproperty(pt.recorder, :swap_rates)
            swaps = Float64[pt.recorder.swap_rates...]
        elseif hasproperty(pt, :shared) && hasproperty(pt.shared, :swap_rates)
            swaps = Float64[pt.shared.swap_rates...]
        end
    catch
    end
    return swaps
end

"""
    generate_param_names(D, K)

Generate descriptive parameter names given model dimensions.
Returns a vector of strings suitable for passing to `posterior_summary_table`.

Naming:
  β[r,c]   — regression coefficients (r=response cat, c=covariate)
  log_a    — log overdispersion
  p[j]     — baseline mixture weight j (ALR-transformed)
  w[j]     — component weight j (logit-transformed)
"""
function generate_param_names(D::Int, K::Int)
    names = String[]
    n_beta = (D - 1) * K

    for r in 1:(D - 1)
        for c in 1:K
            push!(names, "β[$(r),$(c)]")
        end
    end
    push!(names, "log_aplus")
    for j in 1:(D - 1)
        push!(names, "p[$j]")
    end
    for j in 1:D
        push!(names, "w[$j]")
    end

    return names
end

"""
    restore_parameters(θ, D, K)

Convert a single unconstrained parameter vector θ to structured parameters
and return a descriptive NamedTuple.

Returns (beta, aplus, p, w_norm).
"""
function restore_parameters(θ::AbstractVector, D::Int, K::Int)
    beta_mat, aplus, p, w_norm = unpack_efdm(θ, D, K)
    return (beta=beta_mat, aplus=aplus, p=p, w_norm=w_norm)
end
