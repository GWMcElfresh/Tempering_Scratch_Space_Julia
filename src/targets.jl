# =============================================================================
# EFDM Targets — Pigeons.jl Log-Potential and Reference Distributions
# =============================================================================
# These structs implement the LogDensityProblems.jl interface required by
# Pigeons.jl for parallel tempering.
#
# Both EFDMLogPotential and EFDMReference are callable and ForwardDiff-
# compatible, enabling GaussianReference variational PT.
# =============================================================================

# ─── Target: Log-posterior (unnormalized) ──────────────────────────────────────

"""
    EFDMLogPotential

Log-posterior density (up to constant) for the EFDM model.
Serves as the `target` distribution in Pigeons.jl PT.

Fields:
  Y              — N×D integer count matrix
  X              — N×K design matrix (covariates)
  n_obs          — N-vector of row sums of Y (pre-computed)
  D, K           — dimensions (response categories, covariates)
  beta_sd        — prior SD for regression coefficients
  aplus_log_mean — prior mean for log(aplus)
  aplus_log_sd   — prior SD for log(aplus)
  p_alr_sd       — prior SD for ALR-transformed p
  w_logit_sd     — prior SD for logit-transformed w
"""
struct EFDMLogPotential
    Y::Matrix{Float64}
    X::Matrix{Float64}
    n_obs::Vector{Float64}
    D::Int
    K::Int
    beta_sd::Float64
    aplus_log_mean::Float64
    aplus_log_sd::Float64
    p_alr_sd::Float64
    w_logit_sd::Float64
end

"""
    EFDMLogPotential(Y, X, D, K, beta_sd;
                     aplus_log_mean=3.5, aplus_log_sd=1.0,
                     p_alr_sd=2.0, w_logit_sd=2.0)

Construct an EFDMLogPotential from count matrix Y and design matrix X.

Recommended defaults (from lessons_learned.txt):
  - beta_sd = 1.0 (≤ 1.0 to avoid excessive prior-posterior gaps)
  - aplus_log_mean = 3.5  (prior belief: a≈33, typical for PBMC)
  - aplus_log_sd = 1.0    (moderately informative)
  - p_alr_sd = 2.0        (weakly informative on simplex)
  - w_logit_sd = 2.0      (weakly informative on (0,1) scale)
"""
function EFDMLogPotential(Y::Matrix{Float64}, X::Matrix{Float64},
                          D::Int, K::Int, beta_sd::Float64;
                          aplus_log_mean::Float64=3.5, aplus_log_sd::Float64=1.0,
                          p_alr_sd::Float64=2.0, w_logit_sd::Float64=2.0)
    n_obs = vec(sum(Y; dims=2))
    return EFDMLogPotential(Y, X, n_obs, D, K,
                            beta_sd, aplus_log_mean, aplus_log_sd,
                            p_alr_sd, w_logit_sd)
end

function (lp::EFDMLogPotential)(θ::AbstractVector)
    D, K = lp.D, lp.K
    beta_mat, aplus, p, w_norm = unpack_efdm(θ, D, K)
    mu_i = similar(p)
    ll = zero(eltype(p))

    for i in 1:length(lp.n_obs)
        softmax_linear_row!(mu_i, @view(lp.X[i, :]), beta_mat)
        ll += efdm_loglik_obs(@view(lp.Y[i, :]), lp.n_obs[i], mu_i, p, w_norm, aplus)
    end

    result = ll + efdm_logprior(θ, lp.D, lp.K,
                                lp.beta_sd, lp.aplus_log_mean, lp.aplus_log_sd,
                                lp.p_alr_sd, lp.w_logit_sd)
    return isnan(result) ? oftype(result, -Inf) : result
end

# LogDensityProblems.jl interface
LogDensityProblems.capabilities(::Type{<:EFDMLogPotential}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(lp::EFDMLogPotential) = efdm_n_params(lp.D, lp.K)
LogDensityProblems.logdensity(lp::EFDMLogPotential, θ) = lp(θ)

function LogDensityProblems.logdensity_and_gradient(lp::EFDMLogPotential, θ)
    result = ForwardDiff.DiffResults.GradientResult(θ)
    ForwardDiff.gradient!(result, lp, θ)
    return ForwardDiff.DiffResults.value(result), ForwardDiff.DiffResults.gradient(result)
end

# Pigeons.jl initialization and prior sampling
function Pigeons.initialization(lp::EFDMLogPotential, rng::AbstractRNG, ::Int)
    D, K = lp.D, lp.K
    θ = randn(rng, efdm_n_params(D, K)) .* 0.3
    θ[(D - 1) * K + 1] = lp.aplus_log_mean + randn(rng) * 0.3
    return θ
end

function Pigeons.sample_iid!(lp::EFDMLogPotential, replica, ::Any)
    fill_efdm_prior_sample!(replica.state, lp.D, lp.K,
                            lp.beta_sd, lp.aplus_log_mean, lp.aplus_log_sd,
                            lp.p_alr_sd, lp.w_logit_sd, replica.rng)
end

# ─── Reference: Prior distribution ─────────────────────────────────────────────

"""
    EFDMReference

Prior log-density for the EFDM model.
Serves as the `reference` distribution in Pigeons.jl PT.
"""
struct EFDMReference
    D::Int
    K::Int
    beta_sd::Float64
    aplus_log_mean::Float64
    aplus_log_sd::Float64
    p_alr_sd::Float64
    w_logit_sd::Float64
end

function EFDMReference(D::Int, K::Int, beta_sd::Float64;
                       aplus_log_mean::Float64=3.5, aplus_log_sd::Float64=1.0,
                       p_alr_sd::Float64=2.0, w_logit_sd::Float64=2.0)
    return EFDMReference(D, K, beta_sd,
                         aplus_log_mean, aplus_log_sd,
                         p_alr_sd, w_logit_sd)
end

(ref::EFDMReference)(θ::AbstractVector) =
    efdm_logprior(θ, ref.D, ref.K,
                  ref.beta_sd, ref.aplus_log_mean, ref.aplus_log_sd,
                  ref.p_alr_sd, ref.w_logit_sd)

# LogDensityProblems.jl interface
LogDensityProblems.capabilities(::Type{<:EFDMReference}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(ref::EFDMReference) = efdm_n_params(ref.D, ref.K)
LogDensityProblems.logdensity(ref::EFDMReference, θ) = ref(θ)

# Pigeons.jl initialization and prior sampling
function Pigeons.initialization(ref::EFDMReference, rng::AbstractRNG, ::Int)
    θ = randn(rng, efdm_n_params(ref.D, ref.K)) .* ref.beta_sd
    θ[(ref.D - 1) * ref.K + 1] = ref.aplus_log_mean + randn(rng) * ref.aplus_log_sd
    return θ
end

function Pigeons.sample_iid!(ref::EFDMReference, replica, ::Any)
    fill_efdm_prior_sample!(replica.state, ref.D, ref.K,
                            ref.beta_sd, ref.aplus_log_mean, ref.aplus_log_sd,
                            ref.p_alr_sd, ref.w_logit_sd, replica.rng)
end
