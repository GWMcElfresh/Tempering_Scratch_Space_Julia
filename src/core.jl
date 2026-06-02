# =============================================================================
# EFDM Core Math — Pure Functions, No Side Effects
# =============================================================================
# This file contains the mathematical core of the Extended Flexible Dirichlet
# Multinomial (EFDM) model. Every function is pure: no I/O, no global state,
# no mutable structs beyond local scratch buffers.
#
# Parameter conventions (constants throughout the codebase):
#   N  = number of observations (donors/subjects)
#   D  = number of response categories (cell types / species)
#   K  = number of covariates (intercept + extra columns)
#   nβ = (D-1) * K  = number of regression coefficients
# =============================================================================

"""
    softplus_stable(z)

Numerically stable softplus: log(1 + exp(z)).
Avoids overflow for large positive z and underflow for large negative z.
"""
@inline function softplus_stable(z::Real)
    return z > 0 ? z + log1p(exp(-z)) : log1p(exp(z))
end

"""
    tanh_clamp(x; ε=0.01)

Smooth approximation of clamp(x, 0, 1) using a tanh-based soft clamp.
Replaces hard min/max for differentiability needed by ForwardDiff.
"""
@inline function tanh_clamp(x::Real; ε::Real=0.01)
    return 1.0 - ε * softplus_stable((1.0 - x) / ε)
end

"""
    efdm_n_params(D, K)

Total number of unconstrained parameters in the EFDM model.
Breakdown: (D-1)*K  regression β  +  1 log(aplus)  +  (D-1) ALR(p)  +  D logit(w)
"""
function efdm_n_params(D::Int, K::Int)
    return (D - 1) * K + 1 + (D - 1) + D
end

# ─── Priors ────────────────────────────────────────────────────────────────────

"""
    efdm_logprior(θ, D, K, beta_sd, aplus_log_mean, aplus_log_sd, p_alr_sd, w_logit_sd)

Joint log-prior density for the unconstrained parameter vector θ.
Each block has an independent Normal(0, sd²) prior in unconstrained space.

`p_alr_sd` and `w_logit_sd` can be either scalars (same SD for all elements)
or vectors (per-element SDs for category-specific regularization).
"""
function efdm_logprior(θ::AbstractVector, D::Int, K::Int,
                       beta_sd::Real, aplus_log_mean::Real, aplus_log_sd::Real,
                       p_alr_sd::AbstractVector{<:Real},
                       w_logit_sd::AbstractVector{<:Real})
    T = eltype(θ)
    n_beta = (D - 1) * K
    total = zero(T)

    inv_beta_var = inv(T(beta_sd)^2)
    @inbounds for i in 1:n_beta
        total -= T(0.5) * inv_beta_var * θ[i]^2
    end

    inv_ap_var = inv(T(aplus_log_sd)^2)
    diff_a = θ[n_beta + 1] - T(aplus_log_mean)
    total -= T(0.5) * inv_ap_var * diff_a^2

    @assert length(p_alr_sd) == D - 1 "p_alr_sd must have length D-1 ($(D-1)), got $(length(p_alr_sd))"
    base_alr = n_beta + 2
    @inbounds for (j, i) in enumerate(base_alr:(base_alr + D - 2))
        total -= T(0.5) * inv(T(p_alr_sd[j])^2) * θ[i]^2
    end

    @assert length(w_logit_sd) == D "w_logit_sd must have length D ($D), got $(length(w_logit_sd))"
    base_w = n_beta + 2 + (D - 1)
    @inbounds for (j, i) in enumerate(base_w:(base_w + D - 1))
        total -= T(0.5) * inv(T(w_logit_sd[j])^2) * θ[i]^2
    end

    return total
end

# Scalar convenience wrappers
function efdm_logprior(θ::AbstractVector, D::Int, K::Int,
                       beta_sd::Real, aplus_log_mean::Real, aplus_log_sd::Real,
                       p_alr_sd::Real, w_logit_sd::Real)
    return efdm_logprior(θ, D, K, beta_sd, aplus_log_mean, aplus_log_sd,
                         fill(p_alr_sd, D - 1), fill(w_logit_sd, D))
end

"""
    fill_efdm_prior_sample!(θ, D, K, beta_sd, aplus_log_mean, aplus_log_sd, p_alr_sd, w_logit_sd, rng)

Fill θ with an iid draw from the prior (in-place). Used by Pigeons.sample_iid!.

`p_alr_sd` and `w_logit_sd` can be either scalars or vectors, matching `efdm_logprior`.
"""
function fill_efdm_prior_sample!(θ::AbstractVector, D::Int, K::Int,
                                 beta_sd::Real, aplus_log_mean::Real, aplus_log_sd::Real,
                                 p_alr_sd::AbstractVector{<:Real},
                                 w_logit_sd::AbstractVector{<:Real}, rng::AbstractRNG)
    n_beta = (D - 1) * K
    @inbounds for i in 1:n_beta
        θ[i] = randn(rng) * beta_sd
    end
    θ[n_beta + 1] = aplus_log_mean + randn(rng) * aplus_log_sd
    @assert length(p_alr_sd) == D - 1
    base_alr = n_beta + 2
    @inbounds for (j, i) in enumerate(base_alr:(base_alr + D - 2))
        θ[i] = randn(rng) * p_alr_sd[j]
    end
    @assert length(w_logit_sd) == D
    base_w = n_beta + 2 + (D - 1)
    @inbounds for (j, i) in enumerate(base_w:(base_w + D - 1))
        θ[i] = randn(rng) * w_logit_sd[j]
    end
    return θ
end

function fill_efdm_prior_sample!(θ::AbstractVector, D::Int, K::Int,
                                 beta_sd::Real, aplus_log_mean::Real, aplus_log_sd::Real,
                                 p_alr_sd::Real, w_logit_sd::Real, rng::AbstractRNG)
    return fill_efdm_prior_sample!(θ, D, K, beta_sd, aplus_log_mean, aplus_log_sd,
                                   fill(p_alr_sd, D - 1), fill(w_logit_sd, D), rng)
end

# ─── Parameter unpacking ───────────────────────────────────────────────────────

"""
    unpack_efdm(θ, D, K)

Convert the unconstrained parameter vector θ into structured EFDM parameters.

Returns tuple: (beta, aplus, p, w_norm)
  - beta:   (D × K) matrix, last row fixed to zero (reference category)
  - aplus:  scalar > 0, overdispersion
  - p:      D-element simplex, baseline mixture weights
  - w_norm: D-element vector in (0, 1), component weights
"""
function unpack_efdm(θ::AbstractVector, D::Int, K::Int)
    idx = 1
    n_beta = (D - 1) * K

    beta_mat = Matrix{eltype(θ)}(undef, D, K)
    @inbounds for row in 1:(D - 1), col in 1:K
        beta_mat[row, col] = θ[idx]; idx += 1
    end
    @inbounds for col in 1:K
        beta_mat[D, col] = zero(eltype(θ))
    end

    aplus = exp(min(θ[idx], eltype(θ)(30))); idx += 1

    p = Vector{eltype(θ)}(undef, D)
    denom = one(aplus)
    @inbounds for j in 1:(D - 1)
        value = exp(min(θ[idx], eltype(θ)(30)))
        p[j] = value; denom += value; idx += 1
    end
    inv_denom = inv(denom)
    @inbounds for j in 1:(D - 1); p[j] *= inv_denom; end
    p[D] = inv_denom

    w_norm = Vector{eltype(θ)}(undef, D)
    clamp_eps = one(aplus) * 1e-4; upper = one(aplus) - clamp_eps
    @inbounds for j in 1:D
        w_norm[j] = clamp(inv(one(aplus) + exp(-θ[idx])), clamp_eps, upper); idx += 1
    end

    return beta_mat, aplus, p, w_norm
end

"""
    unpack_efdm_p!(buf, θ, D, K)

Fast unpack of only the p (mixture weights) from θ into pre-allocated buffer.
Used during diagnostic extraction when only p is needed (e.g., bimodality checks).
"""
function unpack_efdm_p!(buf::AbstractVector, θ::AbstractVector, D::Int, K::Int)
    idx = (D - 1) * K + 2
    denom = one(eltype(buf))
    @inbounds for j in 1:(D - 1)
        value = exp(min(θ[idx], eltype(buf)(30)))
        buf[j] = value; denom += value; idx += 1
    end
    inv_denom = inv(denom)
    @inbounds for j in 1:(D - 1); buf[j] *= inv_denom; end
    buf[D] = inv_denom
    return buf
end

# ─── Softmax helper ────────────────────────────────────────────────────────────

"""
    softmax_linear_row!(out, x, beta)

Compute softmax(xᵀ β) for a single observation. Fills `out` in-place.
Uses the max-subtraction trick for numerical stability.

  out[d] = exp(x ⋅ β[d]) / Σₖ exp(x ⋅ β[k])
"""
function softmax_linear_row!(out::AbstractVector, x::AbstractVector, beta_mat::AbstractMatrix)
    D = size(beta_mat, 1); K = size(beta_mat, 2)
    score = zero(eltype(out))

    @inbounds for k in 1:K; score += x[k] * beta_mat[1, k]; end
    out[1] = score; max_score = score

    @inbounds for d in 2:D
        score = zero(eltype(out))
        for k in 1:K; score += x[k] * beta_mat[d, k]; end
        out[d] = score
        if score > max_score; max_score = score; end
    end

    denom = zero(eltype(out))
    @inbounds for d in 1:D
        value = exp(out[d] - max_score); out[d] = value; denom += value
    end
    inv_denom = inv(denom)
    @inbounds for d in 1:D; out[d] *= inv_denom; end
    return out
end

# ─── Log-likelihood ────────────────────────────────────────────────────────────

"""
    efdm_loglik_obs(Y_i, n_i, mu_i, p, w_norm, aplus; ε=0.01)

Log-likelihood of a single multinomial observation under the EFDM model.

Y_i   — D-vector of integer counts for this observation
n_i   — total count (sum(Y_i))
mu_i  — D-vector of regression-driven proportions (from softmax of Xβ)
p     — D-vector of baseline mixture weights (simplex)
w_norm — D-vector of component weights in (0,1)
aplus — scalar overdispersion parameter > 0

The log-likelihood uses log-sum-exp for the mixture sum over categories,
and numerically stable log-gamma functions for the Dirichlet-Multinomial
components.
"""
function efdm_loglik_obs(Y_i::AbstractVector, n_i::Real,
                          mu_i::AbstractVector, p::AbstractVector,
                          w_norm::AbstractVector, aplus::Real; ε::Real=0.01)
    D = length(Y_i)

    # Compute t2 = Σ p[j] * w[j] * tanh_clamp(mu[j]/p[j])
    temp2 = 0.0
    for j in 1:D
        temp2 += p[j] * w_norm[j] * tanh_clamp(mu_i[j] / p[j]; ε=ε)
    end
    temp2 = min(temp2, 1.0 - 1e-8)

    # Log-likelihood with log-sum-exp (LSE) for numerical stability
    temp = loggamma(n_i + 1.0)
    lse_max = -Inf
    for j in 1:D
        wj = w_norm[j] * tanh_clamp(mu_i[j] / p[j]; ε=ε)
        αj = max(aplus * (mu_i[j] - p[j] * wj) / (1.0 - temp2), 1e-8)
        τj = max(aplus * wj / max(1.0 - wj, 1e-6), 1e-8)
        Yj = Float64(Y_i[j])

        temp += loggamma(αj + Yj) - loggamma(αj) - loggamma(Yj + 1.0)

        val = (log(p[j])
               + loggamma(aplus + τj)       - loggamma(aplus + n_i + τj)
               + loggamma(αj)               + loggamma(αj + τj + Yj)
               - loggamma(αj + τj)          - loggamma(αj + Yj))
        lse_max = max(lse_max, val)
    end

    lse_sum = 0.0
    for j in 1:D
        wj = w_norm[j] * tanh_clamp(mu_i[j] / p[j]; ε=ε)
        αj = max(aplus * (mu_i[j] - p[j] * wj) / (1.0 - temp2), 1e-8)
        τj = max(aplus * wj / max(1.0 - wj, 1e-6), 1e-8)
        Yj = Float64(Y_i[j])

        val = (log(p[j])
               + loggamma(aplus + τj)       - loggamma(aplus + n_i + τj)
               + loggamma(αj)               + loggamma(αj + τj + Yj)
               - loggamma(αj + τj)          - loggamma(αj + Yj))
        lse_sum += exp(val - lse_max)
    end

    return temp + lse_max + log(lse_sum)
end

# ─── Simulation ────────────────────────────────────────────────────────────────

"""
    simulate_efdm(rng, N, n_trials, X, beta_true, p_true, w_norm_true, aplus_true)

Generate synthetic EFDM data for testing and validation.

Returns: (Y, mu)
  Y  — N × D integer count matrix
  mu — N × D expected proportion matrix (before mixture with p)
"""
function simulate_efdm(rng::AbstractRNG, N::Int, n_trials::Int,
                       X::Matrix{Float64}, beta_true::Matrix{Float64},
                       p_true::Vector{Float64}, w_norm_true::Vector{Float64},
                       aplus_true::Real)
    D = length(p_true)
    Y = Matrix{Int}(undef, N, D)
    mu = Matrix{Float64}(undef, N, D)
    scratch_w = Vector{Float64}(undef, D)
    scratch_dir = Vector{Float64}(undef, D)
    probs = Vector{Float64}(undef, D)
    cat_dist = Categorical(p_true)
    ε_sim = 1e-6

    for i in 1:N
        mu_i = @view(mu[i, :])
        softmax_linear_row!(mu_i, @view(X[i, :]), beta_true)

        temp2 = 0.0
        for j in 1:D
            wj = clamp(w_norm_true[j] * min(1.0, mu_i[j] / p_true[j]), 0.0, 1.0 - ε_sim)
            scratch_w[j] = wj; temp2 += p_true[j] * wj
        end
        temp2 = clamp(temp2, 0.0, 1.0 - ε_sim)
        inv_scale = inv(1.0 - temp2)

        for j in 1:D
            base_mass = max(mu_i[j] - p_true[j] * scratch_w[j], 0.0)
            scratch_dir[j] = max(aplus_true * base_mass * inv_scale, ε_sim)
        end

        r = rand(rng, cat_dist)
        w_r = clamp(scratch_w[r], 0.0, 1.0 - ε_sim)
        tau_r = max(aplus_true * w_r / max(1.0 - w_r, ε_sim), ε_sim)
        scratch_dir[r] += tau_r

        total_mass = 0.0
        for j in 1:D
            draw = rand(rng, Gamma(max(scratch_dir[j], ε_sim), 1.0))
            probs[j] = draw; total_mass += draw
        end
        if total_mass > 0.0 && isfinite(total_mass)
            inv_total = inv(total_mass)
            for j in 1:D; probs[j] *= inv_total; end
        else
            fill!(probs, inv(D))
        end

        remaining = n_trials; remaining_mass = 1.0
        for j in 1:D
            if j == D; Y[i, j] = remaining; break; end
            if remaining <= 0; Y[i, j] = 0; continue; end
            mass = clamp(probs[j], 0.0, remaining_mass)
            prob = clamp(mass / remaining_mass, 0.0, 1.0)
            draw = rand(rng, Binomial(remaining, prob))
            Y[i, j] = draw; remaining -= draw; remaining_mass -= mass
        end
    end

    return Y, mu
end
