#!/usr/bin/env julia
# =============================================================================
# EFDM Runner — Run EFDM analysis from config + data CSV
# =============================================================================
# Usage:
#   julia scripts/run_efdm.jl config/model_config.toml data/my_counts.csv
#
# Or, if the config already specifies the data path:
#   julia scripts/run_efdm.jl config/model_config.toml
#
# This script:
#   1. Parses the TOML config
#   2. Loads and preprocesses the data CSV
#   3. Builds the design matrix and merges rare types
#   4. Runs Pigeons.jl PT sampling (optionally with auto-tune loop)
#   5. Prints convergence diagnostics + shrinkage analysis
#   6. Generates all diagnostic and effect plots + covariance heatmaps
#   7. Saves posterior summaries to CSV
# =============================================================================

using Pigeons, Distributions, Random, Statistics, Printf
using CSV, DataFrames, TOML

# Include the EFDMAnalysis package
script_dir = @__DIR__
project_root = abspath(joinpath(script_dir, ".."))
push!(LOAD_PATH, project_root)
include(joinpath(project_root, "src", "EFDMAnalysis.jl"))
using .EFDMAnalysis

# ─── 1. Parse command line ─────────────────────────────────────────────────
if length(ARGS) < 1
    println("Usage: julia run_efdm.jl <config.toml> [data.csv]")
    println()
    println("  config.toml — TOML configuration file (see config/model_config.toml)")
    println("  data.csv   — optional: override data path in config")
    exit(1)
end

config_path = abspath(ARGS[1])
config = TOML.parsefile(config_path)

data_path = if length(ARGS) ≥ 2
    abspath(ARGS[2])
else
    abspath(config["data"]["path"])
end

println("="^70)
println("EFDM ANALYSIS RUN")
println("="^70)
println("Config: $(config_path)")
println("Data:   $(data_path)")

# ─── 2. Load data ──────────────────────────────────────────────────────────
println("\n── Loading data ──")
df = CSV.read(data_path, DataFrame)

resp_cols = config["data"]["response_columns"]
if !all(c -> c ∈ names(df), resp_cols)
    missing_cols = filter(c -> !(c ∈ names(df)), resp_cols)
    error("Response columns not found in data: $missing_cols\nAvailable: $(names(df))")
end

Y_raw = Matrix{Int}(df[!, resp_cols])
N = size(Y_raw, 1)
D_raw = size(Y_raw, 2)

@printf "  %d observations, %d response categories\n" N D_raw

subject_col = get(config["data"], "subject_column", nothing)
donor_names = if subject_col !== nothing && subject_col ∈ names(df)
    string.(df[!, subject_col])
else
    ["Obs_$i" for i in 1:N]
end

cov_cols = config["data"]["covariate_columns"]
if !all(c -> c ∈ names(df), cov_cols)
    missing_cols = filter(c -> !(c ∈ names(df)), cov_cols)
    error("Covariate columns not found in data: $missing_cols\nAvailable: $(names(df))")
end

n_covariates = length(cov_cols)
X_raw = Matrix{Float64}(df[!, cov_cols])
K = n_covariates + 1
X = hcat(ones(N), X_raw)

covariate_names = vcat(["Intercept"], [string(c) for c in cov_cols])

# ─── 3. Merge rare types (optional) ───────────────────────────────────────
println("\n── Preprocessing ──")

if get(config["data"], "merge_rare_types", false)
    threshold = get(config["data"], "merge_threshold", 0.015)
    props = Y_raw ./ sum(Y_raw, dims=2)
    mean_prop = vec(mean(props, dims=1))
    keep = mean_prop .>= threshold
    n_keep = sum(keep)

    if n_keep < D_raw
        Y_merged = hcat(Y_raw[:, keep], sum(Y_raw[:, .!keep], dims=2))
        D = n_keep + 1
        @printf "  Merged %d rare types (<%.1f%%) → %d total types (incl. 'Other')\n" (D_raw - n_keep) (threshold * 100) D
    else
        Y_merged = Y_raw
        D = D_raw
        @printf "  No types below %.1f%% threshold. Keeping all %d types.\n" (threshold * 100) D
    end
else
    Y_merged = Y_raw
    D = D_raw
    @printf "  Keeping all %d response categories (merge_rare_types=false)\n" D
end

@printf "  Final dimensions: N=%d, D=%d, K=%d\n" N D K

# ─── 4. Extract priors from config ────────────────────────────────────────
println("\n── Priors ──")

beta_sd = config["priors"]["beta"]["sd"]
ap_log_mean = config["priors"]["aplus"]["log_mean"]
ap_log_sd = config["priors"]["aplus"]["log_sd"]

# Per-category support: config can supply either a scalar or a vector
function parse_sd(val, expected_len::Int)
    if val isa Number
        return fill(Float64(val), expected_len)
    elseif val isa Vector
        @assert length(val) == expected_len "Expected $expected_len values, got $(length(val))"
        return Float64.(val)
    else
        return fill(Float64(val), expected_len)
    end
end

p_alr_sd = parse_sd(config["priors"]["p"]["sd"], D - 1)
w_logit_sd = parse_sd(config["priors"]["w"]["sd"], D)

@printf "  β ~ Normal(0, %.1f²)\n" beta_sd
@printf "  log(aplus) ~ Normal(%.1f, %.1f²)\n" ap_log_mean ap_log_sd

p_sd_range = (minimum(p_alr_sd), maximum(p_alr_sd))
w_sd_range = (minimum(w_logit_sd), maximum(w_logit_sd))
if p_sd_range[1] ≈ p_sd_range[2]
    @printf "  ALR(p) ~ Normal(0, %.1f²)  (uniform)\n" p_sd_range[1]
else
    @printf "  ALR(p) ~ Normal(0, sd²)  (per-category, sd range [%.2f, %.2f])\n" p_sd_range[1] p_sd_range[2]
end
if w_sd_range[1] ≈ w_sd_range[2]
    @printf "  logit(w) ~ Normal(0, %.1f²)  (uniform)\n" w_sd_range[1]
else
    @printf "  logit(w) ~ Normal(0, sd²)  (per-category, sd range [%.2f, %.2f])\n" w_sd_range[1] w_sd_range[2]
end

# ─── 5. Build target and reference (closures for auto-tune) ──────────────
println("\n── Building target & reference ──")
n_params = efdm_n_params(D, K)
@printf "  Total parameters: %d (D=%d, K=%d)\n" n_params D K

function make_target(; p_alr_sd=nothing, w_logit_sd=nothing)
    p_sd = p_alr_sd === nothing ? config["priors"]["p"]["sd"] : p_alr_sd
    w_sd = w_logit_sd === nothing ? config["priors"]["w"]["sd"] : w_logit_sd
    return EFDMLogPotential(
        Float64.(Y_merged), X, D, K, beta_sd;
        aplus_log_mean=ap_log_mean, aplus_log_sd=ap_log_sd,
        p_alr_sd=p_sd, w_logit_sd=w_sd
    )
end

function make_ref(; p_alr_sd=nothing, w_logit_sd=nothing)
    p_sd = p_alr_sd === nothing ? config["priors"]["p"]["sd"] : p_alr_sd
    w_sd = w_logit_sd === nothing ? config["priors"]["w"]["sd"] : w_logit_sd
    return EFDMReference(
        D, K, beta_sd;
        aplus_log_mean=ap_log_mean, aplus_log_sd=ap_log_sd,
        p_alr_sd=p_sd, w_logit_sd=w_sd
    )
end

# ─── 6. Sampler configuration ─────────────────────────────────────────────
sampler_config = config["sampler"]
n_chains = sampler_config["n_chains"]
n_rounds = sampler_config["n_rounds"]
use_var = get(sampler_config, "use_variational", true)
multithreaded = get(sampler_config, "multithreaded", false)

explorer_str = get(sampler_config, "explorer", "SliceSampler")
explorer = if explorer_str == "SliceSampler"
    SliceSampler()
elseif explorer_str == "AutoMALA"
    AutoMALA()
else
    error("Unknown explorer: $explorer_str. Use 'SliceSampler' or 'AutoMALA'.")
end

# Build sampler kwargs (shared between single-run and auto-tune paths)
sampler_kwargs = (
    n_chains=n_chains, n_rounds=n_rounds,
    explorer=explorer, multithreaded=multithreaded,
    extended_traces=true,
    record=[traces; round_trip; index_process; record_default()],
)

# Build pt_kwargs for the single-run path. Use a Dict rather than a
# Vector{Pair} so that duplicate keys throw errors instead of silently
# overwriting.
pt_kwargs = Dict{Symbol, Any}(
    :target => make_target(),
    :reference => make_ref(),
    :n_chains => n_chains,
    :n_rounds => n_rounds,
    :explorer => explorer,
    :multithreaded => multithreaded,
    :extended_traces => true,
    :record => [traces; round_trip; index_process; record_default()],
)

if use_var
    ft_round = get(sampler_config, "first_tuning_round", 3)
    n_var_chains = get(sampler_config, "n_chains_variational", div(n_chains, 3))
    pt_kwargs[:variational] = GaussianReference(first_tuning_round=ft_round)
    pt_kwargs[:n_chains_variational] = n_var_chains
    @printf "  Variational PT: on (n_var=%d, first_tuning_round=%d)\n" n_var_chains ft_round
else
    @printf "  Variational PT: off (only recommended for testing)\n"
end

@printf "  n_chains=%d, n_rounds=%d, explorer=%s\n" n_chains n_rounds explorer_str
flush(stdout)

# ─── 6b. Output directory ─────────────────────────────────────────────────
output_dir = get(config, "output", Dict()) |> d -> get(d, "dir", "efdm_output")
mkpath(output_dir)
plots_dir = joinpath(output_dir, "plots")
mkpath(plots_dir)

# ─── 7. Auto-tune or single run ───────────────────────────────────────────
auto_tune_config = get(config, "auto_tune", Dict())
auto_tune_enabled = get(auto_tune_config, "enabled", false)

if auto_tune_enabled
    println("\n" * "="^70)
    println("EMPIRICAL BAYES AUTO-TUNE LOOP")
    println("="^70)

    n_iter = get(auto_tune_config, "n_iter", 3)
    inflation = get(auto_tune_config, "inflation", 1.5)

    target_fn = (; p_alr_sd, w_logit_sd) -> begin
        t = make_target(p_alr_sd=p_alr_sd, w_logit_sd=w_logit_sd)
        r = make_ref(p_alr_sd=p_alr_sd, w_logit_sd=w_logit_sd)
        return t, r
    end

    pt_final, p_sd_final, w_sd_final, history = run_auto_tune(
        target_fn, (args...) -> target_fn(; args...)[2],
        sampler_kwargs,
        D=D, K=K, n_iter=n_iter, inflation=inflation,
        output_dir=output_dir,
        X=X, Y=Float64.(Y_merged),
        covariate_names=covariate_names,
    )
    pt = pt_final

    # Save tuned prior SDs
    tuned_path = joinpath(output_dir, "tuned_priors.csv")
    tuned_df = DataFrame(
        block=vcat(fill("w_logit", D), fill("p_alr", D - 1)),
        index=vcat(1:D, 1:(D - 1)),
        sd=vcat(w_sd_final, p_sd_final),
    )
    CSV.write(tuned_path, tuned_df)
    @info "Saved tuned per-category prior SDs: $tuned_path"
else
    println("\n── Running Pigeons.jl PT ──")
    pt = pigeons(; pt_kwargs...)
    println("\n")
end

# ─── 8. Diagnostics & Report ──────────────────────────────────────────────
summary = convergence_summary(pt)
report = convergence_report_str(summary)
println(report)

report_path = joinpath(output_dir, "convergence_report.txt")
write(report_path, report)

# ─── 9. Posterior summary & shrinkage ─────────────────────────────────────
# Posterior summary
model_n_params = efdm_n_params(D, K)  # model parameters only (Pigeons may append extras)
param_names = generate_param_names(D, K)
samples_raw = sample_array(pt)
samples = samples_raw[:, 1:model_n_params, :]  # drop non-model columns
tci = extract_target_chain_indices(pt)
post_df = posterior_summary_table(samples, param_names, tci)
post_csv = joinpath(output_dir, "posterior_summary.csv")
CSV.write(post_csv, post_df)
@info "Saved posterior summary: $post_csv"

# Shrinkage analysis
shr = shrinkage_report(pt, D, K, beta_sd, ap_log_sd,
                        p_alr_sd, w_logit_sd)
println(shrinkage_report_str(shr; D=D, K=K))
plot_shrinkage_barchart(shr; D=D, K=K, output_dir=plots_dir)

# ─── 10. Plots ────────────────────────────────────────────────────────────
plot_aplus_posterior(pt, D, K; output_dir=plots_dir)
plot_conditional_effects(pt, D, K, X, Float64.(Y_merged),
                          covariate_names; output_dir=plots_dir)
plot_rhat_summary(pt; output_dir=plots_dir)

# Trace plots for key parameters
key_params = Int[(D-1)*K + 1]
key_names = String["log_aplus"]
n_beta = (D-1) * K
for i in 1:min(n_beta, 4)
    push!(key_params, i)
    push!(key_names, "β[$i]")
end
plot_traces(pt, key_params, key_names; output_dir=plots_dir)

# ─── 11. Covariance heatmaps (optional) ───────────────────────────────────
do_cov_plots = get(auto_tune_config, "covariance_plots",
                    get(config, "output", Dict()) |> d -> get(d, "covariance_plots", true))
if do_cov_plots
    println("\n── Computing posterior covariance heatmaps ──")
    plot_all_covariances(pt, D, K, X, covariate_names; output_dir=plots_dir)
end

# ─── 12. aplus-specific summary ───────────────────────────────────────────
idx_aplus = (D - 1) * K + 1
log_aplus_draws = extract_parameter_draws(samples, idx_aplus, tci)
aplus_draws = exp.(min.(log_aplus_draws, 700.0))
aplus_mean_val = mean(aplus_draws)
aplus_ci = quantile(aplus_draws, [0.025, 0.975])

@printf "\n  Overdispersion (aplus): mean=%.1f  median=%.1f  [%.1f, %.1f]\n" aplus_mean_val median(aplus_draws) aplus_ci[1] aplus_ci[2]

# ─── 13. Save PT result ───────────────────────────────────────────────────
save_path = joinpath(output_dir, "pt_result.jld2")
try
    @eval using JLD2
    JLD2.save(save_path,
              "pt", pt, "samples", samples,
              "tci", tci, "D", D, "K", K,
              "X", X, "Y_merged", Y_merged,
              "covariate_names", covariate_names)
    @info "Saved full PT result: $save_path"
catch
    @info "JLD2 not available. To save PT result later: Pkg.add(\"JLD2\")"
end

println("\nDone! All outputs in: $(abspath(output_dir))")
