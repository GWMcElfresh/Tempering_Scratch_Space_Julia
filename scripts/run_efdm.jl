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
#   4. Runs Pigeons.jl PT sampling
#   5. Prints convergence diagnostics
#   6. Generates all diagnostic and effect plots
#   7. Saves posterior summaries to CSV
# =============================================================================

using Pigeons, Distributions, Random, Statistics, Printf
using CSV, DataFrames, TOML

# Include the EFDMAnalysis package
# Adjust path relative to script location
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

# Identify response columns
resp_cols = config["data"]["response_columns"]
if !all(c -> c ∈ names(df), resp_cols)
    missing_cols = filter(c -> !(c ∈ names(df)), resp_cols)
    error("Response columns not found in data: $missing_cols\nAvailable: $(names(df))")
end

Y_raw = Matrix{Int}(df[!, resp_cols])
N = size(Y_raw, 1)
D_raw = size(Y_raw, 2)

@printf "  %d observations, %d response categories\n" N D_raw

# Subject labels (optional)
subject_col = get(config["data"], "subject_column", nothing)
donor_names = if subject_col !== nothing && subject_col ∈ names(df)
    string.(df[!, subject_col])
else
    ["Obs_$i" for i in 1:N]
end

# Identify covariate columns
cov_cols = config["data"]["covariate_columns"]
if !all(c -> c ∈ names(df), cov_cols)
    missing_cols = filter(c -> !(c ∈ names(df)), cov_cols)
    error("Covariate columns not found in data: $missing_cols\nAvailable: $(names(df))")
end

# Build design matrix: intercept + covariates
n_covariates = length(cov_cols)
X_raw = Matrix{Float64}(df[!, cov_cols])
K = n_covariates + 1  # +1 for intercept
X = hcat(ones(N), X_raw)  # (N × K) design matrix

# Covariate names (for plotting)
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
p_alr_sd = config["priors"]["p"]["sd"]
w_logit_sd = config["priors"]["w"]["sd"]

@printf "  β ~ Normal(0, %.1f²)\n" beta_sd
@printf "  log(aplus) ~ Normal(%.1f, %.1f²)\n" ap_log_mean ap_log_sd
@printf "  ALR(p) ~ Normal(0, %.1f²)\n" p_alr_sd
@printf "  logit(w) ~ Normal(0, %.1f²)\n" w_logit_sd

# ─── 5. Build target and reference ────────────────────────────────────────
println("\n── Building target & reference ──")
n_params = efdm_n_params(D, K)
@printf "  Total parameters: %d (D=%d, K=%d)\n" n_params D K

target = EFDMLogPotential(
    Float64.(Y_merged), X, D, K, beta_sd;
    aplus_log_mean=ap_log_mean, aplus_log_sd=ap_log_sd,
    p_alr_sd=p_alr_sd, w_logit_sd=w_logit_sd
)

reference = EFDMReference(
    D, K, beta_sd;
    aplus_log_mean=ap_log_mean, aplus_log_sd=ap_log_sd,
    p_alr_sd=p_alr_sd, w_logit_sd=w_logit_sd
)

# ─── 6. Run Pigeons.jl PT ────────────────────────────────────────────────
println("\n── Running Pigeons.jl PT ──")

sampler_config = config["sampler"]
n_chains = sampler_config["n_chains"]
n_rounds = sampler_config["n_rounds"]
use_var = get(sampler_config, "use_variational", true)
multithreaded = get(sampler_config, "multithreaded", false)

# Parse explorer
explorer_str = get(sampler_config, "explorer", "SliceSampler")
explorer = if explorer_str == "SliceSampler"
    SliceSampler()
elseif explorer_str == "AutoMALA"
    AutoMALA()
else
    error("Unknown explorer: $explorer_str. Use 'SliceSampler' or 'AutoMALA'.")
end

# Build keyword arguments
pt_kwargs = [
    :target => target,
    :reference => reference,
    :n_chains => n_chains,
    :n_rounds => n_rounds,
    :explorer => explorer,
    :multithreaded => multithreaded,
    :extended_traces => true,
    :record => [traces; round_trip; index_process; record_default()],
]

if use_var
    ft_round = get(sampler_config, "first_tuning_round", 3)
    n_var_chains = get(sampler_config, "n_chains_variational", div(n_chains, 3))
    push!(pt_kwargs, :variational => GaussianReference(first_tuning_round=ft_round))
    push!(pt_kwargs, :n_chains_variational => n_var_chains)
    @printf "  Variational PT: on (n_var=%d, first_tuning_round=%d)\n" n_var_chains ft_round
else
    @printf "  Variational PT: off (only recommended for testing)\n"
end

@printf "  n_chains=%d, n_rounds=%d, explorer=%s\n" n_chains n_rounds explorer_str
flush(stdout)

pt = pigeons(; pt_kwargs...)

# ─── 7. Diagnostics & Report ──────────────────────────────────────────────
println("\n")

# Get output directory from config
output_dir = get(config, "output", Dict()) |> d -> get(d, "dir", "efdm_output")

# Run all diagnostics and plots
summary = plot_all_diagnostics(pt, D, K, X, Float64.(Y_merged),
                                covariate_names; output_dir=output_dir)

# ─── 8. Additional: aplus-specific summary ────────────────────────────────
samples = sample_array(pt)
tci = extract_target_chain_indices(pt)
idx_aplus = (D - 1) * K + 1
log_aplus_draws = extract_parameter_draws(samples, idx_aplus, tci)
aplus_draws = exp.(min.(log_aplus_draws, 700.0))
aplus_mean = mean(aplus_draws)
aplus_ci = quantile(aplus_draws, [0.025, 0.975])

@printf "\n  Overdispersion (aplus): mean=%.1f  median=%.1f  [%.1f, %.1f]\n" aplus_mean median(aplus_draws) aplus_ci[1] aplus_ci[2]

# ─── 9. Save key objects for interactive use ──────────────────────────────
save_path = joinpath(output_dir, "pt_result.jld2")
try
    using JLD2
    @save save_path pt samples tci D K X Y_merged covariate_names
    @info "Saved full PT result: $save_path"
catch
    @info "JLD2 not available. To save PT result for later reuse: Pkg.add(\"JLD2\")"
end

println("\nDone! All outputs in: $(abspath(output_dir))")
