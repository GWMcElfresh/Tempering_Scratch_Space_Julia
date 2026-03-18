"""
custom_stan_fit.jl

Demonstrates fitting an arbitrary Stan model file with a CSV data file using
the low-level `fit_stan_with_pigeons` function.

Usage:
    julia --project=.. custom_stan_fit.jl

Or from the Julia REPL:
    include("custom_stan_fit.jl")
"""

using EFDMTempering

# ── Paths ──────────────────────────────────────────────────────────────────────
stan_file = stan_model_path(:dm)          # use the bundled DM model
data_file = joinpath(@__DIR__, "..", "data", "example_counts.csv")

# ── Fit ────────────────────────────────────────────────────────────────────────
# The DM model uses the same CSV format; prepare_efdm_data handles both
# (w_hyper is ignored when passed to DM – only sd_prior matters there,
#  but prepare_efdm_data builds the full dict; unused keys are silently
#  ignored by StanLogPotential).
println("Fitting Dirichlet-Multinomial model …")
pt = fit_stan_with_pigeons(
    stan_file,
    data_file;
    n_rounds = 8,
    n_chains = 4,
)

println("\nRound-trip rates:")
using Pigeons: report_round_trips
report_round_trips(pt)
