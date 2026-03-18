"""
basic_efdm_fit.jl

Demonstrates the high-level EFDMTempering API:
  1. Load count-composition data from a CSV file
  2. Fit the EFDM model using Pigeons.jl parallel tempering
  3. Summarise the posterior samples with MCMCChains

Usage:
    julia --project=.. basic_efdm_fit.jl

Or from the Julia REPL (with the project activated):
    include("basic_efdm_fit.jl")
"""

using EFDMTempering
using MCMCChains

# ── 1. Path to example data ────────────────────────────────────────────────────
data_path = joinpath(@__DIR__, "..", "data", "example_counts.csv")

# ── 2. Fit the EFDM model ──────────────────────────────────────────────────────
# Columns whose names start with "Y" are treated as response (counts);
# remaining columns become covariates (an intercept is automatically prepended).
println("Fitting EFDM model with Pigeons.jl parallel tempering …")
pt = fit_efdm(
    data_path;
    n_rounds = 10,    # increase for production use (12–16 recommended)
    n_chains = 8,
)

# ── 3. Extract posterior chains ────────────────────────────────────────────────
chains = get_chains(pt)
println("\nPosterior summary (selected parameters):")
println(chains)

# ── 4. Sanity check: round-trip diagnostic ────────────────────────────────────
using Pigeons: stepping_stone
println("\nLog-marginal-likelihood estimate (stepping-stone):")
println(stepping_stone(pt))
