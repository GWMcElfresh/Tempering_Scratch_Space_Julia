# EFDMAnalysis — Extended Flexible Dirichlet Multinomial Regression

Bayesian analysis of multivariate count data using the EFDM model with
Parallel Tempering (Pigeons.jl). Designed for cell-type composition analysis
from single-cell genomics data, but applicable to any multivariate count
regression problem.

## Quick Start

```bash
# 1. Copy the config template
cp config/model_config.toml my_config.toml
# Edit: set data path, response columns, covariate columns

# 2. Run
julia scripts/run_efdm.jl my_config.toml path/to/counts.csv
```

The runner will:
1. Load data and build design matrix
2. Run parallel tempering MCMC
3. Print convergence diagnostics (barrier, restarts, Rhat)
4. Generate plots and posterior summaries in `efdm_output/`

## Installation

This is a standalone Julia package (no registry publication needed). To use it:

```julia
# From the efdm-analysis directory
julia> ] activate .
julia> using Pigeons, Distributions, CSV, DataFrames, TOML, MCMCChains, SpecialFunctions
julia> include("src/EFDMAnalysis.jl")
julia> using .EFDMAnalysis
```

Or for development:
```julia
julia> using Pkg; Pkg.develop(path=".")
```

## Architecture

```
efdm-analysis/
├── Project.toml                # Dependencies
├── config/
│   └── model_config.toml       # All model parameters documented inline
├── src/
│   ├── EFDMAnalysis.jl         # Module entry point
│   ├── core.jl                 # Pure EFDM math (no side effects)
│   ├── targets.jl              # Pigeons.jl target + reference distributions
│   ├── diagnostics.jl          # Post-MCMC convergence analysis
│   └── plotting.jl             # Diagnostic and effect visualizations
└── scripts/
    └── run_efdm.jl             # Config + CSV driven runner
```

## Data Format

### Expected CSV schema (from GoodWorkflows `ingest_tabulate`)

| donor_id | CT0 | CT1 | CT2 | CT3 | treatment | age |
|----------|-----|-----|-----|-----|-----------|-----|
| donor_a  | 120 | 340 | 210 | 45  | drug      | 45  |
| donor_b  | 95  | 410 | 180 | 30  | control   | 52  |
| ...      | ... | ... | ... | ... | ...       | ... |

- **Rows** = subjects/donors/observations
- **Numeric columns** = integer cell-type counts (response variables)
- **Metadata columns** = covariates, subject IDs, experimental conditions

The `process_pbmc.R` pipeline produces this format from `TENxPBMCData`.

### Specifying the model in config

```toml
[data]
path = "data/counts.csv"
response_columns = ["CT0", "CT1", "CT2", "CT3", "CT4", "CT5"]
covariate_columns = ["treatment", "age"]
subject_column = "donor_id"
merge_rare_types = true
merge_threshold = 0.015
```

The runner automatically adds an intercept column. Categorical covariates should
be pre-encoded as numeric (0/1 for binary, one-hot for multi-level). The config
maps response and covariate columns by name, so the CSV can contain additional
unused columns without issues.

## Parameter Reference

### β — Regression Coefficients

| Property | Description |
|----------|-------------|
| **Shape** | (D-1) × K matrix (unconstrained). Last row is fixed to zero (reference category). |
| **Role** | `mu_d = softmax(X @ β_d)` gives expected cell-type proportions from covariates. |
| **Prior** | Normal(0, sd²). Default sd = 1.0. |
| **Interpretation** | exp(β[r, c]) = multiplicative change in odds of category r vs. reference D per unit increase in covariate c. |
| **Convergence note** | Keep sd ≤ 1.0. sd > 1.5 creates massive prior–posterior gaps (Λ > 10). |
| **Example** | β[2, 3] = 0.5 → a 1-unit increase in covariate 3 multiplies the odds of CT2 vs. reference by exp(0.5) ≈ 1.65. |

### aplus (a.k.a. a) — Overdispersion

| Property | Description |
|----------|-------------|
| **Shape** | Scalar, strictly positive. Sampled as log(aplus) ~ Normal. |
| **Role** | Controls how much the Dirichlet-Multinomial deviates from pure Multinomial. |
| **Interpretation** | a → ∞ = Multinomial (no overdispersion). a → 0 = extreme overdispersion. |
| **Prior** | log(a) ~ Normal(log_mean, log_sd²). Default: log_mean=3.5 (≈ a=33), log_sd=1.0. |
| **Typical range** | PBMC data: a ≈ 10-30. If a > 100, data barely needs EFDM. If a < 1, very strong overdispersion. |
| **Convergence note** | a and p form a posterior funnel. log_sd=1.0 is tight enough to stabilize convergence. |

### p — Baseline Mixture Weights

| Property | Description |
|----------|-------------|
| **Shape** | D-element simplex (sums to 1). Additive log-ratio transformed. |
| **Role** | Expected proportion = (1-w) · mu + w · p. The "pure" baseline that mixes with regression-driven mu. |
| **Prior** | Dirichlet-like via Normal(0, sd²) on ALR scale. Default sd = 2.0. |
| **Convergence note** | The a-p funnel means p and aplus covary strongly. SliceSampler handles this much better than AutoMALA. |

### w — Component Weights

| Property | Description |
|----------|-------------|
| **Shape** | D elements, each in (0, 1). Sigmoid-transformed. |
| **Role** | w_j → 0 means regression determines everything. w_j → 1 means baseline p_j dominates. |
| **Prior** | Normal(0, sd²) on logit scale. Default sd = 2.0. |
| **Convergence note** | If data are well-explained by covariates, w is pulled toward 0. Monitor during inference. |

## Convergence Recommendations

These are distilled from extensive testing documented in `lessons_learned.txt`.

### 1. Always use GaussianReference variational PT

```toml
[sampler]
use_variational = true
first_tuning_round = 3
n_chains_variational = 10
```

The prior–posterior gap for EFDM can be enormous (log Z₁/Z₀ ≈ −3000).
Variational PT bridges this gap by learning a Gaussian approximation to the
posterior as an intermediate rung on the temperature ladder. Without it,
even 80 chains may produce zero restarts.

### 2. Default to SliceSampler, not AutoMALA

The a-p posterior has a funnel geometry: high a + wrong p ≈ low a + right p.
Gradient-based samplers (MALA, HMC) struggle with this. SliceSampler is
coordinate-wise adaptive and non-rejecting.

```toml
[sampler]
explorer = "SliceSampler"
```

### 3. Watch the "three numbers" in order

```
1. Tempered restarts (rst)  — must be > 5 for any mixing
2. Variational barrier (Λ_var) — should be < 3
3. Max Rhat — check only after rst > 5, should be < 1.05
```

### 4. Fast iteration protocol

| Phase | n_chains | n_rounds | Purpose |
|-------|----------|----------|---------|
| Quick check | 20 | 6-7 | Does the pipeline work? Λ < 3? |
| Diagnosis | 30 | 8 | Are restarts flowing? Rhat improving? |
| Production | 30 | 10-12 | Final inference |

## Output Interpretation

### Convergence assessment

```
CONVERGENCE DIAGNOSTICS
============================================================
  Λ (variational leg):        1.236
  Tempered restarts:          42
  Round trips:                5
  Max Rhat:                   1.023
  Min ESS:                    184.5

  Result: CONVERGED  (Λ≤3, rst≥10, Rhat<1.05)
```

The runner prints this table after every run. Interpretation:

| Scenario | Diagnosis | Next step |
|----------|-----------|-----------|
| rst = 0, Λ > 5 | Ladder is broken | Enable variational PT or tighten priors |
| rst > 5, Λ < 3 | Chains are mixing | Increase n_rounds to improve Rhat |
| Rhat < 1.05 | All parameters converged | Trust the posterior |
| Rhat < 1.05 but rst = 0 | False positive | Chains stuck in one mode — mismatch |

### Output files

All files are written to the configured output directory (default: `efdm_output/`).

| File | Description |
|------|-------------|
| `convergence_report.txt` | Text summary of convergence diagnostics |
| `posterior_summary.csv` | Mean, SD, and quantiles for all parameters |
| `aplus_posterior.html` | Histogram of aplus with CI annotation |
| `aplus_posterior_data.csv` | Raw aplus draws |
| `effect_*.html` | Conditional effect plots (one per continuous covariate) |
| `conditional_effects_data.csv` | Grid predictions for all covariates |
| `rhat_summary.html` | Ordered Rhat bar chart with 1.05 threshold |
| `rhat_data.csv` | Rhat values per parameter |
| `trace_*.html` | Trace plots for key parameters |
| `trace_data.csv` | All trace data in long format |

### Conditional effects plots

For each continuous covariate, the expected cell-type proportions at the
posterior mean β are plotted across the covariate's range, holding all other
covariates at their means. This shows how the composition shifts with each
predictor.

## Programmatic API

For interactive use or custom workflows:

```julia
using Pigeons, Distributions, Random
include("src/EFDMAnalysis.jl")
using .EFDMAnalysis

# Load data
df = CSV.read("data/counts.csv", DataFrame)
Y = Matrix{Int}(df[!, [:CT0, :CT1, :CT2]])
X = hcat(ones(size(Y, 1)), df.treatment)

# Build model
target = EFDMLogPotential(Float64.(Y), X, 3, 2, 1.0)
reference = EFDMReference(3, 2, 1.0)

# Run PT
pt = pigeons(;
    target=target, reference=reference,
    n_chains=30, n_rounds=8,
    explorer=SliceSampler(),
    variational=GaussianReference(first_tuning_round=3),
    n_chains_variational=10,
    extended_traces=true,
    record=[traces; round_trip; index_process; record_default()],
)

# Diagnostics
summary = convergence_summary(pt)
println(convergence_report_str(summary))

# Extract samples
samples = sample_array(pt)
tci = extract_target_chain_indices(pt)
aplus_draws = extract_parameter_draws(samples, (3-1)*2+1, tci)
```

## Acknowledgements

This package was developed from the `Julia_PT` analysis repository (GWMcElfresh/Julia_PT),
which documents the full convergence investigation of EFDM + Pigeons.jl in
`lessons_learned.txt`. The core EFDM math is ported from `pigeons_efdm_examples.jl`
(Olivier Binette, 2023).

## License

MIT
