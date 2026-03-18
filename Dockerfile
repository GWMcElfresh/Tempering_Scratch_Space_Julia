# ─── Stage 1: Dependency image ────────────────────────────────────────────────
# This layer is cached monthly by the dockerDependencies CI workflow.
# Re-run only when Julia version or key dependencies change.
FROM julia:1.10 AS deps

WORKDIR /app

# Install system-level Stan prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy only the project manifests first to maximise layer-cache reuse
COPY Project.toml /app/Project.toml

# Pre-install and pre-compile all Julia dependencies
RUN julia --project=/app -e " \
    import Pkg; \
    Pkg.instantiate(); \
    Pkg.precompile(); \
    "

# ─── Stage 2: Runtime image ────────────────────────────────────────────────────
FROM deps AS runtime

WORKDIR /app

# Copy the full package source
COPY . /app

# Final precompile with the full source tree in place
RUN julia --project=/app -e " \
    import Pkg; \
    Pkg.precompile(); \
    "

# Default command: run the basic example
CMD ["julia", "--project=/app", "examples/basic_efdm_fit.jl"]
