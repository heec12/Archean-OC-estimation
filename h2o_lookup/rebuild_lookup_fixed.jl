"""
rebuild_lookup_fixed.jl
=======================
Rebuilds the bound-H2O lookup tables with a CORRECTED parser, reusing the
Perple_X builds already on disk (only re-queries points — no perplex_configure,
no vertex reruns).

The bug: bound H2O was computed as (bulk − free_fluid), and the free fluid was
only recognised when named "fluid" or "H2O". When Perple_X names the free water
phase "h2oL" (and other cases), it was never subtracted, so bound H2O wrongly
equalled the full system water (~4.762 wt%) — the vertical stripe.

The fix: read the "Solid Only" H2O wt% straight out of the Bulk Composition
block. Perple_X already separates solid-hosted water from fluid/melt there, so
there is nothing to subtract and no phase name to match.

Writes *_fixed_* CSVs so you can compare against the old ones before overwriting.

  julia --threads 4 rebuild_lookup_fixed.jl
"""

using StatGeochem, CSV, DataFrames, Statistics

const PERPLEX_VERSION = "v2"   # set to match the scratch dirs you want to reuse
const DATA_DIR    = "/storage/home/hcoda1/7/hchoi342/scratch/Archean-OC-estimation/Bayesian/bayesian_lower_crust_outputs"
const OUTPUT_DIR  = "/storage/home/hcoda1/7/hchoi342/scratch/Archean-OC-estimation/h2o_lookup"
const SCRATCH_DIR = "/storage/home/hcoda1/7/hchoi342/scratch/Archean-OC-estimation/perplex_ensemble"
const SCENARIOS   = ["homogeneous_crust", "layered_cumulate_lower_crust"]

const P_VEC = collect(range(0.0001 * 10000, 8.0 * 10000, length=40))  # bar
const T_VEC = collect(range(273.0, 1600.0, length=40))                # K
const N_SAMPLES = 0   # 0 = all

# =============================================================================
# FIXED PARSER — read solid-hosted H2O from the Bulk Composition block
# =============================================================================
function parse_bound_h2o(point_str::String)
    isempty(strip(point_str)) && return NaN
    in_bulk = false
    for line in split(point_str, "\n")
        if occursin("Bulk Composition", line)
            in_bulk = true
            continue
        end
        if in_bulk
            occursin("Other Bulk Properties", line) && break
            toks = split(strip(line))
            if !isempty(toks) && toks[1] == "H2O"
                # Complete:  mol g wt% mol/kg   |   Solid Only: mol g wt% mol/kg
                # toks:  1    2  3  4    5            6   7  8    9
                if length(toks) >= 8            # fluid/melt present -> Solid Only wt%
                    v = tryparse(Float64, toks[8])
                    v !== nothing && return max(0.0, v)
                elseif length(toks) >= 4        # no fluid -> solid == complete
                    v = tryparse(Float64, toks[4])
                    v !== nothing && return max(0.0, v)
                end
            end
        end
    end
    return NaN
end

# =============================================================================
# OUTPUT WRITER (verbatim from the pipeline)
# =============================================================================
function write_lookup_table(grid::Matrix{Float64}, path::String)
    lines = String[]
    push!(lines, "," * join([string(round(t, digits=4)) for t in T_VEC], ","))
    for (j, P_bar) in enumerate(P_VEC)
        p_str = string(round(P_bar / 10000.0, digits=6))
        vals = [isnan(grid[j, k]) ? "" : string(round(grid[j, k], digits=6))
                for k in 1:length(T_VEC)]
        push!(lines, p_str * "," * join(vals, ","))
    end
    write(path, join(lines, "\r"))
    println("  Saved: $path")
end

# =============================================================================
# REBUILD
# =============================================================================
for scenario in SCENARIOS
    println("\n" * "="^60, "\nScenario: $scenario\n", "="^60)
    df_ens = CSV.read(joinpath(DATA_DIR, "ensemble_$(scenario).csv"), DataFrame)
    n = (N_SAMPLES == 0) ? nrow(df_ens) : min(N_SAMPLES, nrow(df_ens))

    ens = fill(NaN, n, length(P_VEC), length(T_VEC))

    Threads.@threads for i in 1:n
        scratchdir_i = joinpath(SCRATCH_DIR, "$(PERPLEX_VERSION)_$(scenario)_sample_$(i)")
        isdir(joinpath(scratchdir_i, "out1")) || (@warn "  #$i: no build, skipping"; continue)
        try
            for (j, P) in enumerate(P_VEC), (k, T) in enumerate(T_VEC)
                ens[i, j, k] = parse_bound_h2o(perplex_query_point(scratchdir_i, P, T))
            end
            println("  sample $i / $n done")
        catch e
            @warn "  #$i failed: $e"
        end
    end

    nanpct(x, p) = (v = filter(!isnan, x); isempty(v) ? NaN : quantile(v, p/100))
    p05 = [nanpct(ens[:, j, k], 5)  for j in 1:length(P_VEC), k in 1:length(T_VEC)]
    p50 = [nanpct(ens[:, j, k], 50) for j in 1:length(P_VEC), k in 1:length(T_VEC)]
    p95 = [nanpct(ens[:, j, k], 95) for j in 1:length(P_VEC), k in 1:length(T_VEC)]

    write_lookup_table(p05, joinpath(OUTPUT_DIR, "$(PERPLEX_VERSION)_h2o_p05_fixed_$(scenario).csv"))
    write_lookup_table(p50, joinpath(OUTPUT_DIR, "$(PERPLEX_VERSION)_h2o_p50_fixed_$(scenario).csv"))
    write_lookup_table(p95, joinpath(OUTPUT_DIR, "$(PERPLEX_VERSION)_h2o_p95_fixed_$(scenario).csv"))
end

println("\nDone. Plot the *_fixed_* tables against the old ones — the stripe should be gone.")

