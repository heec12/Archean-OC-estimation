# 5H2O_test_v1_parallel.jl
# =======================
# Runs Perple_X for each posterior bulk composition sample and builds
# a P-T-H2O lookup table with uncertainty from compositional spread.
#
# Needs to be run with multiple threads:
#   julia --threads 4 5H2O_test_v1_parallel.jl
#
# Versioning:
#   PERPLEX_VERSION is parsed from this script's own filename — the "_vN"
#   token (e.g. 5H2O_test_v2_parallel.jl -> "v2"). It tracks the Perple_X
#   SOLUTION-MODEL version and tags ONLY the things that change with it:
#   the lookup tables and the scratch directories. So this exact file can be
#   copied to 5H2O_test_v1_parallel.jl and it will pick up "v1" with no edits.
#
#   The ensemble is a BAYESIAN artifact — it does not change when the
#   Perple_X solution models change, so it is read from a fixed, unversioned
#   name (ensemble_<scenario>.csv) and shared across all Perple_X versions.
#   If you ever change the Bayesian model itself, version the ensemble on its
#   own separate axis (not with PERPLEX_VERSION).
#
# Output format matches reference CSV:
# - Top row: empty cell, then T values in K (plain numbers, no units)
# - First column: P in GPa (plain numbers, no units)
# - CR-only line endings (\r)

using StatGeochem
using CSV, DataFrames, Statistics

# =============================================================================
# SETTINGS
# =============================================================================
# Perple_X solution-model version, taken from this file's name: the "_vN"
# token. 5H2O_test_v1_parallel.jl -> "v1" and 5H2O_test_v2_parallel.jl -> "v2"
# automatically, with the same code in both files. Tags lookup tables and
# scratch dirs only — NOT the (Bayesian) ensemble input.
const PERPLEX_VERSION = let
    fname = basename(@__FILE__)
    m = match(r"_v(\d+)", fname)
    m === nothing && error("Could not parse a _vN version token from filename: $fname")
    "v" * m.captures[1]
end

const DATA_DIR   = "/storage/home/hcoda1/7/hchoi342/scratch/Archean-OC-estimation/Bayesian/bayesian_lower_crust_outputs"
const OUTPUT_DIR = "/storage/home/hcoda1/7/hchoi342/scratch/Archean-OC-estimation/h2o_lookup"
const SCRATCH_DIR = "/storage/home/hcoda1/7/hchoi342/scratch/Archean-OC-estimation/perplex_ensemble"

const H2O_WT = 5.0   # wt% H2O added to every bulk composition

const SCENARIOS = [
    "homogeneous_crust",
    "layered_cumulate_lower_crust",
]

# -- QUICK TEST (10x10, 3 samples) --------------------------------------------
# const P_VEC = collect(range(0.0001 * 10000, 8.0 * 10000, length=10))
# const T_VEC = collect(range(200.0, 1600.0, length=10))
# const N_SAMPLES = 3

# -- FULL RUN (40x40, all samples) --------------------------------------------
const P_VEC = collect(range(0.0001 * 10000, 8.0 * 10000, length=40))   # bar
const T_VEC = collect(range(273.0, 1600.0, length=40))                  # Kelvin
const N_SAMPLES = 0   # 0 = use all available samples

mkpath(OUTPUT_DIR)
mkpath(SCRATCH_DIR)

println("Perple_X version: $PERPLEX_VERSION")

# =============================================================================
# PARSER
# =============================================================================
function parse_bound_h2o(point_str::String)
    isempty(strip(point_str)) && return NaN
    phase_h2o = Dict{String, Float64}()
    bulk_h2o = NaN
    in_phase_block = false
    for line in split(point_str, "\n")
        if occursin("Phase Compositions (weight percentages)", line)
            in_phase_block = true
            continue
        end
        if in_phase_block && occursin("Phase speciation", line)
            in_phase_block = false
        end
        if in_phase_block
            tokens = split(strip(line))
            if length(tokens) == 13
                wt_pct = tryparse(Float64, tokens[2])
                h2o_in_phase = tryparse(Float64, tokens[end])
                if wt_pct !== nothing && h2o_in_phase !== nothing
                    phase_h2o[tokens[1]] = (wt_pct / 100.0) * h2o_in_phase
                end
            end
        end
        m = match(r"^\s+H2O\s+[\d.]+\s+[\d.]+\s+([\d.]+)", line)
        if m !== nothing
            bulk_h2o = parse(Float64, m[1])
        end
    end
    free_fluid_contrib = get(phase_h2o, "fluid", get(phase_h2o, "H2O", 0.0))
    bound_h2o = isnan(bulk_h2o) ? NaN : bulk_h2o - free_fluid_contrib
    return max(0.0, bound_h2o)
end

# =============================================================================
# OUTPUT WRITER
# =============================================================================
function write_lookup_table(grid::Matrix{Float64}, path::String)
    lines = String[]
    # Header: empty cell + T values in K
    t_labels = [string(round(t, digits=4)) for t in T_VEC]
    push!(lines, "," * join(t_labels, ","))
    # Data rows: P in GPa + values
    for (j, P_bar) in enumerate(P_VEC)
        p_gpa = P_bar / 10000.0
        p_str = string(round(p_gpa, digits=6))
        vals = [isnan(grid[j, k]) ? "" : string(round(grid[j, k], digits=6))
                for k in 1:length(T_VEC)]
        push!(lines, p_str * "," * join(vals, ","))
    end
    write(path, join(lines, "\r"))
    println("  Saved: $path ($(length(P_VEC)) P x $(length(T_VEC)) T, CR-only)")
end

# =============================================================================
# MAIN ENSEMBLE LOOP
# =============================================================================
for scenario in SCENARIOS
    println("\n" * "="^60)
    println("Scenario: $scenario")
    println("="^60)

    # Bayesian artifact: fixed name, shared across all Perple_X versions.
    csv_path = joinpath(DATA_DIR, "ensemble_$(scenario).csv")
    df_ens = CSV.read(csv_path, DataFrame)

    # N_SAMPLES = 0 means all rows; positive value caps at that number
    n = (N_SAMPLES == 0) ? nrow(df_ens) : min(N_SAMPLES, nrow(df_ens))
    println("Running $n / $(nrow(df_ens)) ensemble compositions")

    # wrap sampling part with the multithreads, each sample is independent
    bound_h2o_ensemble = fill(NaN, n, length(P_VEC), length(T_VEC))
    n_failed = Threads.Atomic{Int}(0)

    Threads.@threads for i in 1:n
        println("  Sample $i / $n (thread $(Threads.threadid()))")
        comp = vcat(Vector{Float64}(df_ens[i, :]), H2O_WT)

        # Each thread needs its own scratch directory to avoid file conflicts.
        # Tagged with PERPLEX_VERSION so v1 and v2 runs never share a working dir.
        scratchdir_i = joinpath(SCRATCH_DIR, "$(PERPLEX_VERSION)_$(scenario)_sample_$(i)")
        mkpath(scratchdir_i)

        try
            perplex_configure_pseudosection(
                scratchdir_i,
                comp,
                ["SiO2","TiO2","Al2O3","FeO","MgO","CaO","Na2O","H2O"],
                (1, 80000),
                (273.0, 1600.0),
                dataset = "hp62ver.dat",
                solution_phases = "O(HGP)\nCpx(HGP)\nOpx(HGP)\nGt(HGP)\nChl(W)\nEp(HP)\nPheng(HP)\nSp(HGP)\nFsp(HGP)\n",
                # solution_phases = "O(HGP)\nCpx(HGP)\nOpx(HGP)\nGt(HGP)\nFsp(HGP)\nChl(W)\nEp(HP)\nPheng(HP)\n",
                excludes = "ts\nparg\ngl\nged\nfanth\n",
                fluid_eos = 5,
            )

            # n_ok = 0
            for (j, P) in enumerate(P_VEC)
                for (k, T) in enumerate(T_VEC)
                    pt = perplex_query_point(scratchdir_i, P, T)
                    bh2o = parse_bound_h2o(pt)
                    bound_h2o_ensemble[i, j, k] = bh2o
                    # n_ok += 1
                end
            end
            println("  v Sample $i done")
        catch e
            @warn "  Sample $i failed: $e"
            Threads.atomic_add!(n_failed, 1)
        end
    end

    println("Failed samples: $(n_failed[]) / $n")

    n_all_nan = sum(all(isnan.(bound_h2o_ensemble[i,:,:])) for i in 1:n)
    n_cell_nan = count(isnan, bound_h2o_ensemble)
    println("Samples with all NaN: $n_all_nan / $n")
    println("Individual NaN cells: $n_cell_nan / $(n * length(P_VEC) * length(T_VEC))")

    println("P-T cells with >50% NaN:")
    found_any = false
    for j in 1:length(P_VEC), k in 1:length(T_VEC)
        if count(isnan, bound_h2o_ensemble[:, j, k]) > n * 0.5
            println("  P=$(round(P_VEC[j]/10000, digits=4)) GPa, T=$(round(T_VEC[k], digits=1)) K")
            found_any = true
        end
    end
    found_any || println("  none")

    println("\nComputing ensemble statistics...")
    nanmean(x) = (v = filter(!isnan, x); isempty(v) ? NaN : mean(v))
    nanstd(x)  = (v = filter(!isnan, x); length(v) < 2 ? NaN : std(v))

    h2o_mean = [nanmean(bound_h2o_ensemble[:, j, k])
                for j in 1:length(P_VEC), k in 1:length(T_VEC)]
    h2o_std  = [nanstd(bound_h2o_ensemble[:, j, k])
                for j in 1:length(P_VEC), k in 1:length(T_VEC)]

    write_lookup_table(h2o_mean, joinpath(OUTPUT_DIR, "$(PERPLEX_VERSION)_h2o_bound_mean_$(scenario).csv"))
    write_lookup_table(h2o_std,  joinpath(OUTPUT_DIR, "$(PERPLEX_VERSION)_h2o_bound_std_$(scenario).csv"))

    nanpercentile(x, p) = (v = filter(!isnan, x); isempty(v) ? NaN : quantile(v, p/100))
    h2o_p05 = [nanpercentile(bound_h2o_ensemble[:, j, k], 5)
               for j in 1:length(P_VEC), k in 1:length(T_VEC)]
    h2o_p50 = [nanpercentile(bound_h2o_ensemble[:, j, k], 50)
               for j in 1:length(P_VEC), k in 1:length(T_VEC)]
    h2o_p95 = [nanpercentile(bound_h2o_ensemble[:, j, k], 95)
               for j in 1:length(P_VEC), k in 1:length(T_VEC)]

    write_lookup_table(h2o_p05, joinpath(OUTPUT_DIR, "$(PERPLEX_VERSION)_h2o_p05_$(scenario).csv"))
    write_lookup_table(h2o_p50, joinpath(OUTPUT_DIR, "$(PERPLEX_VERSION)_h2o_p50_$(scenario).csv"))
    write_lookup_table(h2o_p95, joinpath(OUTPUT_DIR, "$(PERPLEX_VERSION)_h2o_p95_$(scenario).csv"))
end

println("\nDone. All lookup tables saved to $OUTPUT_DIR  (Perple_X version: $PERPLEX_VERSION)")
