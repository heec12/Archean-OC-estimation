# 5H2O_test.jl
# =======================
# Runs Perple_X for each posterior bulk composition sample and builds
# a P-T-H2O lookup table with uncertainty from compositional spread.
#
# Output format matches reference CSV:
#   - Top row: empty cell, then T values in K (plain numbers, no units)
#   - First column: P in GPa (plain numbers, no units)
#   - CR-only line endings (\r)

using StatGeochem
using CSV, DataFrames, Statistics

# =============================================================================
# SETTINGS
# =============================================================================
const DATA_DIR    = "/Users/hchoi342/Documents/Archean-OC/Bayesian/bayesian_lower_crust_outputs"
const OUTPUT_DIR  = "/Users/hchoi342/Documents/Archean-OC/h2o_lookup"
const SCRATCH_DIR = "/tmp/perplex_ensemble"
const H2O_WT      = 5.0   # wt% H2O added to every bulk composition

const SCENARIOS = [
    "homogeneous_crust",
    "layered_cumulate_lower_crust",
]

# ── QUICK TEST (10x10) — confirm output format is correct ────────────────────
const P_VEC = collect(range(0.0001 * 10000, 8.0 * 10000, length=10))  # bar
const T_VEC = collect(range(200.0, 1600.0, length=10))                 # Kelvin

# ── FULL RUN (80x80) — switch to this once format is confirmed ───────────────
# const P_VEC = collect(range(0.0001 * 10000, 8.0 * 10000, length=80))  # bar
# const T_VEC = collect(range(200.0, 1600.0, length=80))                 # Kelvin

const N_SAMPLES_TEST = 3   # set to nrow(df_ens) for full run

mkpath(OUTPUT_DIR)
mkpath(SCRATCH_DIR)

# =============================================================================
# PARSER — extract bound H2O from a perplex_query_point string
# =============================================================================
function parse_bound_h2o(point_str::String)
    isempty(strip(point_str)) && return NaN

    phase_h2o      = Dict{String, Float64}()
    bulk_h2o       = NaN
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
                wt_pct       = tryparse(Float64, tokens[2])
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
# OUTPUT WRITER — CR-only line endings, P in GPa, T in K, no units
# =============================================================================
function write_lookup_table(grid::Matrix{Float64}, path::String)
    """
    Write (n_P × n_T) matrix as CSV matching the reference format:
      - Empty top-left cell
      - Header: T values in K (plain numbers)
      - First column: P in GPa (plain numbers)
      - Values: H2O wt% (NaN written as empty string)
      - Line endings: CR only (\r)
    """
    lines = String[]

    # Header row: empty cell + T values in K
    t_labels = [string(round(t, digits=4)) for t in T_VEC]
    push!(lines, "," * join(t_labels, ","))

    # Data rows: P in GPa + values
    for (j, P_bar) in enumerate(P_VEC)
        p_gpa = P_bar / 10000.0
        p_str = string(round(p_gpa, digits=6))
        vals  = [isnan(grid[j, k]) ? "" : string(round(grid[j, k], digits=6))
                 for k in 1:length(T_VEC)]
        push!(lines, p_str * "," * join(vals, ","))
    end

    # Join with CR only — no trailing newline
    content = join(lines, "\r")
    write(path, content)
    println("  Saved: $path  ($(length(P_VEC)) P × $(length(T_VEC)) T, CR-only)")
end

# =============================================================================
# MAIN ENSEMBLE LOOP
# =============================================================================
for scenario in SCENARIOS
    println("\n" * "="^60)
    println("Scenario: $scenario")
    println("="^60)

    csv_path = joinpath(DATA_DIR, "ensemble_$(scenario).csv")
    df_ens   = CSV.read(csv_path, DataFrame)
    n        = min(N_SAMPLES_TEST, nrow(df_ens))
    println("Loaded $n ensemble compositions from: $csv_path")

    # Storage: (n_samples × n_P × n_T)
    bound_h2o_ensemble = fill(NaN, n, length(P_VEC), length(T_VEC))
    n_failed = 0

    for i in 1:n
        println("\n  Sample $i / $n")

        comp = vcat(Vector{Float64}(df_ens[i, :]), H2O_WT)

        scratchdir_i = joinpath(SCRATCH_DIR, "$(scenario)_sample_$(i)")
        mkpath(scratchdir_i)

        try
            perplex_configure_pseudosection(
                scratchdir_i,
                comp,
                ["SiO2","TiO2","Al2O3","FeO","MgO","CaO","Na2O","H2O"],
                (1, 80000),
                (200.0, 1600.0),
                dataset         = "hp62ver.dat",
                solution_phases = "O(HGP)\nCpx(HGP)\nOpx(HGP)\nGt(HGP)\nChl(W)\nEp(HP)\nPheng(HP)\nSp(HGP)\nFsp(HGP)\n",
                excludes        = "ts\nparg\ngl\nged\nfanth\n",
                fluid_eos       = 5,
            )

            n_ok = 0
            for (j, P) in enumerate(P_VEC)
                for (k, T) in enumerate(T_VEC)
                    pt   = perplex_query_point(scratchdir_i, P, T)
                    bh2o = parse_bound_h2o(pt)
                    bound_h2o_ensemble[i, j, k] = bh2o
                    n_ok += 1
                end
            end
            println("    ✓ $n_ok P-T points queried")

        catch e
            @warn "  Sample $i failed: $e"
            n_failed += 1
        end
    end

    println("\nFailed samples: $n_failed / $n")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    # Samples with all NaN (entire run failed)
    n_all_nan = sum(all(isnan.(bound_h2o_ensemble[i,:,:])) for i in 1:n)
    println("Samples with all NaN (full run failure): $n_all_nan / $n")

    # Individual NaN cells
    n_cell_nan = count(isnan, bound_h2o_ensemble)
    println("Individual NaN cells: $n_cell_nan / $(n * length(P_VEC) * length(T_VEC))")

    # P-T regions with most NaN
    println("P-T cells with >50% NaN samples:")
    found_any = false
    for j in 1:length(P_VEC), k in 1:length(T_VEC)
        n_nan_here = count(isnan, bound_h2o_ensemble[:, j, k])
        if n_nan_here > n * 0.5
            println("  P=$(round(P_VEC[j]/10000, digits=4)) GPa, T=$(round(T_VEC[k], digits=1)) K")
            found_any = true
        end
    end
    found_any || println("  none")

    # ── Summarise across ensemble ─────────────────────────────────────────────
    println("\nComputing ensemble statistics...")

    nanmean(x) = (v = filter(!isnan, x); isempty(v) ? NaN : mean(v))
    nanstd(x)  = (v = filter(!isnan, x); length(v) < 2 ? NaN : std(v))

    h2o_mean = [nanmean(bound_h2o_ensemble[:, j, k])
                for j in 1:length(P_VEC), k in 1:length(T_VEC)]
    h2o_std  = [nanstd(bound_h2o_ensemble[:, j, k])
                for j in 1:length(P_VEC), k in 1:length(T_VEC)]

    # ── Save lookup tables in reference format ────────────────────────────────
    mean_path = joinpath(OUTPUT_DIR, "h2o_bound_mean_$(scenario).csv")
    std_path  = joinpath(OUTPUT_DIR, "h2o_bound_std_$(scenario).csv")

    write_lookup_table(h2o_mean, mean_path)
    write_lookup_table(h2o_std,  std_path)
end

println("\nDone. All lookup tables saved to $OUTPUT_DIR")
