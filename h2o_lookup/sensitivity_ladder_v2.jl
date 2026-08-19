#=
sensitivity_ladder_v2.jl
========================
Single-composition pseudosection sensitivity runs: posterior-MEAN bulk
composition only, stepped through a ladder of solution-model sets, to
attribute changes in bound H2O to individual solution models before
committing the full 200-sample ensemble.

Ladder (each rung adds one model to the previous):
    L0_v1_effective : O(HGP) Cpx(HGP) Opx(HGP) Gt(HGP) Chl(W) Fsp(HGP)
                      (= what v1 ACTUALLY ran: Ep(HP), Pheng(HP), Sp(HGP)
                       were silently rejected by build on hp62ver/NCFMAST)
    L1_plus_amph    : + cAmph(G)   (Green et al. 2016, ds62-calibrated)
    L2_plus_talc    : + T
    L3_v2_full      : + Sp(WPC)

Verified against Perple_X 7.1.9 + hp62ver.dat + bundled solution_model.dat
(v719): all models load, vertex runs without interactive prompts.
Do NOT substitute Amph(DPW) (vertex ver017 interactive prompt = stdin EOF
crash under StatGeochem), Sp(HGP) (needs ds633 make definitions),
Ep(HP)/Pheng(HP) (need O2/K2O; rejected in NCFMAST).

Usage:
    julia -t 8 sensitivity_ladder_v2.jl

Inputs:  $(BAYES_DIR)/posterior_compositions_<scenario>.csv  (x_bulk_mean)
Outputs: $(OUT_DIR)/boundH2O_<scenario>_<rung>.csv           (P-T grids)
         $(OUT_DIR)/delta_boundH2O_<scenario>_<rung-pair>.csv (differences)
         $(OUT_DIR)/summary_<scenario>.csv
=#

using StatGeochem
using DelimitedFiles
using Printf

# =============================================================================
# 1. CONFIG
# =============================================================================
# Version tag derived from this script's filename (same convention as the
# ensemble pipeline: Bayesian axis via BAYES_DIR, Perple_X axis via filename)
const LADDER_VERSION = let
    m = match(r"sensitivity_ladder_(v\d+\w*)", basename(@__FILE__))
    m === nothing ? "vX" : m.captures[1]
end

#const BAYES_DIR   = "bayesian_lower_crust_outputs"
const BAYES_DIR = "/storage/home/hcoda1/7/hchoi342/scratch/Archean-OC-estimation/Bayesian/bayesian_lower_crust_outputs"
#const SCRATCH_TOP = "perplex_sensitivity"
const SCRATCH_TOP = joinpath(dirname(@__DIR__), "perplex_sensitivity")
const OUT_DIR     = joinpath(SCRATCH_TOP, "$(LADDER_VERSION)_results")

const SCENARIOS = ["homogeneous_crust", "layered_cumulate_lower_crust"]

# 7-oxide NCFMAST system + H2O appended as 8th component.
# CRITICAL: hp62ver.dat component names are CASE-SENSITIVE and mixed-case.
# "SIO2" etc. are silently rejected by build, leaving an H2O-only system.
const OXIDES   = ["SiO2", "TiO2", "Al2O3", "FeO", "MgO", "CaO", "Na2O"]
const ELEMENTS = vcat(OXIDES, "H2O")

# H2O wt% appended to the bulk composition.
# >>> Set this to the same value used in 5H2O_test_v2_parallel.jl so the
# >>> ladder is directly comparable to the ensemble runs.
const H2O_WT = 5.0

# P-T window and grid resolution.
# >>> Match these to the ensemble pipeline grid for apples-to-apples maps.
const T_RANGE = (473.15, 1473.15)   # K
const P_RANGE = (500.0, 30000.0)    # bar  (0.05 - 3.0 GPa)
const NT = 40                       # T nodes (werami query grid)
const NP = 40                       # P nodes
const XNODES = 40                   # vertex exploratory grid (x = T)
const YNODES = 40

const DATASET  = "hp62ver.dat"
const EXCLUDES = "ged\nfanth\ngl\n"   # StatGeochem default; keep consistent

# Ordered ladder: (rung_name, solution_phases string)
const MODEL_LADDER = [
    ("L0_v1_effective",
     "O(HGP)\nCpx(HGP)\nOpx(HGP)\nGt(HGP)\nChl(W)\nFsp(HGP)\n"),
    ("L1_plus_amph",
     "O(HGP)\nCpx(HGP)\nOpx(HGP)\nGt(HGP)\nChl(W)\nFsp(HGP)\ncAmph(G)\n"),
    ("L2_plus_talc",
     "O(HGP)\nCpx(HGP)\nOpx(HGP)\nGt(HGP)\nChl(W)\nFsp(HGP)\ncAmph(G)\nT\n"),
    ("L3_v2_full",
     "O(HGP)\nCpx(HGP)\nOpx(HGP)\nGt(HGP)\nChl(W)\nFsp(HGP)\ncAmph(G)\nT\nSp(WPC)\n"),
]

# Phase names that are free fluid, NOT bound water (exact match).
const FLUID_NAMES = Set(["H2O", "F", "fluid", "Fluid", "WADDAH",
                         "Aqfl(HGP)", "COH-Fluid", "COH-Fluid+",
                         "HOS-Fluid", "HO-Fluid"])


# =============================================================================
# 2. BOUND-H2O PARSER  (drop-in slot)
# =============================================================================
#=
>>> If you want strict consistency with the ensemble pipeline, REPLACE this
>>> function with the parser from 5H2O_test_v2_parallel.jl. <<<

Default implementation below: parses the "Phase Compositions" table from
perplex_query_point() text. With StatGeochem's configure defaults
(composition_phase = wt), the oxide columns are wt% of each phase, so

    bound H2O (wt% of rock) = sum over solid phases of
                              phase_wt% * phase_H2O_wt% / 100

Free-fluid phases (FLUID_NAMES) are skipped, so fluid H2O is never
double-counted. Returns NaN if the point text is empty/unparseable.
=#
#function parse_bound_h2o(pointtext::AbstractString)
#    isempty(pointtext) && return NaN

#    lines = split(pointtext, '\n')
#    ihead = findfirst(l -> occursin("Phase Compositions", l), lines)
#    ihead === nothing && return NaN

    # Sanity check: this parser assumes wt-basis phase compositions
#    if occursin("molar", lowercase(lines[ihead]))
#        @warn "Phase compositions are molar, not wt; parser assumes wt basis" maxlog=1
#    end

    # Column header line (first non-empty line after the section header)
#    icol = ihead + 1
#    while icol <= length(lines) && isempty(strip(lines[icol]))
#        icol += 1
#    end
#    icol > length(lines) && return NaN
#    header = split(strip(lines[icol]))
#    ih2o = findlast(==("H2O"), header)   # H2O is the last component column
#    ih2o === nothing && return NaN

#    bound = 0.0
#    nsolid = 0
#    for l in lines[icol+1:end]
#        s = strip(l)
#        isempty(s) && break                       # blank line ends the table
#        occursin("Phase speciation", s) && break
#        tok = split(s)
        # Phase rows: name + >= (wt% vol% mol% mol + 8 components) numbers
#        length(tok) < ih2o + 1 && continue
#        name = tok[1]
#        name in FLUID_NAMES && continue           # skip free fluid
#        wtpct  = tryparse(Float64, tok[2])
        # name column shifts the numeric columns by 1 relative to header
#        h2opct = tryparse(Float64, tok[ih2o+1])
#        (wtpct === nothing || h2opct === nothing) && continue
#        bound += max(0.0, wtpct) * max(0.0, h2opct) / 100.0
#        nsolid += 1
#    end
#    return nsolid > 0 ? max(0.0, bound) : NaN
#end
function parse_bound_h2o(pointtext::AbstractString)
    isempty(strip(pointtext)) && return NaN
    lines = split(pointtext, '\n')
    ib = findfirst(l -> startswith(strip(l), "Bulk Composition"), lines)
    ib === nothing && return NaN

    iend = min(ib + 15, length(lines))
    solid_only = any(occursin("Solid Only", l) for l in lines[ib:min(ib+3, length(lines))])

    for l in lines[ib:iend]
        tok = split(strip(l))
        isempty(tok) && continue
        tok[1] == "H2O" || continue
        if solid_only && length(tok) >= 9
            v = tryparse(Float64, tok[8])          # Solid-Only wt% column
            return v === nothing ? NaN : max(0.0, v)
        elseif length(tok) >= 4
            v = tryparse(Float64, tok[4])          # single-table wt% column
            return v === nothing ? NaN : max(0.0, v)
        end
    end
    return NaN
end

# =============================================================================
# 3. PIPELINE HELPERS
# =============================================================================

"Read posterior-mean bulk composition (7 oxides, wt%) for a scenario."
function load_posterior_mean(scenario::String)
    path = joinpath(BAYES_DIR, "posterior_compositions_$(scenario).csv")
    isfile(path) || error("Not found: $path  -- run mass_balance_7oxides_v2.py first")
    raw, head = readdlm(path, ',', header=true)
    jcol = findfirst(==("x_bulk_mean"), vec(head))
    jcol === nothing && error("x_bulk_mean column missing in $path")
    rownames = String.(raw[:, 1])
    comp = zeros(length(OXIDES))
    for (i, ox) in enumerate(OXIDES)
        # Python script indexes rows as e.g. "SiO2_pct"
        irow = findfirst(r -> startswith(r, ox), rownames)
        irow === nothing && error("Oxide $ox not found in $path")
        comp[i] = Float64(raw[irow, jcol])
    end
    return comp
end

"""
Fail-fast checks after perplex_configure_pseudosection.
Aborts loudly instead of letting 1600 werami queries fail silently.
(Lesson from job 9387978: 191 MB of parse warnings, all-NaN output.)
"""
function check_perplex_run(scratchdir::String; index::Int=1)
    prefix = joinpath(scratchdir, "out$(index)")

    buildlog = joinpath(prefix, "build.log")
    if isfile(buildlog)
        blog = read(buildlog, String)
        for m in eachmatch(r"(\S+)\s+is invalid\.", blog)
            error("build rejected solution model or component '$(m.captures[1])' " *
                  "in $scratchdir -- see $buildlog")
        end
    end

    vertexlog = joinpath(prefix, "vertex.log")
    if isfile(vertexlog)
        vlog = read(vertexlog, String)
        if occursin(r"\(Y/N\)\?\s*$", vlog) || occursin("ver017", vlog)
            error("vertex stopped at an interactive prompt (relict model?) " *
                  "in $scratchdir -- see $vertexlog")
        end
        occursin("End of job", vlog) ||
            error("vertex did not finish in $scratchdir -- see $vertexlog")
    else
        error("vertex.log missing in $prefix -- vertex did not run")
    end

    blk = joinpath(prefix, "$(index).blk")
    (isfile(blk) && filesize(blk) > 0) ||
        error("Empty/missing $(index).blk in $prefix -- vertex produced no grid")
    return nothing
end

"Run one (scenario, rung): configure, vertex, query NP x NT grid of bound H2O."
function run_rung(scenario::String, rung::String, solution_phases::String,
                  composition::Vector{Float64})
    scratchdir = joinpath(SCRATCH_TOP,
                          "$(LADDER_VERSION)_$(scenario)_$(rung)") * "/"

    println("[$(scenario) | $(rung)] configuring + running vertex ..."); flush(stdout)
    t0 = time()
    perplex_configure_pseudosection(scratchdir, composition, ELEMENTS,
        P_RANGE, T_RANGE;
        dataset         = DATASET,
        index           = 1,
        xnodes          = XNODES,
        ynodes          = YNODES,
        solution_phases = solution_phases,
        excludes        = EXCLUDES,
    )
    check_perplex_run(scratchdir)
    @printf("[%s | %s] vertex done in %.1f min\n",
            scenario, rung, (time()-t0)/60); flush(stdout)

    Ts = range(first(T_RANGE), last(T_RANGE), length=NT)
    Ps = range(first(P_RANGE), last(P_RANGE), length=NP)
    grid = fill(NaN, NP, NT)

    for (i, P) in enumerate(Ps)
        for (j, T) in enumerate(Ts)
            grid[i, j] = parse_bound_h2o(perplex_query_point(scratchdir, P, T))
        end
        if i % 10 == 0
            @printf("[%s | %s] row %d / %d\n", scenario, rung, i, NP)
            flush(stdout)
        end
    end

    nnan = count(isnan, grid)
    @printf("[%s | %s] grid complete: %d / %d NaN cells\n",
            scenario, rung, nnan, length(grid)); flush(stdout)
    nnan == length(grid) &&
        error("All-NaN grid for $scenario / $rung -- check $scratchdir/out1/")

    return collect(Ts), collect(Ps), grid
end

"Write a P-T grid CSV: T(K) column headers, P(GPa) row labels, LF endings."
function write_grid_csv(path::String, Ts, Ps, grid)
    open(path, "w") do io
        print(io, "")                                  # empty top-left cell
        for T in Ts; @printf(io, ",%.2f", T); end
        print(io, "\n")
        for (i, P) in enumerate(Ps)
            @printf(io, "%.5f", P / 1e4)               # bar -> GPa
            for j in eachindex(Ts)
                isnan(grid[i,j]) ? print(io, ",NaN") : @printf(io, ",%.6f", grid[i,j])
            end
            print(io, "\n")
        end
    end
end


# =============================================================================
# 4. MAIN
# =============================================================================
function main()
    mkpath(OUT_DIR)
    println("="^70)
    println("Solution-model ladder sensitivity -- $(LADDER_VERSION)")
    println("Grid: $(NP) P x $(NT) T | H2O = $(H2O_WT) wt% | dataset = $(DATASET)")
    println("="^70); flush(stdout)

    # All (scenario x rung) jobs are independent (separate scratch dirs),
    # so parallelize the outer product. werami point queries within one
    # scratchdir MUST stay serial (shared werami.bat / 1_1.txt).
    jobs = [(s, r, sp) for s in SCENARIOS for (r, sp) in MODEL_LADDER]
    results = Dict{Tuple{String,String},Any}()
    lk = ReentrantLock()

    Threads.@threads for (scenario, rung, sphases) in jobs
        comp = vcat(load_posterior_mean(scenario), H2O_WT)
        Ts, Ps, grid = run_rung(scenario, rung, sphases, comp)
        lock(lk) do
            results[(scenario, rung)] = (Ts, Ps, grid)
        end
    end

    # -- Save grids, difference maps, and a summary table per scenario --
    for scenario in SCENARIOS
        rungs = first.(MODEL_LADDER)
        summary = ["rung,added_model,mean_boundH2O_wt,max_boundH2O_wt," *
                   "mean_delta_vs_prev,max_abs_delta_vs_prev,n_nan"]
        prev = nothing
        added = Dict("L0_v1_effective" => "(baseline)",
                     "L1_plus_amph"    => "cAmph(G)",
                     "L2_plus_talc"    => "T",
                     "L3_v2_full"      => "Sp(WPC)")

        for rung in rungs
            Ts, Ps, grid = results[(scenario, rung)]
            write_grid_csv(joinpath(OUT_DIR, "boundH2O_$(scenario)_$(rung).csv"),
                           Ts, Ps, grid)

            ok = .!isnan.(grid)
            mh, xh = sum(grid[ok])/count(ok), maximum(grid[ok])
            md, xd = NaN, NaN
            if prev !== nothing
                d = grid .- prev
                write_grid_csv(joinpath(OUT_DIR,
                    "delta_boundH2O_$(scenario)_$(rung)_minus_prev.csv"), Ts, Ps, d)
                dok = .!isnan.(d)
                md, xd = sum(d[dok])/count(dok), maximum(abs.(d[dok]))
            end
            push!(summary, @sprintf("%s,%s,%.4f,%.4f,%.4f,%.4f,%d",
                  rung, added[rung], mh, xh, md, xd, count(isnan, grid)))
            prev = grid
        end

        spath = joinpath(OUT_DIR, "summary_$(scenario).csv")
        open(spath, "w") do io; println.(Ref(io), summary); end
        println("\n--- $(scenario) ---")
        println.(summary)
        flush(stdout)
    end

    println("\nAll outputs in: $(OUT_DIR)/")
end

main()

