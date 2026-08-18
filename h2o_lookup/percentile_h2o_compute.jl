#=
percentile_h2o_v1v2.jl
======================
Cheap bound-H2O uncertainty proxy: instead of the full 200-sample ensemble,
run only 3 OXIDE-WISE PERCENTILE compositions (p05 / p50 / p95) per scenario,
under both solution-model sets (v1 and v2). 6 pseudosections per scenario.

    v1 = O(HGP) Cpx(HGP) Opx(HGP) Gt(HGP) Chl(W) Fsp(HGP)      (no amphibole)
    v2 = v1 + cAmph(G)

CAVEAT (by design choice): the p05/p95 compositions take each oxide at its
5th/95th posterior percentile INDEPENDENTLY, then renormalize to 100. No real
sample is simultaneously extreme in all 7 oxides, so this OVERSTATES the true
bound-H2O spread. It is an upper bound on compositional uncertainty, not the
rigorous band. The rigorous p05/p50/p95 = per-cell percentiles across the full
200-run ensemble (add to 5H2O_test_v2_parallel.jl when it completes).

Bound H2O is read via perplex_query_system(...; include_fluid="n"): the
fluid-excluded system H2O wt% on a SOLID-mass basis (matches the ladder's
solid-basis numbers, not the whole-rock basis).

ISOLATION: own scratch root (perplex_percentile/) and output dir
(percentile_h2o/) -- will not touch a running ensemble job.

Run on a SEPARATE allocation:
    salloc -A gts-ssim33-atlas -q inferno -N1 --ntasks-per-node=6 -t 2:00:00
    module load julia/1.11.3
    export JULIA_DEPOT_PATH=/storage/scratch1/7/hchoi342/.julia
    julia --project=. -t 6 percentile_h2o_v1v2.jl
=#

using StatGeochem, CSV, DataFrames, Statistics, Printf

# =============================================================================
# CONFIG
# =============================================================================
const REPO        = "/storage/home/hcoda1/7/hchoi342/scratch/Archean-OC-estimation"
const BAYES_DIR   = joinpath(REPO, "Bayesian", "bayesian_lower_crust_outputs")
const SCRATCH_TOP = joinpath(REPO, "perplex_percentile")   # separate from ensemble
const OUT_DIR     = joinpath(REPO, "percentile_h2o_csv")

const SCENARIOS = ["homogeneous_crust", "layered_cumulate_lower_crust"]
const SCEN_LBL  = Dict("homogeneous_crust" => "homogeneous",
                       "layered_cumulate_lower_crust" => "layered cumulate")

const OXIDES   = ["SiO2", "TiO2", "Al2O3", "FeO", "MgO", "CaO", "Na2O"]
const ELEMENTS = vcat(OXIDES, "H2O")
const H2O_WT   = 5.0

# Same wide poster domain as median_compare_v1v2.jl, for visual consistency.
const T_RANGE = (273.15, 1573.15)   # K   (0 - 1300 °C)
const P_RANGE = (100.0, 80000.0)    # bar (0.01 - 8.0 GPa)
const NT = 40
const NP = 40
const XNODES = 40
const YNODES = 40

const DATASET  = "hp62ver.dat"
const EXCLUDES = "ged\nfanth\ngl\n"

const MODEL_SETS = [
    ("v1", "O(HGP)\nCpx(HGP)\nOpx(HGP)\nGt(HGP)\nChl(W)\nFsp(HGP)\n"),
    ("v2", "O(HGP)\nCpx(HGP)\nOpx(HGP)\nGt(HGP)\nChl(W)\nFsp(HGP)\ncAmph(G)\n"),
]

const PCTS = [("p05", 0.05), ("p50", 0.50), ("p95", 0.95)]

const P_VEC = collect(range(first(P_RANGE), last(P_RANGE), length=NP))  # bar
const T_VEC = collect(range(first(T_RANGE), last(T_RANGE), length=NT))  # K
const T_C   = T_VEC .- 273.15
const P_GPa = P_VEC ./ 1e4


# =============================================================================
# HELPERS
# =============================================================================
"Oxide-wise percentile bulk composition (wt%), renormalized to 100."
function percentile_composition(scenario::String, q::Float64)
    path = joinpath(BAYES_DIR, "ensemble_$(scenario).csv")
    isfile(path) || error("Not found: $path")
    df   = CSV.read(path, DataFrame)
    cols = [findfirst(c -> startswith(String(c), ox), names(df)) for ox in OXIDES]
    any(isnothing, cols) && error("Missing oxide column in $path")
    M = Matrix{Float64}(df[:, collect(cols)])
    v = [quantile(M[:, k], q) for k in 1:length(OXIDES)]   # oxide-wise percentile
    return 100.0 .* v ./ sum(v)                            # renormalize to 100
end

"Fail-fast: catch build rejections / vertex prompts before querying."
function check_run(scratch::String; index::Int=1)
    pre = joinpath(scratch, "out$(index)")
    blog = joinpath(pre, "build.log")
    if isfile(blog)
        for m in eachmatch(r"(\S+)\s+is invalid\.", read(blog, String))
            error("build rejected '$(m.captures[1])' in $scratch")
        end
    end
    vlog = joinpath(pre, "vertex.log")
    isfile(vlog) || error("vertex.log missing in $pre")
    v = read(vlog, String)
    (occursin(r"\(Y/N\)\?\s*$", v) || occursin("ver017", v)) &&
        error("vertex stopped at interactive prompt in $scratch")
    occursin("End of job", v) || error("vertex did not finish in $scratch")
end

"Build one pseudosection; return NP x NT grid of solid-basis bound H2O (wt%)."
function bound_h2o_grid(setname::String, pct::String, scenario::String,
                        comp7::Vector{Float64})
    scratch = joinpath(SCRATCH_TOP, "$(setname)_$(pct)_$(scenario)") * "/"
    sphases = Dict(MODEL_SETS)[setname]
    comp = vcat(comp7, H2O_WT)

    println("[$setname | $pct | $scenario] configure + vertex ..."); flush(stdout)
    perplex_configure_pseudosection(scratch, comp, ELEMENTS, P_RANGE, T_RANGE;
        dataset=DATASET, index=1, xnodes=XNODES, ynodes=YNODES,
        solution_phases=sphases, excludes=EXCLUDES)
    check_run(scratch)

    # Flatten the grid (i over P = rows, j over T = cols), query in one call.
    Pflat = Float64[]; Tflat = Float64[]
    for i in 1:NP, j in 1:NT
        push!(Pflat, P_VEC[i]); push!(Tflat, T_VEC[j])
    end
    res = perplex_query_system(scratch, Pflat, Tflat; include_fluid="n")

    grid = fill(NaN, NP, NT)
    if haskey(res, "H2O")
        h = res["H2O"]
        for i in 1:NP, j in 1:NT
            k = (i - 1) * NT + j
            if k <= length(h)
                x = h[k]
                grid[i, j] = (x isa Number && !isnan(x)) ? max(0.0, Float64(x)) : NaN
            end
        end
    else
        @warn "no H2O column for $setname/$pct/$scenario; keys=$(collect(keys(res)))"
    end
    return grid
end

"Write one P-T grid CSV: T(K) headers, P(GPa) row labels (matches read_grid)."
function write_grid_csv(path, grid)
    open(path, "w") do io
        print(io, "")
        for T in T_VEC; @printf(io, ",%.2f", T); end
        print(io, "\n")
        for i in 1:NP
            @printf(io, "%.5f", P_VEC[i] / 1e4)        # bar -> GPa
            for j in 1:NT
                isnan(grid[i,j]) ? print(io, ",NaN") : @printf(io, ",%.6f", grid[i,j])
            end
            print(io, "\n")
        end
    end
end

# =============================================================================
# MAIN
# =============================================================================
function main()
    mkpath(OUT_DIR)
    println("="^64)
    println("Oxide-wise p05/p50/p95 bound-H2O (CSV only) - v1 vs v2")
    println("grid $(NP)x$(NT) | H2O=$(H2O_WT) wt% | $(DATASET) | solid basis")
    println("="^64); flush(stdout)

    # 12 independent jobs: scenario x model-set x percentile.
    jobs = [(sc, s, pl, pq) for sc in SCENARIOS
                            for (s, _) in MODEL_SETS
                            for (pl, pq) in PCTS]
    lk = ReentrantLock()
    Threads.@threads for (scenario, setname, plabel, q) in jobs
        outcsv = joinpath(OUT_DIR, "boundH2O_$(scenario)_$(setname)_$(plabel).csv")
        if isfile(outcsv)
            println("[skip - exists] $(basename(outcsv))"); flush(stdout); continue
        end
        try
            comp = percentile_composition(scenario, q)
            g = bound_h2o_grid(setname, plabel, scenario, comp)
            write_grid_csv(outcsv, g)               # checkpoint immediately
            nnan = count(isnan, g)
            lock(lk) do
                @printf("[done] %s  (%d/%d NaN cells)\n", basename(outcsv),
                        nnan, length(g)); flush(stdout)
            end
        catch e
            lock(lk) do; @warn "failed: $(basename(outcsv))" exception=e; end
        end
    end

    println("\nCSVs in: $(OUT_DIR)/  -> scp to Mac and plot with plot_wilson_style.py")
end

main()

