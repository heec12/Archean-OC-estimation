#=
5H2O_test_v2_parallel.jl
========================
Full 200-sample posterior ensemble of bound-H2O P-T lookup tables for the
v2-amphibole solution model set (v1-effective + cAmph(G)).

Justification (from the sensitivity ladder): adding cAmph(G) dominates the
bound-H2O budget across the greenschist-amphibolite-blueschist window;
talc (T) and spinel (Sp(WPC)) contribute little, so the scientifically
essential v2 case is "v1 plus an amphibole model." This script runs that
case over the full posterior ensemble to propagate Bayesian bulk-composition
uncertainty into the P-T-H2O lookup tables.

KEY DIFFERENCE FROM THE LADDER SCRIPT
  Bound H2O is read in ONE werami call per sample (2-D grid mode, property
  36, system composition, fluid EXCLUDED) instead of 1600 point queries.
  Verified against Perple_X 7.1.9: the system H2O,wt% column with fluid
  excluded equals bound H2O (cold nodes -> full bound, hot nodes -> 0).
  Falls back to per-point queries if QUERY_MODE == :point.

Verified solution-phase set (Perple_X 7.1.9 + hp62ver.dat + v719 models):
  do NOT use Amph(DPW) (vertex ver017 prompt -> stdin EOF crash),
  Sp(HGP) (needs ds633 make defs), Ep(HP)/Pheng(HP) (need O2/K2O).

Usage:
    julia --project=. -t 10 5H2O_test_v2_parallel.jl
    # or on PACE:  export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
=#

using StatGeochem
using StatGeochem.Perple_X_jll
using DelimitedFiles
using Printf

# =============================================================================
# 1. CONFIG
# =============================================================================
const PERPLEX_VERSION = let
    # Derive "v2" from 5H2O_test_v2_parallel.jl (Perple_X solution-model axis).
    m = match(r"(v\d+\w*?)_parallel", basename(@__FILE__))
    m === nothing ? "v2" : m.captures[1]
end

# Bayesian artifacts are UNVERSIONED (shared across Perple_X versions).
const BAYES_DIR = get(ENV, "BAYES_DIR",
    joinpath(dirname(@__DIR__), "Bayesian", "bayesian_lower_crust_outputs"))

const SCRATCH_TOP = joinpath(dirname(@__DIR__), "perplex_ensemble")
const OUT_DIR     = joinpath(dirname(@__DIR__), "lookup_tables",
                             "$(PERPLEX_VERSION)")

const SCENARIOS = ["homogeneous_crust", "layered_cumulate_lower_crust"]

# 7-oxide NCFMAST + H2O. hp62ver.dat names are CASE-SENSITIVE, mixed-case.
const OXIDES   = ["SiO2", "TiO2", "Al2O3", "FeO", "MgO", "CaO", "Na2O"]
const ELEMENTS = vcat(OXIDES, "H2O")

# >>> Match these to your downstream / ladder settings.
const H2O_WT   = 5.0
const T_RANGE  = (473.15, 1473.15)   # K
const P_RANGE  = (500.0, 30000.0)    # bar (0.05-3.0 GPa)
const NT = 40                        # canonical output T nodes
const NP = 40                        # canonical output P nodes
const XNODES = 40                    # vertex exploratory grid (x = T)
const YNODES = 40

const DATASET  = "hp62ver.dat"
const EXCLUDES = "ged\nfanth\ngl\n"

# v2-amphibole solution model set (= ladder rung L1)
const SOLUTION_PHASES =
    "O(HGP)\nCpx(HGP)\nOpx(HGP)\nGt(HGP)\nChl(W)\nFsp(HGP)\ncAmph(G)\n"

const ENSEMBLE_SIZE = 200
const QUERY_MODE    = :grid   # :grid (one call/sample) or :point (1600/sample)

const WERAMI = joinpath(Perple_X_jll.PATH[], "werami")


# =============================================================================
# 2. INPUT
# =============================================================================
"Read the ENSEMBLE_SIZE x 7 posterior bulk compositions (wt%) for a scenario."
function load_ensemble(scenario::String)
    path = joinpath(BAYES_DIR, "ensemble_$(scenario).csv")
    isfile(path) || error("Not found: $path")
    raw, head = readdlm(path, ',', header=true)
    head = vec(String.(head))
    cols = [findfirst(h -> startswith(h, ox), head) for ox in OXIDES]
    any(isnothing, cols) && error("Missing oxide column in $path")
    comps = Float64.(raw[:, cols])
    n = min(ENSEMBLE_SIZE, size(comps, 1))
    return comps[1:n, :]
end


# =============================================================================
# 3. BOUND-H2O VIA SINGLE GRID QUERY  (fast path)
# =============================================================================
"""
One werami call: 2-D grid (mode 2), property 36 (all system properties),
option 1 (system only), fluid EXCLUDED. Returns (Tvec, Pvec, H2Ogrid) where
H2Ogrid is the system H2O wt% with fluid removed = bound H2O. The returned
T/P node vectors are whatever werami used (depends on grid_levels in the
option file); callers interpolate onto the canonical NT x NP axes.
"""
function query_grid_boundh2o(scratchdir::String; index::Int=1)
    prefix = joinpath(scratchdir, "out$(index)") * "/"

    # Ensure grid-on path (avoids the "change variable range" prompt that the
    # grid-off path injects). Finest resolution menu entry chosen below.
    run(pipeline(`sed -e "s/sample_on_grid .*|/sample_on_grid                   T |/" -i.backup $(prefix)perplex_option.dat`))

    batch = prefix * "werami_grid.bat"
    open(batch, "w") do io
        # index, mode2, prop36, system-only, exclude fluid, finest grid (4), exit
        write(io, "$index\n2\n36\n1\nn\n4\n0\n")
    end
    rm(prefix * "$(index)_1.tab", force=true)
    ld = "DYLD_LIBRARY_PATH=$(first(Perple_X_jll.LIBPATH_list)):\$DYLD_LIBRARY_PATH"
    run(pipeline(`bash -c "export $ld; cd $prefix; $WERAMI < werami_grid.bat > werami_grid.log 2>&1"`))

    tab = prefix * "$(index)_1.tab"
    isfile(tab) || error("werami produced no .tab in $prefix")
    return parse_tab_boundh2o(tab)
end

"Parse a werami 2-D .tab: returns (Tvec, Pvec, H2Ogrid[NP_w, NT_w])."
function parse_tab_boundh2o(tabpath::String)
    lines = readlines(tabpath)
    # Header block (|6.6.6 format): line 3 = ndim; then per axis:
    #   name / min / delta / npts.  Then column-count, column-names, data.
    # Robust approach: find the column-name row (contains "T(K)" and "H2O").
    icol = findfirst(l -> occursin("T(K)", l) && occursin("H2O", l), lines)
    icol === nothing && error("Could not find column header in $tabpath")
    cols = split(strip(lines[icol]))
    jT   = findfirst(==("T(K)"), cols)
    jP   = findfirst(==("P(bar)"), cols)
    jH2O = findfirst(c -> startswith(c, "H2O"), cols)
    (jT === nothing || jP === nothing || jH2O === nothing) &&
        error("Missing T/P/H2O column in $tabpath")

    Ts = Float64[]; Ps = Float64[]; Hs = Float64[]
    for l in lines[icol+1:end]
        tok = split(strip(l))
        length(tok) < jH2O && continue
        t = tryparse(Float64, tok[jT]); p = tryparse(Float64, tok[jP])
        h = tryparse(Float64, tok[jH2O])
        (t === nothing || p === nothing) && continue
        push!(Ts, t); push!(Ps, p)
        push!(Hs, h === nothing ? NaN : max(0.0, h))
    end

    Tu = sort(unique(Ts)); Pu = sort(unique(Ps))
    grid = fill(NaN, length(Pu), length(Tu))
    for k in eachindex(Hs)
        i = searchsortedfirst(Pu, Ps[k]); j = searchsortedfirst(Tu, Ts[k])
        grid[i, j] = Hs[k]
    end
    return Tu, Pu, grid
end

"Bilinear-interpolate a (Pvec,Tvec,grid) onto the canonical NP x NT axes."
function regrid(Tvec, Pvec, grid, Tout, Pout)
    out = fill(NaN, length(Pout), length(Tout))
    for (i, P) in enumerate(Pout), (j, T) in enumerate(Tout)
        ip = clamp(searchsortedlast(Pvec, P), 1, length(Pvec)-1)
        jt = clamp(searchsortedlast(Tvec, T), 1, length(Tvec)-1)
        P1, P2 = Pvec[ip], Pvec[ip+1]; T1, T2 = Tvec[jt], Tvec[jt+1]
        fp = (P - P1) / (P2 - P1); ft = (T - T1) / (T2 - T1)
        q11, q12 = grid[ip, jt],   grid[ip, jt+1]
        q21, q22 = grid[ip+1, jt], grid[ip+1, jt+1]
        any(isnan, (q11, q12, q21, q22)) && continue
        out[i, j] = (1-fp)*(1-ft)*q11 + (1-fp)*ft*q12 + fp*(1-ft)*q21 + fp*ft*q22
    end
    return out
end


# =============================================================================
# 4. BOUND-H2O VIA POINT QUERIES  (fallback, :point mode)
# =============================================================================
"Parse bound H2O (solid-only wt%) from one perplex_query_point text block."
function parse_point_boundh2o(text::AbstractString)
    isempty(strip(text)) && return NaN
    lines = split(text, '\n')
    ib = findfirst(l -> startswith(strip(l), "Bulk Composition"), lines)
    ib === nothing && return NaN
    iend = min(ib + 15, length(lines))
    solid = any(occursin("Solid Only", l) for l in lines[ib:min(ib+3,length(lines))])
    for l in lines[ib:iend]
        tok = split(strip(l)); isempty(tok) && continue
        tok[1] == "H2O" || continue
        if solid && length(tok) >= 9
            v = tryparse(Float64, tok[8]); return v === nothing ? NaN : max(0.0, v)
        elseif length(tok) >= 4
            v = tryparse(Float64, tok[4]); return v === nothing ? NaN : max(0.0, v)
        end
    end
    return NaN
end

function query_point_grid(scratchdir, Tout, Pout)
    grid = fill(NaN, length(Pout), length(Tout))
    for (i, P) in enumerate(Pout), (j, T) in enumerate(Tout)
        grid[i, j] = parse_point_boundh2o(perplex_query_point(scratchdir, P, T))
    end
    return grid
end


# =============================================================================
# 5. FAIL-FAST + PER-SAMPLE DRIVER
# =============================================================================
function check_perplex_run(scratchdir::String; index::Int=1)
    prefix = joinpath(scratchdir, "out$(index)")
    blog = joinpath(prefix, "build.log")
    if isfile(blog)
        for m in eachmatch(r"(\S+)\s+is invalid\.", read(blog, String))
            error("build rejected '$(m.captures[1])' in $scratchdir")
        end
    end
    vlog = joinpath(prefix, "vertex.log")
    isfile(vlog) || error("vertex.log missing in $prefix")
    v = read(vlog, String)
    (occursin(r"\(Y/N\)\?\s*$", v) || occursin("ver017", v)) &&
        error("vertex stopped at interactive prompt in $scratchdir")
    occursin("End of job", v) || error("vertex did not finish in $scratchdir")
    blk = joinpath(prefix, "$(index).blk")
    (isfile(blk) && filesize(blk) > 0) || error("empty $(index).blk in $prefix")
end

"Configure + vertex + bound-H2O grid for one ensemble sample."
function run_sample(scenario::String, i::Int, comp7::AbstractVector,
                    Tout, Pout)
    scratch = joinpath(SCRATCH_TOP,
                       "$(PERPLEX_VERSION)_$(scenario)_sample_$(i)") * "/"
    comp = vcat(collect(comp7), H2O_WT)

    perplex_configure_pseudosection(scratch, comp, ELEMENTS, P_RANGE, T_RANGE;
        dataset=DATASET, index=1, xnodes=XNODES, ynodes=YNODES,
        solution_phases=SOLUTION_PHASES, excludes=EXCLUDES)
    check_perplex_run(scratch)

    if QUERY_MODE == :grid
        Tv, Pv, g = query_grid_boundh2o(scratch)
        grid = regrid(Tv, Pv, g, Tout, Pout)
    else
        grid = query_point_grid(scratch, Tout, Pout)
    end

    nnan = count(isnan, grid)
    nnan == length(grid) && error("all-NaN grid for $scenario sample $i")
    return grid
end


# =============================================================================
# 6. OUTPUT
# =============================================================================
"Write one P-T grid: T(K) headers, P(GPa) row labels, LF endings."
function write_grid_csv(path, Tout, Pout, grid)
    open(path, "w") do io
        print(io, "")
        for T in Tout; @printf(io, ",%.2f", T); end
        print(io, "\n")
        for (i, P) in enumerate(Pout)
            @printf(io, "%.5f", P / 1e4)
            for j in eachindex(Tout)
                isnan(grid[i,j]) ? print(io, ",NaN") : @printf(io, ",%.6f", grid[i,j])
            end
            print(io, "\n")
        end
    end
end


# =============================================================================
# 7. MAIN
# =============================================================================
function main()
    mkpath(OUT_DIR)
    Tout = collect(range(first(T_RANGE), last(T_RANGE), length=NT))
    Pout = collect(range(first(P_RANGE), last(P_RANGE), length=NP))

    println("="^70)
    println("Ensemble bound-H2O lookup tables -- $(PERPLEX_VERSION)")
    println("Models: $(replace(strip(SOLUTION_PHASES), '\n' => ' '))")
    println("Samples: $(ENSEMBLE_SIZE) | grid: $(NP)x$(NT) | query: $(QUERY_MODE)")
    println("="^70); flush(stdout)

    for scenario in SCENARIOS
        comps = load_ensemble(scenario)
        n = size(comps, 1)
        println("\n[$scenario] $n samples"); flush(stdout)

        # Accumulate the ensemble as a 3-D stack for mean/std.
        stack = fill(NaN, n, NP, NT)
        done  = falses(n)
        lk = ReentrantLock()

        Threads.@threads for i in 1:n
            try
                g = run_sample(scenario, i, view(comps, i, :), Tout, Pout)
                lock(lk) do; stack[i, :, :] = g; done[i] = true; end
            catch e
                lock(lk) do
                    @warn "sample $i failed" exception=e
                end
            end
            if i % 10 == 0
                @printf("[%s] sample %d / %d\n", scenario, i, n); flush(stdout)
            end
        end

        nok = count(done)
        @printf("[%s] completed %d / %d samples\n", scenario, nok, n)
        nok == 0 && (@warn "no samples succeeded for $scenario"; continue)

        # Per-sample lookup tables
        for i in findall(done)
            write_grid_csv(
                joinpath(OUT_DIR, "boundH2O_$(scenario)_sample_$(i).csv"),
                Tout, Pout, stack[i, :, :])
        end

        # Ensemble mean and std (over successful samples, per cell)
        meang = fill(NaN, NP, NT); stdg = fill(NaN, NP, NT)
        for a in 1:NP, b in 1:NT
            col = [stack[i, a, b] for i in findall(done)]
            ok  = filter(!isnan, col)
            if !isempty(ok)
                meang[a, b] = sum(ok) / length(ok)
                stdg[a, b]  = length(ok) > 1 ?
                    sqrt(sum((ok .- meang[a,b]).^2) / (length(ok)-1)) : 0.0
            end
        end
        write_grid_csv(joinpath(OUT_DIR, "boundH2O_$(scenario)_ensemble_mean.csv"),
                       Tout, Pout, meang)
        write_grid_csv(joinpath(OUT_DIR, "boundH2O_$(scenario)_ensemble_std.csv"),
                       Tout, Pout, stdg)
        ok = .!isnan.(meang)
        @printf("[%s] mean bound H2O = %.3f wt%%, max = %.3f wt%%\n",
                scenario, sum(meang[ok])/count(ok), maximum(meang[ok]))
        flush(stdout)
    end

    println("\nLookup tables in: $(OUT_DIR)/")
end

main()
