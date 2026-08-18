#=
median_compare_v1v2.jl
======================
Quick diagnostic: run ONLY the median (p50) bulk composition through
Perple_X for both crustal scenarios, under two solution-model sets, and
plot the difference amphibole makes.

    v1 = v1-effective set (chlorite + pure hydrous phases, NO amphibole)
         O(HGP) Cpx(HGP) Opx(HGP) Gt(HGP) Chl(W) Fsp(HGP)
    v2 = v1 + cAmph(G)   (Green et al. 2016 clinoamphibole)

Produces two 2x2 comparison figures (rows = scenario, cols = v1 | v2):
    fig_hydrous_phase.png  -- dominant hydrous phase
    fig_assemblage.png     -- full phase assemblage (>threshold vol%)

ISOLATION: uses its own scratch root (perplex_median/) and output dir
(median_compare/), so it will NOT collide with a running ensemble job in
perplex_ensemble/ or lookup_tables/.

Run on a SEPARATE allocation from the ensemble job, e.g.:
    salloc -A gts-ssim33-atlas -q inferno -N1 --ntasks-per-node=4 -t 1:00:00
    module load julia/1.11.3
    export JULIA_DEPOT_PATH=/storage/scratch1/7/hchoi342/.julia
    julia --project=. -t 4 median_compare_v1v2.jl
=#

using StatGeochem, CSV, DataFrames, Statistics, LinearAlgebra, Plots

# =============================================================================
# CONFIG
# =============================================================================
const REPO        = "/storage/home/hcoda1/7/hchoi342/scratch/Archean-OC-estimation"
const BAYES_DIR   = joinpath(REPO, "Bayesian", "bayesian_lower_crust_outputs")
const SCRATCH_TOP = joinpath(REPO, "perplex_median")    # separate from ensemble
const OUT_DIR     = joinpath(REPO, "median_compare")

const SCENARIOS = ["homogeneous_crust", "layered_cumulate_lower_crust"]
const SCEN_LBL  = Dict("homogeneous_crust" => "homogeneous",
                       "layered_cumulate_lower_crust" => "layered cumulate")

# 7-oxide NCFMAST + H2O. hp62ver.dat names are CASE-SENSITIVE, mixed-case.
const OXIDES   = ["SiO2", "TiO2", "Al2O3", "FeO", "MgO", "CaO", "Na2O"]
const ELEMENTS = vcat(OXIDES, "H2O")
const H2O_WT   = 5.0

# Wider domain for the poster figure. NOTE: this intentionally DIFFERS from
# the ensemble grid (473-1473 K / 500-30000 bar) -- these standalone v1-vs-v2
# panels are internally consistent with each other, but do not overlay them
# on ensemble-derived maps expecting identical axes.
const T_RANGE = (273.15, 1573.15)   # K   (0 - 1300 °C)
const P_RANGE = (100.0, 80000.0)    # bar (0.01 - 8.0 GPa)
# Grid resolution: heatmap field boundaries are blocky at 40x40 over this
# wider domain. 60x60 gives noticeably smoother fields for the poster at
# little cost (4 single-composition runs). Lower back to 40 if vertex is slow.
const NT = 60
const NP = 60
const XNODES = 60
const YNODES = 60

const DATASET  = "hp62ver.dat"
const EXCLUDES = "ged\nfanth\ngl\n"

const MODEL_SETS = [
    ("v1", "O(HGP)\nCpx(HGP)\nOpx(HGP)\nGt(HGP)\nChl(W)\nFsp(HGP)\n"),
    ("v2", "O(HGP)\nCpx(HGP)\nOpx(HGP)\nGt(HGP)\nChl(W)\nFsp(HGP)\ncAmph(G)\n"),
]

# Hydrous phases to track for the "dominant hydrous phase" panel.
# (canonical names; immiscible duplicates like cAmph(G)_2 are folded in.)
const HYDROUS = ["cAmph(G)", "Chl(W)", "ta", "law", "zo", "cz",
                 "phA", "atg", "liz", "br", "tr", "anth"]

# Phases to track for the assemblage panel (hydrous + major anhydrous).
const TRACKED = vcat(HYDROUS,
    ["Gt(HGP)", "Cpx(HGP)", "Opx(HGP)", "O(HGP)", "Fsp(HGP)",
     "sph", "ru", "q", "coe", "ky"])

const ASSEM_THRESHOLD = 2.0   # vol% to count a phase as present

const P_VEC = collect(range(first(P_RANGE), last(P_RANGE), length=NP))  # bar
const T_VEC = collect(range(first(T_RANGE), last(T_RANGE), length=NT))  # K
const T_C   = T_VEC .- 273.15
const P_GPa = P_VEC ./ 1e4


# =============================================================================
# HELPERS
# =============================================================================
"Oxide-wise median (p50) bulk composition (wt%) for a scenario."
function median_composition(scenario::String)
    path = joinpath(BAYES_DIR, "ensemble_$(scenario).csv")
    isfile(path) || error("Not found: $path")
    df   = CSV.read(path, DataFrame)
    cols = [findfirst(c -> startswith(String(c), ox), names(df)) for ox in OXIDES]
    any(isnothing, cols) && error("Missing oxide column in $path")
    M = Matrix{Float64}(df[:, collect(cols)])
    return vec(median(M, dims=1))   # 7-element p50 vector
end

"Strip werami's immiscibility suffix so cAmph(G)_2 folds into cAmph(G)."
canonical(k::AbstractString) = replace(k, r"_\d+$" => "")

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

"Build one pseudosection and return phase-mode grids Dict(phase => NP x NT)."
function phase_grids_for(setname::String, scenario::String, comp7::Vector{Float64})
    scratch = joinpath(SCRATCH_TOP, "$(setname)_$(scenario)") * "/"
    sphases = Dict(MODEL_SETS)[setname]
    comp = vcat(comp7, H2O_WT)

    println("[$setname | $scenario] configure + vertex ..."); flush(stdout)
    perplex_configure_pseudosection(scratch, comp, ELEMENTS, P_RANGE, T_RANGE;
        dataset=DATASET, index=1, xnodes=XNODES, ynodes=YNODES,
        solution_phases=sphases, excludes=EXCLUDES)
    check_run(scratch)

    test = perplex_query_modes(scratch, [P_VEC[1]], [T_VEC[1]])
    found = sort(unique(canonical(k) for k in keys(test)
                        if k ∉ ("T(K)", "P(bar)", "elements")))
    println("[$setname | $scenario] phases found: ", found); flush(stdout)

    grids = Dict{String, Matrix{Float64}}()
    for (j, P) in enumerate(P_VEC)
        res = perplex_query_modes(scratch, fill(P, NT), T_VEC)
        for (k, vals) in res
            k in ("T(K)", "P(bar)", "elements") && continue
            ck = canonical(k)
            g  = get!(grids, ck, fill(NaN, NP, NT))
            for t in 1:NT
                x = vals[t]
                isnan(x) && continue
                g[j, t] = isnan(g[j, t]) ? x : g[j, t] + x   # fold duplicates
            end
        end
    end
    return grids
end

"Dominant hydrous phase (string grid) from phase-mode grids."
function dominant_hydrous(grids)
    dom = fill("", NP, NT)
    for i in 1:NP, j in 1:NT
        best, bestv = "", 0.0
        for ph in HYDROUS
            haskey(grids, ph) || continue
            v = grids[ph][i, j]
            if !isnan(v) && v > bestv
                best, bestv = ph, v
            end
        end
        dom[i, j] = best
    end
    return dom
end

"Assemblage label grid (sorted '+'-joined names above threshold)."
function assemblage(grids)
    asm = fill("", NP, NT)
    for i in 1:NP, j in 1:NT
        present = String[]
        for ph in TRACKED
            haskey(grids, ph) || continue
            v = grids[ph][i, j]
            (!isnan(v) && v > ASSEM_THRESHOLD) && push!(present, ph)
        end
        asm[i, j] = join(sort(present), "+")
    end
    return asm
end

"Heatmap of a categorical string grid using a shared code map; annotate fields."
function category_panel(strgrid, codemap, title; wrap=false)
    ncat = max(length(codemap), 1)
    z = [get(codemap, strgrid[i, j], 0) / ncat for i in 1:NP, j in 1:NT]
    pl = heatmap(T_C, P_GPa, z; c=:tab20, clims=(0.0, 1.0), colorbar=false,
                 xlabel="T (°C)", ylabel="P (GPa)", title=title, titlefontsize=9)
    for (lbl, _) in codemap
        mask = strgrid .== lbl
        idxs = findall(mask); isempty(idxs) && continue
        mi = mean(getindex.(idxs, 1)); mj = mean(getindex.(idxs, 2))
        ti = T_C[clamp(round(Int, mj), 1, NT)]
        pp = P_GPa[clamp(round(Int, mi), 1, NP)]
        txt = wrap ? replace(lbl, "+" => "+\n") : lbl
        annotate!(pl, ti, pp, text(txt, 5, :white, :center))
    end
    return pl
end


# =============================================================================
# MAIN
# =============================================================================
function main()
    mkpath(OUT_DIR)
    println("="^64)
    println("Median (p50) v1-vs-v2 pseudosection comparison")
    println("grid $(NP)x$(NT) | H2O=$(H2O_WT) wt% | $(DATASET)")
    println("="^64); flush(stdout)

    # Build all four pseudosections (independent -> parallel over jobs).
    jobs = [(s, sc) for (s, _) in MODEL_SETS for sc in SCENARIOS]
    results = Dict{Tuple{String,String}, Any}()
    lk = ReentrantLock()
    Threads.@threads for (setname, scenario) in jobs
        comp = median_composition(scenario)
        g = phase_grids_for(setname, scenario, comp)
        lock(lk) do; results[(setname, scenario)] = g; end
    end

    # ---- Figure 1: dominant hydrous phase ----
    doms = Dict((s, sc) => dominant_hydrous(results[(s, sc)])
                for (s, _) in MODEL_SETS for sc in SCENARIOS)
    hyd_active = sort(unique(filter(!isempty,
        vcat([vec(d) for d in values(doms)]...))))
    hyd_code = Dict(p => i for (i, p) in enumerate(hyd_active))
    println("\nActive hydrous phases (all panels): ", hyd_active)

    panels1 = []
    for sc in SCENARIOS, (s, _) in MODEL_SETS
        push!(panels1, category_panel(doms[(s, sc)], hyd_code,
              "$(SCEN_LBL[sc]) — $s"))
    end
    fig1 = plot(panels1...; layout=(2, 2), size=(1600, 1150),
                plot_title="Dominant hydrous phase (p50 bulk)",
                margin=5Plots.mm, dpi=300)
    savefig(fig1, joinpath(OUT_DIR, "fig_hydrous_phase.pdf"))   # vector, poster
    savefig(fig1, joinpath(OUT_DIR, "fig_hydrous_phase.png"))   # quick preview

    # ---- Figure 2: full assemblage ----
    asms = Dict((s, sc) => assemblage(results[(s, sc)])
                for (s, _) in MODEL_SETS for sc in SCENARIOS)
    asm_active = sort(unique(filter(!isempty,
        vcat([vec(a) for a in values(asms)]...))))
    asm_code = Dict(a => i for (i, a) in enumerate(asm_active))
    println("Unique assemblages (all panels): ", length(asm_active))

    panels2 = []
    for sc in SCENARIOS, (s, _) in MODEL_SETS
        push!(panels2, category_panel(asms[(s, sc)], asm_code,
              "$(SCEN_LBL[sc]) — $s"; wrap=true))
    end
    fig2 = plot(panels2...; layout=(2, 2), size=(1800, 1250),
                plot_title="Phase assemblage (p50 bulk, >$(ASSEM_THRESHOLD) vol%)",
                margin=5Plots.mm, dpi=300)
    savefig(fig2, joinpath(OUT_DIR, "fig_assemblage.pdf"))      # vector, poster
    savefig(fig2, joinpath(OUT_DIR, "fig_assemblage.png"))      # quick preview

    println("\nSaved to $(OUT_DIR)/:")
    println("  fig_hydrous_phase.pdf / .png")
    println("  fig_assemblage.pdf / .png")
end

main()

