using StatGeochem, CSV, DataFrames, Statistics, LinearAlgebra, Plots

const P_VEC = collect(range(0.0001 * 10000, 8.0 * 10000, length=40))  # bar
const T_VEC = collect(range(273.0, 1600.0, length=40))                 # Kelvin

scratchdir = "/tmp/perplex_ensemble/homogeneous_crust_sample_41"

# =============================================================================
# BUILD 2D PHASE GRID
# =============================================================================
n_P = length(P_VEC)
n_T = length(T_VEC)

# Get phase names from a single point
test        = perplex_query_modes(scratchdir, [P_VEC[1]], [T_VEC[1]])
phase_names = [k for k in keys(test) if k ∉ ["T(K)", "P(bar)", "elements"]]
println("Phases found: ", phase_names)

# Storage
phase_grids = Dict(p => fill(NaN, n_P, n_T) for p in phase_names)

for (j, P) in enumerate(P_VEC)
    result = perplex_query_modes(scratchdir, fill(P, n_T), T_VEC)
    for phase in phase_names
        if haskey(result, phase)
            phase_grids[phase][j, :] = result[phase]
        end
    end
end

println("Grid built: ", n_P, " × ", n_T)

# =============================================================================
# COMPUTE BOUND H2O (vol% of hydrous solid phases)
# =============================================================================
hydrous_solids = ["Chl(W)", "law", "zo", "cz", "phA", "liz", "br"]
bound_h2o_grid = fill(0.0, n_P, n_T)
for phase in hydrous_solids
    if haskey(phase_grids, phase)
        bound_h2o_grid .+= replace(phase_grids[phase], NaN => 0.0)
    end
end
println("Bound H2O vol% range: ", minimum(bound_h2o_grid), " to ", maximum(bound_h2o_grid))

T_K   = T_VEC
P_GPa = P_VEC ./ 10000

# =============================================================================
# DOMINANT HYDROUS PHASE GRID
# =============================================================================
hydrous = ["Chl(W)", "law", "zo", "cz", "phA", "liz", "br"]

dominant = fill("", n_P, n_T)
for i in 1:n_P, j in 1:n_T
    best_phase = ""
    best_vol   = 0.0
    for phase in hydrous
        if haskey(phase_grids, phase)
            v = phase_grids[phase][i, j]
            if !isnan(v) && v > best_vol
                best_vol   = v
                best_phase = phase
            end
        end
    end
    dominant[i, j] = best_phase
end

all_phases = unique(filter(!isempty, vec(dominant)))
println("Active hydrous phases: ", all_phases)
code     = Dict(p => i for (i, p) in enumerate(all_phases))
code[""] = 0
dom_code = [code[dominant[i,j]] for i in 1:n_P, j in 1:n_T]

# =============================================================================
# LOAD ENSEMBLE H2O UNCERTAINTY CSVs
# =============================================================================
function read_cr_csv(path)
    raw    = read(path, String)
    rows   = split(raw, '\r')
    header = split(rows[1], ',')[2:end]
    data   = zeros(Float64, length(rows)-1, length(header))
    p_vals = Float64[]
    for (i, row) in enumerate(rows[2:end])
        isempty(strip(row)) && continue
        cells = split(row, ',')
        push!(p_vals, parse(Float64, cells[1]))
        for (j, c) in enumerate(cells[2:end])
            data[i, j] = isempty(c) ? NaN : parse(Float64, c)
        end
    end
    return p_vals, parse.(Float64, header), data
end

std_path  = "/Users/hchoi342/Documents/Archean-OC/h2o_lookup/h2o_bound_std_homogeneous_crust.csv"
mean_path = "/Users/hchoi342/Documents/Archean-OC/h2o_lookup/h2o_bound_mean_homogeneous_crust.csv"

p_ens, t_ens, h2o_mean_ens = read_cr_csv(mean_path)
_,     _,     h2o_std_ens  = read_cr_csv(std_path)

println("Ensemble mean H2O range: ", minimum(filter(!isnan, h2o_mean_ens)), " to ", maximum(filter(!isnan, h2o_mean_ens)))
println("Ensemble std  H2O range: ", minimum(filter(!isnan, h2o_std_ens)),  " to ", maximum(filter(!isnan, h2o_std_ens)))

# =============================================================================
# PLOTS
# =============================================================================

# Colour palette for hydrous phases — one colour per phase
phase_colors = Dict(
    "Chl(W)" => :teal,
    "law"    => :mediumorchid,
    "zo"     => :goldenrod,
    "cz"     => :orange,
    "phA"    => :crimson,
    "liz"    => :forestgreen,
    "br"     => :steelblue,
)
default_color = :lightgray  # anhydrous / no dominant hydrous phase

# Build colour matrix for heatmap
color_idx   = palette(:tab10)
phase_list  = collect(all_phases)   # ordered list of active phases
n_phases    = length(phase_list)
dom_float   = [code[dominant[i,j]] for i in 1:n_P, j in 1:n_T] ./ max(n_phases, 1)

# ── Plot 1 (top-left): Dominant hydrous phase with legend ────────────────────
p1 = heatmap(
    T_K .- 273.15, P_GPa, dom_float,
    c        = :tab10,
    xlabel   = "Temperature (°C)",
    ylabel   = "Pressure (GPa)",
    title    = "Dominant hydrous phase (sample 41)",
    colorbar = false,
    clims    = (0.0, 1.0),
)
# Add legend as text annotations in a corner box
# legend_x = maximum(T_K .- 273.15) * 0.65
# legend_y_start = maximum(P_GPa) * 0.95
# dy = maximum(P_GPa) * 0.07
# for (k, phase) in enumerate(phase_list)
#     frac = code[phase] / max(n_phases, 1)
#     # Draw a small filled rectangle as colour swatch
#     px = legend_x
#     py = legend_y_start - (k-1) * dy
#     annotate!(p1, px - 0.02 * maximum(T_K .- 273.15), py,
#         text("■", 14, color_idx[code[phase]], :right))
#     annotate!(p1, px, py,
#         text(" " * phase, 8, :black, :left))
# end

# ── Plot 2 (top-right): Full phase diagram — all phases ──────────────────────
threshold = 2.0   # vol% — phases below this are considered absent

# Define which phases to track (skip pure end-members and fluid)
tracked = ["Chl(W)", "law", "zo", "cz", "Gt(HGP)", "Cpx(HGP)", "Opx(HGP)",
           "O(HGP)", "Fsp(HGP)", "phA", "liz", "br", "sph", "ru", "q", "ta"]

# Build assemblage string for each P-T cell
assemblage = fill("", n_P, n_T)
for i in 1:n_P, j in 1:n_T
    phases_present = []
    for phase in tracked
        if haskey(phase_grids, phase)
            v = phase_grids[phase][i, j]
            if !isnan(v) && v > threshold
                push!(phases_present, phase)
            end
        end
    end
    assemblage[i, j] = join(sort(phases_present), "+")
end

# Assign integer code to each unique assemblage
unique_assemblages = unique(filter(!isempty, vec(assemblage)))
println("Unique assemblages ($(length(unique_assemblages))):")
for (k, a) in enumerate(unique_assemblages)
    println("  $k: $a")
end

assem_code    = Dict(a => i for (i, a) in enumerate(unique_assemblages))
assem_code[""] = 0
assem_float   = [assem_code[assemblage[i,j]] / max(length(unique_assemblages), 1)
                 for i in 1:n_P, j in 1:n_T]

p2 = heatmap(
    T_K .- 273.15, P_GPa, assem_float,
    c        = :tab20,
    xlabel   = "Temperature (°C)",
    ylabel   = "Pressure (GPa)",
    title    = "Phase assemblage (sample 41)",
    colorbar = false,
    clims    = (0.0, 1.0),
)

# Annotate each assemblage field at its centroid
for assem in unique_assemblages
    mask  = assemblage .== assem
    idxs  = findall(mask)
    isempty(idxs) && continue
    mean_i = mean(getindex.(idxs, 1))
    mean_j = mean(getindex.(idxs, 2))
    ti = T_K[round(Int, clamp(mean_j, 1, n_T))] - 273.15
    pi = P_GPa[round(Int, clamp(mean_i, 1, n_P))]
    # Wrap long label to two lines
    label  = replace(assem, "+" => "+\n")
    annotate!(p2, ti, pi, text(label, 5, :white, :center))
end

# ── Plot 3 (bottom-left): Bound H2O vol% ─────────────────────────────────────
p3 = heatmap(
    T_K .- 273.15, P_GPa, bound_h2o_grid,
    c              = :blues,
    xlabel         = "Temperature (°C)",
    ylabel         = "Pressure (GPa)",
    title          = "Bound H₂O vol% (sample 41)",
    colorbar_title = "vol%",
)

# ── Plot 4 (bottom-right): H2O uncertainty 1σ ────────────────────────────────
p4 = heatmap(
    t_ens .- 273.15, p_ens, h2o_std_ens,
    c              = :heat,
    xlabel         = "Temperature (°C)",
    ylabel         = "Pressure (GPa)",
    title          = "H₂O uncertainty 1σ (200-sample ensemble)",
    colorbar_title = "σ wt%",
)

# ── Combine: top-left=hydrous, top-right=phase diagram,
#             bottom-left=bound H2O, bottom-right=uncertainty ────────────────
fig = plot(p1, p2, p3, p4,
    layout = (2, 2),
    size   = (1400, 1000),
    margin = 5Plots.mm,
)

outpath = "/Users/hchoi342/Documents/Archean-OC/pseudosection/pseudosection_uncertainty.png"
savefig(fig, outpath)
println("Saved: ", outpath)
