# using HDF5
# using DelimitedFiles
# using SparseArrays
# using Statistics

# # ========== Helper Functions ==========

# function ang2cm(lam)
#     return 1e8 / lam
# end

# function cm2ang(lam)
#     return 1e8 / lam
# end

# fwhm2sigma = 1 / (2 * sqrt(2 * log(2)))

# function lsf_sigma(λ, R)
#     return (λ / R) * fwhm2sigma
# end

# function lsf_sigma_and_bounds(λ, R; σ_window=10)
#     σ = lsf_sigma(λ, R)
#     return σ, (λ - σ_window * σ, λ + σ_window * σ)
# end

# function instrumental_lsf_kernel(λ, λc, R)
#     σ, (lower, upper) = lsf_sigma_and_bounds(λc, R)
#     msk = (lower .<= λ .<= upper)
#     ϕ = exp.(-((λ[msk] .- λc).^2) / (2σ^2))
#     ϕ ./= dropdims(sum(ϕ, dims=1), dims=1)
#     return msk, ϕ
# end

# function instrument_lsf_sparse_matrix(λ_input, λ_output, R)
#     row, col, val = Int[], Int[], Float64[]
#     for (indx, λ) in enumerate(λ_output)
#         msk, ϕ = instrumental_lsf_kernel(λ_input, λ, R)
#         push!(row, fill(indx, count(msk))...)
#         push!(col, findall(msk)...)
#         push!(val, ϕ...)
#     end
#     return sparse(row, col, val)
# end

# # ========== Parse Filenames for Asymmetric Top ==========

# function parse_pgo_filename(filename)
#     # Example filename: spec_T30.0_AB0.001200_C0.001200_FAB1.0010_FC1.0010.txt
#     parts = split(filename, "_")
#     temp_str = replace(parts[2], "T" => "")
#     ab_str = replace(parts[3], "AB" => "")
#     c_str = replace(parts[4], "C" => "")
#     fraction_ab_str = replace(parts[5], "FAB" => "")
#     fraction_c_str = replace(replace(parts[6], "FC" => ""), ".txt" => "")
#     return (
#         parse(Float64, temp_str),
#         parse(Float64, ab_str),
#         parse(Float64, c_str),
#         parse(Float64, fraction_ab_str),
#         parse(Float64, fraction_c_str)
#     )
# end

# # ========== Find Spectrum Start ==========

# function find_spectra_start(filepath)
#     open(filepath) do file
#         for (i, line) in enumerate(eachline(file))
#             if occursin("Molecule Manifol", line)
#                 return i
#             end
#         end
#     end
# end

# # ========== Main Processing Function ==========

# function process_pgo_outputs(output_dir, savename)
#     files = readdir(output_dir)
#     spec_files = filter(f -> startswith(f, "spec_") && endswith(f, ".txt"), files)

#     temps = Float64[]
#     abvals = Float64[]
#     cvals = Float64[]
#     fractions_ab = Float64[]
#     fractions_c = Float64[]

#     for file in spec_files
#         temp, abval, cval, fraction_ab, fraction_c = parse_pgo_filename(file)
#         push!(temps, temp)
#         push!(abvals, abval)
#         push!(cvals, cval)
#         push!(fractions_ab, fraction_ab)
#         push!(fractions_c, fraction_c)
#     end

#     temps = sort(unique(temps))
#     abvals = sort(unique(abvals))
#     cvals = sort(unique(cvals))
#     fractions_ab = sort(unique(fractions_ab))
#     fractions_c = sort(unique(fractions_c))

#     xwave = [15257.02999338, 15257.24077849, 15257.45156652, 15257.66235746,
#              15257.87315131, 15258.08394807, 15258.29474775, 15258.50555034,
#              15258.71635584, 15258.92716425, 15259.13797558, 15259.34878982,
#              15259.55960697, 15259.77042703, 15259.98125001, 15260.19207589,
#              15260.4029047, 15260.61373641, 15260.82457104, 15261.03540858,
#              15261.24624903, 15261.45709239, 15261.66793867, 15261.87878786,
#              15262.08963997, 15262.30049498, 15262.51135291, 15262.72221376,
#              15262.93307751, 15263.14394418, 15263.35481377, 15263.56568626,
#              15263.77656167, 15263.98743999, 15264.19832123, 15264.40920538,
#              15264.62009244, 15264.83098242, 15265.04187531, 15265.25277112,
#              15265.46366983, 15265.67457146, 15265.88547601, 15266.09638347,
#              15266.30729384, 15266.51820713, 15266.72912333, 15266.94004244,
#              15267.15096447, 15267.36188941, 15267.57281727, 15267.78374804,
#              15267.99468172, 15268.20561832, 15268.41655784, 15268.62750026,
#              15268.83844561, 15269.04939386, 15269.26034503, 15269.47129912,
#              15269.68225612, 15269.89321603, 15270.10417886, 15270.3151446,
#              15270.52611326, 15270.73708483, 15270.94805932, 15271.15903672,
#              15271.37001703, 15271.58100027, 15271.79198641, 15272.00297547,
#              15272.21396745, 15272.42496234, 15272.63596014, 15272.84696087,
#              15273.0579645, 15273.26897105, 15273.47998052, 15273.6909929,
#              15273.9020082, 15274.11302641, 15274.32404754, 15274.53507158,
#              15274.74609854, 15274.95712841, 15275.1681612, 15275.37919691,
#              15275.59023553, 15275.80127706, 15276.01232151, 15276.22336888,
#              15276.43441916, 15276.64547236, 15276.85652848, 15277.06758751,
#              15277.27864945, 15277.48971432, 15277.7007821, 15277.91185279,
#              15278.1229264, 15278.33400293, 15278.54508237, 15278.75616473,
#              15278.96725, 15279.1783382, 15279.3894293, 15279.60052333,
#              15279.81162027, 15280.02272013, 15280.2338229, 15280.44492859,
#              15280.6560372, 15280.86714872, 15281.07826316, 15281.28938051,
#              15281.50050079, 15281.71162398, 15281.92275008, 15282.13387911,
#              15282.34501105, 15282.55614591, 15282.76728368, 15282.97842437,
#              15283.18956798, 15283.4007145, 15283.61186395, 15283.82301631,
#              15284.03417158, 15284.24532978, 15284.45649089, 15284.66765492,
#              15284.87882186, 15285.08999173, 15285.30116451, 15285.51234021,
#              15285.72351882, 15285.93470036, 15286.14588481, 15286.35707218,
#              15286.56826246, 15286.77945567, 15286.99065179]

#     spectra = zeros(length(temps), length(abvals), length(cvals), length(fractions_ab), length(fractions_c), length(xwave))

#     for file in spec_files
#         temp, abval, cval, fraction_ab, fraction_c = parse_pgo_filename(file)
#         t_idx = findfirst(isequal(temp), temps)
#         ab_idx = findfirst(isequal(abval), abvals)
#         c_idx = findfirst(isequal(cval), cvals)
#         fab_idx = findfirst(isequal(fraction_ab), fractions_ab)
#         fc_idx = findfirst(isequal(fraction_c), fractions_c)

#         filepath = joinpath(output_dir, file)
#         start_line = find_spectra_start(filepath)
#         data = readdlm(filepath, skipstart=start_line)
#         wavenum = data[1:end-1, 10]
#         waveang = cm2ang.(wavenum)
#         intofline = data[1:end-1, 11]

#         flux = zeros(length(xwave))
#         for i = 1:length(waveang)
#             msk, ϕ = instrumental_lsf_kernel(xwave, waveang[i], 22_500)
#             flux[msk] .+= intofline[i] * ϕ
#         end

#         spectra[t_idx, ab_idx, c_idx, fab_idx, fc_idx, :] = flux
#     end

#     h5open(savename * ".h5", "w") do file
#         write(file, "spectra", spectra)
#         write(file, "temperatures", temps)
#         write(file, "ab_values", abvals)
#         write(file, "c_values", cvals)
#         write(file, "fraction_ab", fractions_ab)
#         write(file, "fraction_c", fractions_c)
#         write(file, "wavelengths", xwave)
#     end
# end

# if length(ARGS) != 2
#     error("Usage: julia ingest_pgo_asym.jl <output_dir> <savename>")
# end

# process_pgo_outputs(ARGS[1], ARGS[2])

# using Distributed
# using ProgressMeter
# using HDF5
# using DelimitedFiles
# using SparseArrays
# using Statistics
# # Detect allocated CPUs from environment variables commonly set by schedulers
# function get_allocated_cpus()
#     for var in ["SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "PBS_NP", "NSLOTS"]
#         if haskey(ENV, var)
#             cpus = tryparse(Int, ENV[var])
#             if cpus !== nothing && cpus > 0
#                 return cpus
#             end
#         end
#     end
#     # Fallback if no env var found: use 1 less than system threads or 1
#     return max(Sys.CPU_THREADS-1, 1)
# end

# allocated_cpus = get_allocated_cpus()

# if nworkers() == 1
#     addprocs(allocated_cpus)  # minus 1 because main process counts as one
# end
# # ========== Helper Functions ==========

# @everywhere begin
#     using Distributed
#     using ProgressMeter
#     using HDF5
#     using DelimitedFiles
#     using SparseArrays
#     using Statistics

#     function ang2cm(lam)
#         return 1e8 / lam
#     end

#     function cm2ang(lam)
#         return 1e8 / lam
#     end

#     fwhm2sigma = 1 / (2 * sqrt(2 * log(2)))

#     function lsf_sigma(λ, R)
#         return (λ / R) * fwhm2sigma
#     end

#     function lsf_sigma_and_bounds(λ, R; σ_window=10)
#         σ = lsf_sigma(λ, R)
#         return σ, (λ - σ_window * σ, λ + σ_window * σ)
#     end

#     function instrumental_lsf_kernel(λ, λc, R)
#         σ, (lower, upper) = lsf_sigma_and_bounds(λc, R)
#         msk = (lower .<= λ .<= upper)
#         ϕ = exp.(-((λ[msk] .- λc).^2) / (2σ^2))
#         ϕ ./= dropdims(sum(ϕ, dims=1), dims=1)
#         return msk, ϕ
#     end

#     function parse_pgo_filename(filename)
#         parts = split(filename, "_")
#         temp_str = replace(parts[2], "T" => "")
#         ab_str = replace(parts[3], "AB" => "")
#         c_str = replace(parts[4], "C" => "")
#         fraction_ab_str = replace(parts[5], "FAB" => "")
#         fraction_c_str = replace(replace(parts[6], "FC" => ""), ".txt" => "")
#         return (
#             parse(Float64, temp_str),
#             parse(Float64, ab_str),
#             parse(Float64, c_str),
#             parse(Float64, fraction_ab_str),
#             parse(Float64, fraction_c_str)
#         )
#     end

#     function find_spectra_start(filepath)
#         open(filepath) do file
#             for (i, line) in enumerate(eachline(file))
#                 if occursin("Molecule Manifol", line)
#                     return i
#                 end
#             end
#         end
#     end

#     function process_single_file(file, output_dir, xwave, temps, abvals, cvals, fractions_ab, fractions_c)
#         temp, abval, cval, fraction_ab, fraction_c = parse_pgo_filename(file)

#         t_idx = findfirst(isequal(temp), temps)
#         ab_idx = findfirst(isequal(abval), abvals)
#         c_idx = findfirst(isequal(cval), cvals)
#         fab_idx = findfirst(isequal(fraction_ab), fractions_ab)
#         fc_idx = findfirst(isequal(fraction_c), fractions_c)

#         filepath = joinpath(output_dir, file)
#         start_line = find_spectra_start(filepath)
#         data = readdlm(filepath, skipstart=start_line)
#         wavenum = data[1:end-1, 10]
#         waveang = cm2ang.(wavenum)
#         intofline = data[1:end-1, 11]

#         flux = zeros(length(xwave))
#         for i = 1:length(waveang)
#             msk, ϕ = instrumental_lsf_kernel(xwave, waveang[i], 22_500)
#             flux[msk] .+= intofline[i] * ϕ
#         end

#         return (t_idx, ab_idx, c_idx, fab_idx, fc_idx, flux)
#     end
# end

# # ========== Main Processing Function ==========

# function process_pgo_outputs(output_dir, savename)
#     files = readdir(output_dir)
#     spec_files = filter(f -> startswith(f, "spec_") && endswith(f, ".txt"), files)

#     temps = Float64[]
#     abvals = Float64[]
#     cvals = Float64[]
#     fractions_ab = Float64[]
#     fractions_c = Float64[]

#     for file in spec_files
#         temp, abval, cval, fraction_ab, fraction_c = parse_pgo_filename(file)
#         push!(temps, temp)
#         push!(abvals, abval)
#         push!(cvals, cval)
#         push!(fractions_ab, fraction_ab)
#         push!(fractions_c, fraction_c)
#     end

#     temps = sort(unique(temps))
#     abvals = sort(unique(abvals))
#     cvals = sort(unique(cvals))
#     fractions_ab = sort(unique(fractions_ab))
#     fractions_c = sort(unique(fractions_c))

#     # Provided wavelength grid
#     xwave = [15257.02999338, 15257.24077849, 15257.45156652, 15257.66235746, 15257.87315131, 15258.08394807,
#              15258.29474775, 15258.50555034, 15258.71635584, 15258.92716425, 15259.13797558, 15259.34878982,
#              15259.55960697, 15259.77042703, 15259.98125001, 15260.19207589, 15260.40290470, 15260.61373641,
#              15260.82457104, 15261.03540858, 15261.24624903, 15261.45709239, 15261.66793867, 15261.87878786,
#              15262.08963997, 15262.30049498, 15262.51135291, 15262.72221376, 15262.93307751, 15263.14394418,
#              15263.35481377, 15263.56568626, 15263.77656167, 15263.98743999, 15264.19832123, 15264.40920538,
#              15264.62009244, 15264.83098242, 15265.04187531, 15265.25277112, 15265.46366983, 15265.67457146,
#              15265.88547601, 15266.09638347, 15266.30729384, 15266.51820713, 15266.72912333, 15266.94004244,
#              15267.15096447, 15267.36188941, 15267.57281727, 15267.78374804, 15267.99468172, 15268.20561832,
#              15268.41655784, 15268.62750026, 15268.83844561, 15269.04939386, 15269.26034503, 15269.47129912,
#              15269.68225612, 15269.89321603, 15270.10417886, 15270.31514460, 15270.52611326, 15270.73708483,
#              15270.94805932, 15271.15903672, 15271.37001703, 15271.58100027, 15271.79198641, 15272.00297547,
#              15272.21396745, 15272.42496234, 15272.63596014, 15272.84696087, 15273.05796450, 15273.26897105,
#              15273.47998052, 15273.69099290, 15273.90200820, 15274.11302641, 15274.32404754, 15274.53507158,
#              15274.74609854, 15274.95712841, 15275.16816120, 15275.37919691, 15275.59023553, 15275.80127706,
#              15276.01232151, 15276.22336888, 15276.43441916, 15276.64547236, 15276.85652848, 15277.06758751,
#              15277.27864945, 15277.48971432, 15277.70078210, 15277.91185279, 15278.12292640, 15278.33400293,
#              15278.54508237, 15278.75616473, 15278.96725000, 15279.17833820, 15279.38942930, 15279.60052333,
#              15279.81162027, 15280.02272013, 15280.23382290, 15280.44492859, 15280.65603720, 15280.86714872,
#              15281.07826316, 15281.28938051, 15281.50050079, 15281.71162398, 15281.92275008, 15282.13387911,
#              15282.34501105, 15282.55614591, 15282.76728368, 15282.97842437, 15283.18956798, 15283.40071450,
#              15283.61186395, 15283.82301631, 15284.03417158, 15284.24532978, 15284.45649089, 15284.66765492,
#              15284.87882186, 15285.08999173, 15285.30116451, 15285.51234021, 15285.72351882, 15285.93470036,
#              15286.14588481, 15286.35707218, 15286.56826246, 15286.77945567, 15286.99065179]

#     spectra = zeros(length(temps), length(abvals), length(cvals), length(fractions_ab), length(fractions_c), length(xwave))

#     println("Processing $(length(spec_files)) files using $(nworkers()) workers...")
#     results = @showprogress "Processing files in parallel" pmap(file -> process_single_file(file, output_dir, xwave, temps, abvals, cvals, fractions_ab, fractions_c), spec_files)

#     for (t_idx, ab_idx, c_idx, fab_idx, fc_idx, flux) in results
#         spectra[t_idx, ab_idx, c_idx, fab_idx, fc_idx, :] = flux
#     end

#     h5open(savename * ".h5", "w") do file
#         write(file, "spectra", spectra)
#         write(file, "temperatures", temps)
#         write(file, "ab_values", abvals)
#         write(file, "c_values", cvals)
#         write(file, "fraction_ab", fractions_ab)
#         write(file, "fraction_c", fractions_c)
#         write(file, "wavelengths", xwave)
#     end
# end

# if length(ARGS) != 2
#     error("Usage: julia ingest_pgo_asym.jl <output_dir> <savename>")
# end

# process_pgo_outputs(ARGS[1], ARGS[2])

using Distributed
using ProgressMeter
using HDF5
using DelimitedFiles
using SparseArrays
using Statistics

# ========== Allocate CPUs ==========
function get_allocated_cpus()
    for var in ["SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "PBS_NP", "NSLOTS"]
        if haskey(ENV, var)
            cpus = tryparse(Int, ENV[var])
            if cpus !== nothing && cpus > 0
                return cpus
            end
        end
    end
    return max(Sys.CPU_THREADS - 1, 1)
end

if nworkers() == 1
    addprocs(get_allocated_cpus())
end

# ========== Share Functions Across All Processes ==========
@everywhere using DelimitedFiles, SparseArrays, Statistics, ProgressMeter

@everywhere begin
    function ang2cm(lam)
        return 1e8 / lam
    end

    function cm2ang(lam)
        return 1e8 / lam
    end

    fwhm2sigma = 1 / (2 * sqrt(2 * log(2)))

    function lsf_sigma(λ, R)
        return (λ / R) * fwhm2sigma
    end

    function lsf_sigma_and_bounds(λ, R; σ_window=10)
        σ = lsf_sigma(λ, R)
        return σ, (λ - σ_window * σ, λ + σ_window * σ)
    end

    function instrumental_lsf_kernel(λ, λc, R)
        σ, (lower, upper) = lsf_sigma_and_bounds(λc, R)
        msk = (lower .<= λ .<= upper)
        ϕ = exp.(-((λ[msk] .- λc).^2) / (2σ^2))
        ϕ ./= dropdims(sum(ϕ, dims=1), dims=1)
        return msk, ϕ
    end

    function parse_pgo_filename(filename)
        parts = split(filename, "_")
        temp_str = replace(parts[2], "T" => "")
        ab_str = replace(parts[3], "AB" => "")
        c_str = replace(parts[4], "C" => "")
        fraction_ab_str = replace(parts[5], "FAB" => "")
        fraction_c_str = replace(replace(parts[6], "FC" => ""), ".txt" => "")
        return (
            parse(Float64, temp_str),
            parse(Float64, ab_str),
            parse(Float64, c_str),
            parse(Float64, fraction_ab_str),
            parse(Float64, fraction_c_str)
        )
    end

    function find_spectra_start(filepath)
        open(filepath) do file
            for (i, line) in enumerate(eachline(file))
                if occursin("Molecule Manifol", line)
                    return i
                end
            end
        end
    end
end

# ========== File Processing Task ==========
@everywhere function process_single_file(file, output_dir, xwave, temps, abvals, cvals, fractions_ab, fractions_c)
    temp, abval, cval, fraction_ab, fraction_c = parse_pgo_filename(file)

    t_idx = findfirst(isequal(temp), temps)
    ab_idx = findfirst(isequal(abval), abvals)
    c_idx = findfirst(isequal(cval), cvals)
    fab_idx = findfirst(isequal(fraction_ab), fractions_ab)
    fc_idx = findfirst(isequal(fraction_c), fractions_c)

    filepath = joinpath(output_dir, file)
    start_line = find_spectra_start(filepath)
    data = readdlm(filepath, skipstart=start_line)
    wavenum = data[1:end-1, 10]
    waveang = cm2ang.(wavenum)
    intofline = data[1:end-1, 11]

    flux = zeros(length(xwave))
    for i = 1:length(waveang)
        msk, ϕ = instrumental_lsf_kernel(xwave, waveang[i], 22_500)
        flux[msk] .+= intofline[i] * ϕ
    end

    return (t_idx, ab_idx, c_idx, fab_idx, fc_idx, flux)
end

# ========== Parallel Processing ==========
function process_pgo_outputs(output_dir, savename)
    files = readdir(output_dir)
    spec_files = filter(f -> startswith(f, "spec_") && endswith(f, ".txt"), files)

    temps = Float64[]
    abvals = Float64[]
    cvals = Float64[]
    fractions_ab = Float64[]
    fractions_c = Float64[]

    for file in spec_files
        temp, abval, cval, fraction_ab, fraction_c = parse_pgo_filename(file)
        push!(temps, temp)
        push!(abvals, abval)
        push!(cvals, cval)
        push!(fractions_ab, fraction_ab)
        push!(fractions_c, fraction_c)
    end

    temps = sort(unique(temps))
    abvals = sort(unique(abvals))
    cvals = sort(unique(cvals))
    fractions_ab = sort(unique(fractions_ab))
    fractions_c = sort(unique(fractions_c))

    xwave = [15257.02999338, 15257.24077849, 15257.45156652, 15257.66235746, 15257.87315131, 15258.08394807,
            15258.29474775, 15258.50555034, 15258.71635584, 15258.92716425, 15259.13797558, 15259.34878982,
            15259.55960697, 15259.77042703, 15259.98125001, 15260.19207589, 15260.40290470, 15260.61373641,
            15260.82457104, 15261.03540858, 15261.24624903, 15261.45709239, 15261.66793867, 15261.87878786,
            15262.08963997, 15262.30049498, 15262.51135291, 15262.72221376, 15262.93307751, 15263.14394418,
            15263.35481377, 15263.56568626, 15263.77656167, 15263.98743999, 15264.19832123, 15264.40920538,
            15264.62009244, 15264.83098242, 15265.04187531, 15265.25277112, 15265.46366983, 15265.67457146,
            15265.88547601, 15266.09638347, 15266.30729384, 15266.51820713, 15266.72912333, 15266.94004244,
            15267.15096447, 15267.36188941, 15267.57281727, 15267.78374804, 15267.99468172, 15268.20561832,
            15268.41655784, 15268.62750026, 15268.83844561, 15269.04939386, 15269.26034503, 15269.47129912,
            15269.68225612, 15269.89321603, 15270.10417886, 15270.31514460, 15270.52611326, 15270.73708483,
            15270.94805932, 15271.15903672, 15271.37001703, 15271.58100027, 15271.79198641, 15272.00297547,
            15272.21396745, 15272.42496234, 15272.63596014, 15272.84696087, 15273.05796450, 15273.26897105,
            15273.47998052, 15273.69099290, 15273.90200820, 15274.11302641, 15274.32404754, 15274.53507158,
            15274.74609854, 15274.95712841, 15275.16816120, 15275.37919691, 15275.59023553, 15275.80127706,
            15276.01232151, 15276.22336888, 15276.43441916, 15276.64547236, 15276.85652848, 15277.06758751,
            15277.27864945, 15277.48971432, 15277.70078210, 15277.91185279, 15278.12292640, 15278.33400293,
            15278.54508237, 15278.75616473, 15278.96725000, 15279.17833820, 15279.38942930, 15279.60052333,
            15279.81162027, 15280.02272013, 15280.23382290, 15280.44492859, 15280.65603720, 15280.86714872,
            15281.07826316, 15281.28938051, 15281.50050079, 15281.71162398, 15281.92275008, 15282.13387911,
            15282.34501105, 15282.55614591, 15282.76728368, 15282.97842437, 15283.18956798, 15283.40071450,
            15283.61186395, 15283.82301631, 15284.03417158, 15284.24532978, 15284.45649089, 15284.66765492,
            15284.87882186, 15285.08999173, 15285.30116451, 15285.51234021, 15285.72351882, 15285.93470036,
            15286.14588481, 15286.35707218, 15286.56826246, 15286.77945567, 15286.99065179]



    spectra = zeros(length(temps), length(abvals), length(cvals), length(fractions_ab), length(fractions_c), length(xwave))

    println("Processing $(length(spec_files)) files using $(nworkers()) workers...")

    results = @showprogress "Processing files in parallel" pmap(
        file -> process_single_file(file, output_dir, xwave, temps, abvals, cvals, fractions_ab, fractions_c),
        spec_files
    )

    for (t_idx, ab_idx, c_idx, fab_idx, fc_idx, flux) in results
        spectra[t_idx, ab_idx, c_idx, fab_idx, fc_idx, :] = flux
    end
    print(savename)
    h5open(savename * ".h5", "w") do file
        write(file, "spectra", spectra)
        write(file, "temperatures", temps)
        write(file, "ab_values", abvals)
        write(file, "c_values", cvals)
        write(file, "fraction_ab", fractions_ab)
        write(file, "fraction_c", fractions_c)
        write(file, "wavelengths", xwave)
    end
end

# ========== Script Entry ==========
if length(ARGS) != 2
    error("Usage: julia ingest_pgo_asym.jl <output_dir> <savename>")
end

process_pgo_outputs(ARGS[1], ARGS[2])