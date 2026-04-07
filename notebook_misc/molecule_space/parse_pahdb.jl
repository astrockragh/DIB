using DelimitedFiles

"""
Structure to hold geometry information for a single atom
"""
struct GeometryEntry
    position::Int
    x::Float64
    y::Float64
    z::Float64
    type::Int
end

"""
Structure to hold a PAH database entry with UID and geometry
"""
struct PAHEntry
    uid::Int
    geometry::Vector{GeometryEntry}
end

"""
Parse the PAH database ASCII file and extract UID and geometry information
"""
function parse_pahdb(filename::String)
    entries = PAHEntry[]
    
    open(filename, "r") do file
        current_uid = nothing
        current_geometry = GeometryEntry[]
        in_geometry_section = false
        
        for line in eachline(file)
            # Check for UID
            if startswith(line, "UID:")
                # If we already have a UID with geometry, save it
                if current_uid !== nothing && !isempty(current_geometry)
                    push!(entries, PAHEntry(current_uid, copy(current_geometry)))
                end
                
                # Start new entry
                current_uid = parse(Int, strip(split(line, ":")[2]))
                current_geometry = GeometryEntry[]
                in_geometry_section = false
                
            # Check if we're entering the geometry section
            elseif startswith(line, "# GEOMETRY:")
                in_geometry_section = true
                
            # Check if we're leaving the geometry section
            elseif startswith(line, "# TRANSITIONS:")
                in_geometry_section = false
                
            # Parse geometry data
            elseif in_geometry_section && !startswith(line, "#")
                # Skip empty lines
                stripped = strip(line)
                if isempty(stripped)
                    continue
                end
                
                # Parse the geometry line
                parts = split(stripped)
                if length(parts) == 5
                    try
                        position = parse(Int, parts[1])
                        x = parse(Float64, parts[2])
                        y = parse(Float64, parts[3])
                        z = parse(Float64, parts[4])
                        atom_type = parse(Int, parts[5])
                        
                        push!(current_geometry, GeometryEntry(position, x, y, z, atom_type))
                    catch e
                        # Skip lines that can't be parsed
                        continue
                    end
                end
            end
        end
        
        # Don't forget the last entry
        if current_uid !== nothing && !isempty(current_geometry)
            push!(entries, PAHEntry(current_uid, copy(current_geometry)))
        end
    end
    
    return entries
end

"""
Get atomic mass from atomic number
"""
function get_atomic_mass(atomic_number::Int; rounding::Bool=false)
    # Atomic masses in atomic mass units (u)
    atomic_masses = Dict(
        1 => 1.008,      # Hydrogen
        6 => 12.011,     # Carbon
        7 => 14.007,     # Nitrogen
        8 => 15.999,     # Oxygen
        9 => 18.998,     # Fluorine
        15 => 30.974,    # Phosphorus
        16 => 32.06,     # Sulfur
        17 => 35.45,     # Chlorine
    )
    if rounding
        return round(Int,get(atomic_masses, atomic_number, 0.0))
    else
        return get(atomic_masses, atomic_number, 0.0)
    end
end

"""
Extract x, y, z, and mass vectors from a PAH entry

Returns a named tuple with fields:
- x: Vector of x coordinates
- y: Vector of y coordinates
- z: Vector of z coordinates
- mass: Vector of atomic masses
"""
function get_xyzm_vectors(entry::PAHEntry; rounding::Bool=false)
    n_atoms = length(entry.geometry)
    
    x = Vector{Float64}(undef, n_atoms)
    y = Vector{Float64}(undef, n_atoms)
    z = Vector{Float64}(undef, n_atoms)
    mass = Vector{Float64}(undef, n_atoms)
    
    for i in 1:n_atoms
        geo = entry.geometry[i]
        x[i] = geo.x
        y[i] = geo.y
        z[i] = geo.z
        mass[i] = get_atomic_mass(geo.type, rounding=rounding)
    end
    
    return (x=x, y=y, z=z, mass=mass)
end

"""
Print a summary of a PAH entry
"""
function print_entry_summary(entry::PAHEntry)
    println("UID: $(entry.uid)")
    println("Number of atoms: $(length(entry.geometry))")
    println("First 3 geometry entries:")
    for i in 1:min(3, length(entry.geometry))
        geo = entry.geometry[i]
        println("  Position $(geo.position): x=$(geo.x), y=$(geo.y), z=$(geo.z), type=$(geo.type)")
    end
    println()
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    # Parse the database
    println("Parsing PAH database...")
    filename = "/uufs/chpc.utah.edu/common/home/u6039752/scratch1/working/2025_10_13/pahdb-complete-theoretical-v4.00-alpha_kacca9ms0lgo93HfUcn.ascii"
    
    entries = parse_pahdb(filename)
    
    println("Successfully parsed $(length(entries)) entries")
    println("\nShowing first 3 entries:\n")
    
    for i in 1:min(3, length(entries))
        print_entry_summary(entries[i])
    end
    
    println("Data structure info:")
    println("- Each entry is of type PAHEntry with fields: uid, geometry")
    println("- geometry is a Vector{GeometryEntry}")
    println("- Each GeometryEntry has fields: position, x, y, z, type")
    println("\nExample usage:")
    println("  entries[1].uid                    # Access UID of first entry")
    println("  entries[1].geometry              # Access geometry vector")
    println("  entries[1].geometry[1].x         # Access x coordinate of first atom")
    
    println("\n" * "="^60)
    println("Testing get_xyzm_vectors function:")
    println("="^60)
    
    # Test the new function
    xyzm = get_xyzm_vectors(entries[1])
    println("\nFor entry UID=$(entries[1].uid):")
    println("Number of atoms: $(length(xyzm.x))")
    println("\nFirst 5 atoms:")
    for i in 1:min(5, length(xyzm.x))
        println("  Atom $i: x=$(xyzm.x[i]), y=$(xyzm.y[i]), z=$(xyzm.z[i]), mass=$(xyzm.mass[i]) u")
    end
    
    println("\nUsage:")
    println("  xyzm = get_xyzm_vectors(entries[1])")
    println("  xyzm.x                           # Vector of x coordinates")
    println("  xyzm.y                           # Vector of y coordinates")
    println("  xyzm.z                           # Vector of z coordinates")
    println("  xyzm.mass                        # Vector of atomic masses")
end

