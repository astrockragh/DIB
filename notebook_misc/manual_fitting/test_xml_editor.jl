include("xml_editor.jl")

file = "asym_top_15272_C2v.pgo"

println("=== All parameters in the file ===")
list_parameters(file)

println("\n=== Get a specific parameter ===")
value = get_parameter(file, "Species.AsymmetricTop.Excited.v=1.A")
println("Species.AsymmetricTop.Excited.v=1.A = $value")

println("\n=== Edit the parameter ===")
edit_parameter(file, "Species.AsymmetricTop.Excited.v=1.A", "0.08", save_path="modified_asym_top.pgo")

println("\n=== Verify the change ===")
new_value = get_parameter("modified_asym_top.pgo", "Species.AsymmetricTop.Excited.v=1.A")
println("New value: $new_value")

