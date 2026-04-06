include("xml_editor.jl")

file = "asym_top_15272_C2v.pgo"

println("=== Load the XML file ===")
editor = load_xml(file)

println("\n=== Check current values ===")
A_excited = get_parameter(editor, "Species.AsymmetricTop.Excited.v=1.A")
temp = get_parameter(editor, "Temperature")
println("Excited A value: $A_excited")
println("Temperature: $temp")

println("\n=== Make multiple edits (not saved yet) ===")
set_parameter!(editor, "Species.AsymmetricTop.Excited.v=1.A", "0.08")
set_parameter!(editor, "Species.AsymmetricTop.Excited.v=1.B", "0.012")
set_parameter!(editor, "Temperature", "25")
set_parameter!(editor, "Gaussian", "0.003")

println("\n=== Verify changes in memory ===")
println("New Excited A: ", get_parameter(editor, "Species.AsymmetricTop.Excited.v=1.A"))
println("New Temperature: ", get_parameter(editor, "Temperature"))

println("\n=== Save all changes at once ===")
save_xml(editor, save_path="modified_asym_top.pgo")

println("\n=== Done! ===")
println("Original file unchanged: $file")
println("Modified file saved as: modified_asym_top.pgo")

