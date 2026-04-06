using EzXML

"""
    XMLDocument wrapper type for editing multiple parameters before saving
"""
mutable struct XMLEditor
    doc::EzXML.Document
    file_path::String
end

"""
    load_xml(file_path)

Load an XML file for editing. Returns an XMLEditor object.

# Example
```julia
editor = load_xml("asym_top_15272_C2v.pgo")
set_parameter!(editor, "Species.AsymmetricTop.Excited.v=1.A", "0.08")
set_parameter!(editor, "Temperature", "25")
save_xml(editor, "modified.pgo")
```
"""
function load_xml(file_path)
    doc = readxml(file_path)
    return XMLEditor(doc, file_path)
end

"""
    set_parameter!(editor, path_string, new_value)

Set a parameter value in the loaded XML document without saving.

# Example
```julia
editor = load_xml("file.pgo")
set_parameter!(editor, "Species.AsymmetricTop.Excited.v=1.A", "0.08")
set_parameter!(editor, "Temperature", "25")
save_xml(editor)
```
"""
function set_parameter!(editor::XMLEditor, path_string, new_value; verbose=false)
    root_node = root(editor.doc)
    path_parts = split(path_string, ".")
    current = root_node

    for (i, part) in enumerate(path_parts)
        is_last = (i == length(path_parts))

        if is_last
            # Find and update the parameter
            found = false
            for child in eachelement(current)
                if nodename(child) == "Parameter"
                    name_attr = child["Name"]
                    if name_attr == part
                        child["Value"] = string(new_value)
                        if verbose
                            println("Updated $path_string: $(child["Value"])")
                        end
                        found = true
                        break
                    end
                end
            end

            if !found
                error("Parameter '$part' not found at path '$path_string'")
            end
        else
            # Navigate to next level
            found = false
            for child in eachelement(current)
                child_name = nodename(child)
                name_attr = try
                    child["Name"]
                catch KeyError
                    nothing
                end

                if !isnothing(name_attr) && name_attr == part
                    current = child
                    found = true
                    break
                elseif child_name == part
                    current = child
                    found = true
                    break
                end
            end

            if !found
                error("Element '$part' not found in path '$path_string'")
            end
        end
    end

    return editor
end

"""
    get_parameter(editor, path_string)

Get a parameter value from the loaded XML document.

# Example
```julia
editor = load_xml("file.pgo")
value = get_parameter(editor, "Temperature")
```
"""
function get_parameter(editor::XMLEditor, path_string)
    root_node = root(editor.doc)
    path_parts = split(path_string, ".")
    current = root_node

    for (i, part) in enumerate(path_parts)
        is_last = (i == length(path_parts))

        if is_last
            for child in eachelement(current)
                if nodename(child) == "Parameter"
                    name_attr = child["Name"]
                    if name_attr == part
                        return child["Value"]
                    end
                end
            end
            error("Parameter '$part' not found at path '$path_string'")
        else
            found = false
            for child in eachelement(current)
                child_name = nodename(child)
                name_attr = try
                    child["Name"]
                catch KeyError
                    nothing
                end

                if !isnothing(name_attr) && name_attr == part
                    current = child
                    found = true
                    break
                elseif child_name == part
                    current = child
                    found = true
                    break
                end
            end

            if !found
                error("Element '$part' not found in path '$path_string'")
            end
        end
    end
end

"""
    save_xml(editor; save_path=nothing)

Save the XML document to a file.

# Example
```julia
editor = load_xml("file.pgo")
set_parameter!(editor, "Temperature", "25")
save_xml(editor, save_path="modified.pgo")  # Save to new file
# or
save_xml(editor)  # Save to original file
```
"""
function save_xml(editor::XMLEditor; save_path=nothing, verbose=false)
    output_path = isnothing(save_path) ? editor.file_path : save_path
    write(output_path, editor.doc)
    if verbose
        println("Saved to: $output_path")
    end
    return output_path
end

"""
    edit_parameter(file_path, path_string, new_value; save_path=nothing)

Edit a parameter in a PGOPHER XML file using a dot-separated path.

# Example
```julia
edit_parameter("asym_top_15272_C2v.pgo", "Species.AsymmetricTop.Excited.v=1.A", "0.08")
```
"""
function edit_parameter(file_path, path_string, new_value; save_path=nothing, verbose=false)
    # Read the XML document
    doc = readxml(file_path)
    root_node = root(doc)

    # Split the path into components
    path_parts = split(path_string, ".")

    # Start at the root
    current = root_node

    # Navigate through each path component
    for (i, part) in enumerate(path_parts)
        is_last = (i == length(path_parts))

        if is_last
            # This is the parameter name we want to edit
            # Find the <Parameter Name="X" Value="Y"/> element
            found = false
            for child in eachelement(current)
                if nodename(child) == "Parameter"
                    name_attr = child["Name"]
                    if name_attr == part
                        # Update the value
                        child["Value"] = string(new_value)
                        if verbose
                            println("Updated $path_string: $(child["Value"])")
                        end
                        found = true
                        break
                    end
                end
            end

            if !found
                error("Parameter '$part' not found at path '$path_string'")
            end
        else
            # Navigate to the next level
            # Check if this component has an '=' (like "v=1")
            found = false
            for child in eachelement(current)
                child_name = nodename(child)
                name_attr = try
                    child["Name"]
                catch KeyError
                    nothing
                end

                # Try to match against Name attribute first, then node name
                if !isnothing(name_attr) && name_attr == part
                    current = child
                    found = true
                    break
                elseif child_name == part
                    current = child
                    found = true
                    break
                end
            end

            if !found
                error("Element '$part' not found in path '$path_string'")
            end
        end
    end

    # Save the document
    output_path = isnothing(save_path) ? file_path : save_path
    write(output_path, doc)
    if verbose
        println("Saved to: $output_path")
    end

    return doc
end

"""
    get_parameter(file_path, path_string)

Get the value of a parameter using a dot-separated path.

# Example
```julia
value = get_parameter("asym_top_15272_C2v.pgo", "Species.AsymmetricTop.Excited.v=1.A")
```
"""
function get_parameter(file_path, path_string)
    doc = readxml(file_path)
    root_node = root(doc)

    path_parts = split(path_string, ".")
    current = root_node

    for (i, part) in enumerate(path_parts)
        is_last = (i == length(path_parts))

        if is_last
            # Find the parameter
            for child in eachelement(current)
                if nodename(child) == "Parameter"
                    name_attr = child["Name"]
                    if name_attr == part
                        return child["Value"]
                    end
                end
            end
            error("Parameter '$part' not found at path '$path_string'")
        else
            # Navigate to next level
            found = false
            for child in eachelement(current)
                child_name = nodename(child)
                name_attr = try
                    child["Name"]
                catch KeyError
                    nothing
                end

                # Try to match against Name attribute first, then node name
                if !isnothing(name_attr) && name_attr == part
                    current = child
                    found = true
                    break
                elseif child_name == part
                    current = child
                    found = true
                    break
                end
            end

            if !found
                error("Element '$part' not found in path '$path_string'")
            end
        end
    end
end

"""
    list_parameters(file_path)

List all parameters in the XML file with their paths and values.
"""
function list_parameters(file_path)
    doc = readxml(file_path)

    function traverse(node, path="")
        results = []

        for child in eachelement(node)
            child_name = nodename(child)

            # If this is a Parameter element itself, add it to results
            if child_name == "Parameter"
                param_name = child["Name"]
                param_value = child["Value"]
                full_path = isempty(path) ? param_name : "$path.$param_name"
                push!(results, (full_path, param_value))
                continue  # Don't recurse into Parameter elements
            end

            # Build the path component
            # Try to get Name attribute, fall back to node name
            component = try
                child["Name"]
            catch KeyError
                child_name
            end

            new_path = isempty(path) ? component : "$path.$component"

            # Recurse into this child
            append!(results, traverse(child, new_path))
        end

        return results
    end

    params = traverse(root(doc))

    for (path, value) in params
        println("$path = $value")
    end

    return params
end

