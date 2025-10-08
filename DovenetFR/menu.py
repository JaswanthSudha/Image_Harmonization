import nuke
from dovenet_client import run_dovenet_via_server

m = nuke.menu("Nuke")
menu = m.addMenu("DoveNet")

def harmonize_selected():
    sel = nuke.selectedNode()
    if sel.Class() == "Read":
        in_path = sel["file"].value()
        out_path = in_path.replace(".png", "_harmonized.png")  # adjust as needed
        node = run_dovenet_via_server(in_path, out_path, use_patch=True)
        if node:
            node.setXYpos(sel.xpos() + 150, sel.ypos())

def create_composite_with_mask():
    """
    Extracts alpha channel from CG image to create mask.
    User can then manually create merge and select both merge+mask for harmonization.
    """
    # Get selected node (should be CG Read node)
    selected_nodes = nuke.selectedNodes()
    
    if len(selected_nodes) != 1:
        nuke.message("Please select exactly 1 Read node (CG image with alpha)")
        return
    
    cg_node = selected_nodes[0]
    if cg_node.Class() != "Read":
        nuke.message("Please select a Read node")
        return
    
    # Get position for new node
    base_x = cg_node.xpos()
    base_y = cg_node.ypos()
    
    # Extract alpha channel from CG as mask
    alpha_mask = nuke.nodes.Shuffle(
        inputs=[cg_node],
        red="alpha",
        green="alpha", 
        blue="alpha",
        alpha="alpha",
        name="CG_AlphaMask"
    )
    alpha_mask.setXYpos(base_x + 150, base_y)
    
    nuke.message("Alpha mask created! Now:\n1. Manually create merge with background\n2. Select both merge node and alpha mask\n3. Use 'Send Merge+Mask to DoveNet'")

def send_merge_and_mask_to_dovenet(use_patch=True):
    """
    Sends selected merge node and mask to DoveNet server
    """
    selected_nodes = nuke.selectedNodes()
    if len(selected_nodes) != 2:
        nuke.message("Please select exactly 2 nodes: merge node and mask node")
        return
    # Find merge node and mask node
    merge_node = None
    mask_node = None
    for node in selected_nodes:
        if node.Class() == "Merge2":
            merge_node = node
        else:
            mask_node = node
    if not merge_node:
        nuke.message("Please select one Merge2 node")
        return
    if not mask_node:
        nuke.message("Please select one mask node (alpha channel or other)")
        return
    # Get positions for temporary write nodes (won't stay in graph)
    base_x = merge_node.xpos()
    base_y = merge_node.ypos()

    # Create temporary file paths with better path handling
    import tempfile
    import os
    import time
    
    # Use a safe, writable directory
    script_dir = nuke.script_directory()
    if script_dir and os.path.exists(script_dir) and os.access(script_dir, os.W_OK):
        temp_dir = script_dir
    else:
        # Use user's home directory as fallback
        temp_dir = os.path.expanduser("~")
        # Create a dovenet subfolder in home directory
        dovenet_temp_dir = os.path.join(temp_dir, "dovenet_temp")
        if not os.path.exists(dovenet_temp_dir):
            os.makedirs(dovenet_temp_dir)
        temp_dir = dovenet_temp_dir
    # Create safe filenames without special characters
    timestamp = str(int(time.time()))
    composite_temp_path = os.path.join(temp_dir, f"composite_{timestamp}.exr").replace("\\", "/")
    mask_temp_path = os.path.join(temp_dir, f"mask_{timestamp}.exr").replace("\\", "/")
    # Render the composite and mask
    nuke.message("Rendering composite and mask to temporary files...")
    try:
        # Create temporary Write nodes, render, then delete them
        # Add Reformat nodes to handle Non-commercial resolution limits
        # Create Reformat nodes to limit resolution to 1920x1080 for Non-commercial Nuke
        composite_reformat = nuke.nodes.Reformat(
            inputs=[merge_node],
            format="HD_1080",
            resize="fit",
            name="DoveNet_CompositeReformat"
        )
        mask_reformat = nuke.nodes.Reformat(
            inputs=[mask_node],
            format="HD_1080",
            resize="fit",
            name="DoveNet_MaskReformat"
        )
        composite_write = nuke.nodes.Write(
            inputs=[composite_reformat],
            file=composite_temp_path
        )
        mask_write = nuke.nodes.Write(
            inputs=[mask_reformat],
            file=mask_temp_path
        )
        # Execute the write nodes (render the files)
        print(f"Using temp directory: {temp_dir}")
        print(f"Rendering composite to: {composite_temp_path}")
        print(f"Rendering mask to: {mask_temp_path}")
        # Check if we can write to the target directory
        if not os.access(temp_dir, os.W_OK):
            raise Exception(f"No write permission to directory: {temp_dir}")
        # Verify the directory exists
        if not os.path.exists(temp_dir):
            raise Exception(f"Directory does not exist: {temp_dir}")
        nuke.execute(composite_write.name(), 1, 1)
        nuke.execute(mask_write.name(), 1, 1)
        # Verify files were created successfully
        if not os.path.exists(composite_temp_path):
            raise Exception(f"Failed to create composite file: {composite_temp_path}")
        if not os.path.exists(mask_temp_path):
            raise Exception(f"Failed to create mask file: {mask_temp_path}")
        print("Files rendered successfully")
        # Delete the temporary nodes from the graph
        nuke.delete(composite_write)
        nuke.delete(mask_write)
        nuke.delete(composite_reformat)
        nuke.delete(mask_reformat)
        
        # Prepare output path for harmonized result  
        output_path = os.path.join(temp_dir, f"harmonized_{timestamp}.exr").replace("\\", "/")
        
        # Send to DoveNet server
        harmonized_node = run_dovenet_via_server(
            input_path=composite_temp_path,
            output_path=output_path,
            mask_path=mask_temp_path,
            use_patch=use_patch
        )
        
        if harmonized_node:
            harmonized_node.setXYpos(base_x + 300, base_y)
            nuke.message("Harmonization completed!")
        
        # Clean up temporary files
        try:
            os.remove(composite_temp_path)
            os.remove(mask_temp_path)
        except:
            pass  # Ignore cleanup errors
        
    except Exception as e:
        # Clean up temporary nodes if they exist
        try:
            if 'composite_write' in locals():
                nuke.delete(composite_write)
            if 'mask_write' in locals():
                nuke.delete(mask_write)
            if 'composite_reformat' in locals():
                nuke.delete(composite_reformat)
            if 'mask_reformat' in locals():
                nuke.delete(mask_reformat)
        except:
            pass
        
        nuke.message("Error during rendering: %s" % str(e))

def harmonize_merge_node(use_patch=True):
    """
    Harmonize a selected Merge node by generating mask and sending to server
    """
    sel = nuke.selectedNode()
    if not sel or sel.Class() != "Merge2":
        nuke.message("Please select a Merge node")
        return
    
    # Get the merge node inputs
    bg_input = sel.input(0)  # Background (A input)
    fg_input = sel.input(1)  # Foreground (B input) 
    
    if not bg_input or not fg_input:
        nuke.message("Merge node must have both A and B inputs connected")
        return
    
    base_x = sel.xpos()
    base_y = sel.ypos()
    
    # Generate mask from foreground alpha
    mask_node = nuke.nodes.Shuffle(
        inputs=[fg_input],
        red="alpha",
        green="alpha", 
        blue="alpha",
        alpha="alpha",
        name="DoveNet_AlphaMask"
    )
    mask_node.setXYpos(base_x + 100, base_y + 100)
    
    # Create temporary file paths with better path handling
    import tempfile
    import os
    import time
    
    # Use a safe, writable directory
    script_dir = nuke.script_directory()
    if script_dir and os.path.exists(script_dir) and os.access(script_dir, os.W_OK):
        temp_dir = script_dir
    else:
        # Use user's home directory as fallback
        temp_dir = os.path.expanduser("~")
        # Create a dovenet subfolder in home directory
        dovenet_temp_dir = os.path.join(temp_dir, "dovenet_temp")
        if not os.path.exists(dovenet_temp_dir):
            os.makedirs(dovenet_temp_dir)
        temp_dir = dovenet_temp_dir
    
    # Create safe filenames without special characters
    timestamp = str(int(time.time()))
    composite_temp_path = os.path.join(temp_dir, f"merge_composite_{timestamp}.exr").replace("\\", "/")
    mask_temp_path = os.path.join(temp_dir, f"merge_mask_{timestamp}.exr").replace("\\", "/")
    
    try:
        # Create temporary Write nodes with Reformat for resolution limits
        # Add Reformat nodes to handle Non-commercial resolution limits
        
        composite_reformat = nuke.nodes.Reformat(
            inputs=[sel],
            format="HD_1080",
            resize="fit",
            name="DoveNet_MergeCompositeReformat"
        )
        
        mask_reformat = nuke.nodes.Reformat(
            inputs=[mask_node],
            format="HD_1080",
            resize="fit", 
            name="DoveNet_MergeMaskReformat"
        )
        
        composite_write = nuke.nodes.Write(
            inputs=[composite_reformat],
            file=composite_temp_path
        )
        mask_write = nuke.nodes.Write(
            inputs=[mask_reformat],
            file=mask_temp_path
        )
        
        # Render composite and mask
        print(f"Using temp directory: {temp_dir}")
        print(f"Rendering merge composite to: {composite_temp_path}")
        print(f"Rendering merge mask to: {mask_temp_path}")
        
        # Check if we can write to the target directory
        if not os.access(temp_dir, os.W_OK):
            raise Exception(f"No write permission to directory: {temp_dir}")
        
        # Verify the directory exists
        if not os.path.exists(temp_dir):
            raise Exception(f"Directory does not exist: {temp_dir}")
        
        nuke.execute(composite_write.name(), 1, 1)
        nuke.execute(mask_write.name(), 1, 1)
        
        # Verify files were created successfully
        if not os.path.exists(composite_temp_path):
            raise Exception(f"Failed to create composite file: {composite_temp_path}")
        if not os.path.exists(mask_temp_path):
            raise Exception(f"Failed to create mask file: {mask_temp_path}")
        
        print("Files rendered successfully")
        
        # Delete the temporary nodes from the graph
        nuke.delete(composite_write)
        nuke.delete(mask_write)
        nuke.delete(composite_reformat)
        nuke.delete(mask_reformat)
        nuke.delete(mask_node)  # Also delete the temporary mask node
        
        # Prepare output path for harmonized result
        output_path = os.path.join(temp_dir, f"merge_harmonized_{timestamp}.exr").replace("\\", "/")
        
        harmonized_node = run_dovenet_via_server(
            input_path=composite_temp_path,
            output_path=output_path,
            mask_path=mask_temp_path,
            use_patch=use_patch
        )
        
        if harmonized_node:
            harmonized_node.setXYpos(base_x + 200, base_y)
            nuke.message("Harmonization completed!")
        
        # Clean up temporary files
        try:
            os.remove(composite_temp_path)
            os.remove(mask_temp_path)
        except:
            pass  # Ignore cleanup errors
            
    except Exception as e:
        # Clean up temporary nodes if they exist
        try:
            if 'composite_write' in locals():
                nuke.delete(composite_write)
            if 'mask_write' in locals():
                nuke.delete(mask_write)
            if 'composite_reformat' in locals():
                nuke.delete(composite_reformat)
            if 'mask_reformat' in locals():
                nuke.delete(mask_reformat)
            if 'mask_node' in locals():
                nuke.delete(mask_node)
        except:
            pass
        
        nuke.message("Error: %s" % str(e))

# Add menu commands
menu.addCommand("Run Harmonization (Server)", harmonize_selected)
menu.addSeparator()
menu.addCommand("Extract Alpha Mask from CG", create_composite_with_mask)
menu.addCommand("Send Merge+Mask to DoveNet (Patch-based)", lambda: send_merge_and_mask_to_dovenet(use_patch=True))
menu.addCommand("Send Merge+Mask to DoveNet (Standard)", lambda: send_merge_and_mask_to_dovenet(use_patch=False))
menu.addCommand("Harmonize Merge Node (Patch-based)", lambda: harmonize_merge_node(use_patch=True))
menu.addCommand("Harmonize Merge Node (Standard)", lambda: harmonize_merge_node(use_patch=False))
