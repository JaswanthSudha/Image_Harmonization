"""
StereoCrafter Test Script
Creates a simple test setup for testing the StereoCrafter plugin in Nuke
"""

import nuke

def create_test_setup():
    """Create a basic test setup with Read node and sample project"""
    try:
        # Clear existing nodes (ask user first)
        if nuke.allNodes():
            result = nuke.ask("This will clear existing nodes and create a test setup. Continue?")
            if not result:
                return
        
        # Clear the script
        nuke.scriptClear()
        
        # Set project settings
        root = nuke.root()
        root['first_frame'].setValue(1)
        root['last_frame'].setValue(10)
        root['fps'].setValue(24)
        root['format'].setValue('HD_1080')
        
        # Create a color generator for testing (instead of Read node)
        constant = nuke.createNode("Constant")
        constant.setName("TestSource")
        constant['color'].setValue([0.5, 0.7, 0.9, 1.0])  # Light blue
        constant.setXYpos(100, 100)
        constant['label'].setValue("Test Source\\nFor StereoCrafter")
        
        # Add some animated text for visual feedback
        text = nuke.createNode("Text2")
        text.setInput(0, constant)
        text['message'].setValue("Frame [frame]\\nStereoCrafter Test")
        text['size'].setValue(48)
        text['color'].setValue([1, 1, 1, 1])
        text.setXYpos(100, 200)
        text['label'].setValue("Animated Text\\nChanges per frame")
        
        # Add noise for texture
        noise = nuke.createNode("Noise")
        noise.setInput(0, text)
        noise['size'].setValue(0.1)
        noise['zoffset'].setExpression("frame * 0.1")
        noise.setXYpos(100, 300)
        noise['label'].setValue("Animated Noise\\nFor texture")
        
        # Create a viewer
        viewer = nuke.createNode("Viewer")
        viewer.setInput(0, noise)
        viewer.setXYpos(100, 400)
        
        # Create a Write node for reference
        write = nuke.createNode("Write")
        write.setInput(0, noise)
        write['file'].setValue("[python {nuke.script_directory()}]/test_frames/test_frame_####.jpg")
        write['file_type'].setValue('jpg')
        write.setXYpos(300, 300)
        write['label'].setValue("Reference Output\\n(Optional)")
        
        # Select the test source
        text.setSelected(True)
        
        # Show message
        nuke.message(
            "StereoCrafter Test Setup Created!\n\n"
            "Setup includes:\n"
            "• Constant node (TestSource) - Use this as source\n"
            "• Text2 node with frame counter\n"
            "• Noise node for texture\n"
            "• Viewer and Write nodes\n\n"
            "To test StereoCrafter:\n"
            "1. Go to menu: StereoCrafter > Generate Stereo\n"
            "2. Select 'noise1' as source node\n"
            "3. Enable 'Offline Testing Mode'\n"
            "4. Click 'Generate Stereo'\n\n"
            "Frame range is set to 1-10 for quick testing."
        )
        
        print("StereoCrafter test setup created successfully")
        
    except Exception as e:
        nuke.message(f"Error creating test setup: {str(e)}")
        print(f"Test setup error: {e}")

def test_stereocrafter_ui():
    """Test opening the StereoCrafter UI"""
    try:
        from StereoCrafter.stereo_plugin import show_stereocrafter_panel
        show_stereocrafter_panel()
    except Exception as e:
        nuke.message(f"Error opening StereoCrafter UI: {str(e)}")
        print(f"UI test error: {e}")

def register_test_menu():
    """Register test menu items"""
    try:
        # Add to main menu
        toolbar = nuke.menu("Nodes")
        if toolbar.findItem("StereoCrafter"):
            stereo_menu = toolbar.findItem("StereoCrafter")
            stereo_menu.addSeparator()
            stereo_menu.addCommand("Create Test Setup", create_test_setup)
            stereo_menu.addCommand("Test UI", test_stereocrafter_ui)
        
        print("StereoCrafter test menu registered")
        
    except Exception as e:
        print(f"Failed to register test menu: {e}")

# Auto-register when imported
if __name__ != "__main__":
    register_test_menu()