# Dovenet Plugin for Nuke
# This file is loaded first when Nuke initializes the Dovenet plugin

import nuke

def load_dovenet_plugin():
    """Initialize the Dovenet plugin"""
    try:
        # Import and setup the menu
        import menu
        print("Dovenet plugin loaded successfully")
    except Exception as e:
        nuke.message("Failed to load Dovenet plugin: %s" % str(e))
        print("Dovenet plugin loading error: %s" % str(e))

# Load the plugin when this module is imported
load_dovenet_plugin()