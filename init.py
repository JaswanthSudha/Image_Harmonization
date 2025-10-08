import nuke

# Add existing plugins
nuke.pluginAddPath('NukeSamurai', addToSysPath=True)
nuke.pluginAddPath('NukeSamurai/scripts', addToSysPath=True)
nuke.pluginAddPath('NukeSamurai/icons', addToSysPath=False)
nuke.pluginAddPath('RotoMaker', addToSysPath=True)
nuke.pluginAddPath('DovenetFR', addToSysPath=True)

# Add Dovenet plugin
nuke.pluginAddPath('Dovenet', addToSysPath=True)

# Add StereoCrafter plugin
nuke.pluginAddPath('StereoCrafter', addToSysPath=True)

# Import and register StereoCrafter
try:
    import StereoCrafter
    print("StereoCrafter plugin loaded successfully")
except Exception as e:
    print(f"Failed to load StereoCrafter: {e}")

# Import and register Dovenet
try:
    # Import the menu module to register DoveNet commands
    import menu as dovenet_menu
    print("Dovenet plugin loaded successfully")
except Exception as e:
    print(f"Failed to load Dovenet: {e}")
