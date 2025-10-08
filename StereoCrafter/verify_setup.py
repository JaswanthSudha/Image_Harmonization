"""
StereoCrafter Setup Verification Script
Run this to verify the plugin installation
"""

import os
import sys

def verify_installation():
    """Verify StereoCrafter installation"""
    print("=== StereoCrafter Installation Verification ===\n")
    
    # Check if running in Nuke
    try:
        import nuke
        print("✓ Running in Nuke environment")
        nuke_version = nuke.NUKE_VERSION_STRING
        print(f"  Nuke version: {nuke_version}")
    except ImportError:
        print("✗ Not running in Nuke environment")
        print("  This script should be run within Nuke")
        return False
    
    # Check file structure
    print("\n--- File Structure Check ---")
    nuke_dir = os.path.expanduser("~/.nuke")
    stereo_dir = os.path.join(nuke_dir, "StereoCrafter")
    
    required_files = [
        "__init__.py",
        "stereo_plugin.py", 
        "test_setup.py",
        "README.md"
    ]
    
    if os.path.exists(stereo_dir):
        print(f"✓ StereoCrafter directory exists: {stereo_dir}")
        
        for file in required_files:
            file_path = os.path.join(stereo_dir, file)
            if os.path.exists(file_path):
                print(f"✓ {file}")
            else:
                print(f"✗ {file} - MISSING")
                return False
    else:
        print(f"✗ StereoCrafter directory not found: {stereo_dir}")
        return False
    
    # Check init.py integration
    print("\n--- Init.py Integration Check ---")
    init_file = os.path.join(nuke_dir, "init.py")
    if os.path.exists(init_file):
        with open(init_file, 'r') as f:
            content = f.read()
            if "StereoCrafter" in content:
                print("✓ StereoCrafter registered in init.py")
            else:
                print("✗ StereoCrafter not found in init.py")
                return False
    else:
        print("✗ init.py not found")
        return False
    
    # Check plugin loading
    print("\n--- Plugin Loading Check ---")
    try:
        import StereoCrafter
        print("✓ StereoCrafter module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import StereoCrafter: {e}")
        return False
    
    # Check menu registration
    print("\n--- Menu Registration Check ---")
    try:
        toolbar = nuke.menu("Nodes")
        if toolbar.findItem("StereoCrafter"):
            print("✓ StereoCrafter menu found in Nodes")
        else:
            print("✗ StereoCrafter menu not found in Nodes")
            return False
            
        main_menu = nuke.menu("Nuke")
        if main_menu.findItem("StereoCrafter"):
            print("✓ StereoCrafter found in main menu")
        else:
            print("✗ StereoCrafter not found in main menu")
            return False
    except Exception as e:
        print(f"✗ Menu check failed: {e}")
        return False
    
    # Check dependencies
    print("\n--- Dependencies Check ---")
    
    # Check optional dependencies
    try:
        import requests
        print("✓ requests library available (for server mode)")
    except ImportError:
        print("⚠ requests library not available (offline mode only)")
    
    try:
        import websocket
        print("✓ websocket-client library available (for real-time updates)")
    except ImportError:
        print("⚠ websocket-client library not available (no real-time updates)")
    
    print("\n=== Verification Complete ===")
    print("✓ StereoCrafter is properly installed and ready to use!")
    print("\nNext steps:")
    print("1. Go to Nodes > StereoCrafter > Create Test Setup")
    print("2. Then go to Nodes > StereoCrafter > Generate Stereo...")
    print("3. Enable 'Offline Testing Mode' and test the UI")
    
    return True

def show_usage_instructions():
    """Show usage instructions"""
    instructions = """
=== StereoCrafter Usage Instructions ===

QUICK START (Testing):
1. Nodes menu → StereoCrafter → Create Test Setup
2. Nodes menu → StereoCrafter → Generate Stereo...
3. Enable "Offline Testing Mode"
4. Select "noise1" as Source Node
5. Click "Generate Stereo"

PRODUCTION USE:
1. Load image sequence with Read node
2. Install: pip install requests websocket-client
3. Start StereoCrafter server
4. Nodes menu → StereoCrafter → Generate Stereo...
5. Disable "Offline Testing Mode"
6. Configure server URL and settings
7. Click "Check Server" then "Generate Stereo"

MENU LOCATIONS:
- Nodes → StereoCrafter → Generate Stereo...
- Nodes → StereoCrafter → Create Test Setup
- Nodes → StereoCrafter → About
- Nuke (main menu) → StereoCrafter

For detailed documentation, see:
~/.nuke/StereoCrafter/README.md
    """
    print(instructions)

# Add menu item for verification
def register_verification_menu():
    """Register verification menu"""
    try:
        toolbar = nuke.menu("Nodes")
        if toolbar.findItem("StereoCrafter"):
            stereo_menu = toolbar.findItem("StereoCrafter")
            stereo_menu.addSeparator()
            stereo_menu.addCommand("Verify Installation", verify_installation)
            stereo_menu.addCommand("Usage Instructions", show_usage_instructions)
    except:
        pass

# Auto-register when imported
if __name__ != "__main__":
    register_verification_menu()
else:
    # Run verification if executed directly
    verify_installation()