# Test script for Dovenet plugin
# Run this inside Nuke's Script Editor to verify the plugin is working

try:
    # Test if requests module is available
    import requests
    print("✓ requests module is available")
except ImportError as e:
    print("✗ requests module not found:", e)
    print("You may need to install requests in Nuke's Python environment")

try:
    # Test if dovenet_client can be imported
    from dovenet_client import run_dovenet_via_server
    print("✓ dovenet_client module imported successfully")
except ImportError as e:
    print("✗ dovenet_client import failed:", e)

# Check if DoveNet menu was created
import nuke
main_menu = nuke.menu("Nuke")
dovenet_menu = None
for item in main_menu.items():
    if item.name() == "DoveNet":
        dovenet_menu = item
        break

if dovenet_menu:
    print("✓ DoveNet menu found in Nuke menu bar")
    # List menu items
    for item in dovenet_menu.items():
        print(f"  - {item.name()}")
else:
    print("✗ DoveNet menu not found")

print("\nDovenet plugin integration test completed.")