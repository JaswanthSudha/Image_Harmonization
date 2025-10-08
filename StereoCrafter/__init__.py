"""
StereoCrafter Nuke Plugin - Main Module
AI-powered stereo generation plugin for Nuke
"""

from .stereo_plugin import show_stereocrafter_panel, register_stereocrafter
from .test_setup import register_test_menu
from .verify_setup import register_verification_menu

# Auto-register the plugin when imported
register_stereocrafter()
register_test_menu()
register_verification_menu()

__version__ = "1.0.0"
__author__ = "StereoCrafter Team"