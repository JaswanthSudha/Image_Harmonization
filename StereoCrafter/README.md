# StereoCrafter Nuke Plugin

## Overview
StereoCrafter is an AI-powered stereo generation plugin for Nuke that converts 2D images and videos into stereo 3D content. The plugin supports both online (server-based) and offline (testing) modes.

## Features
- **AI-Powered Stereo Generation**: Convert 2D content to stereo 3D using advanced AI algorithms
- **Multiple Output Formats**: Side-by-Side, Anaglyph, or Both
- **Real-time Progress Tracking**: WebSocket-based progress updates
- **Offline Testing Mode**: Test UI and workflow without server connection
- **Batch Processing**: Process multiple frames or frame sequences
- **Flexible Input**: Works with any Read node or image sequence

## Installation
The plugin is already installed in your Nuke directory:
```
~/.nuke/StereoCrafter/
├── __init__.py
├── stereo_plugin.py
├── test_setup.py
└── README.md
```

The plugin is automatically loaded when Nuke starts via the updated `init.py` file.

## Usage

### Testing Mode (Recommended for First Use)
1. Open Nuke
2. Go to **Nodes** menu → **StereoCrafter** → **Create Test Setup**
3. This creates a test project with animated content
4. Go to **Nodes** menu → **StereoCrafter** → **Generate Stereo...**
5. In the panel:
   - Enable **"Offline Testing Mode"**
   - Select **"noise1"** as Source Node
   - Click **"Generate Stereo"**

### Production Mode (With Server)
1. Set up the StereoCrafter server (separate installation required)
2. Load your image sequence using a Read node
3. Go to **Nodes** menu → **StereoCrafter** → **Generate Stereo...**
4. Configure settings:
   - **Server URL**: Your server address (default: http://localhost:8000)
   - **Source Node**: Select your Read node
   - **Frame Range**: Specify frames to process
   - **Max Disparity**: Depth strength (1-100)
   - **Tile Count**: Use higher values for 2K+ resolution
   - **Output Format**: Choose stereo format
5. Click **"Check Server"** to verify connection
6. Click **"Generate Stereo"** to start processing

## Settings Explained

### Processing Settings
- **Max Disparity (1-100)**: Controls the depth strength in the stereo effect
  - Lower values: Subtle depth effect
  - Higher values: Stronger 3D effect
- **Tile Count (1-4)**: For high-resolution processing
  - 1: Standard processing
  - 2-4: Better for 2K/4K resolution (requires more memory)

### Output Formats
- **Side-by-Side**: Left and right images placed side by side
- **Anaglyph**: Red/cyan 3D glasses compatible format
- **Both**: Generates both formats

## Menu Locations
The plugin adds menu items in multiple locations:
- **Nodes** → **StereoCrafter** → **Generate Stereo...**
- **Nuke** (main menu) → **StereoCrafter**

## Testing Features
- **Create Test Setup**: Creates an animated test project
- **Test UI**: Opens the plugin interface
- **Offline Testing Mode**: Simulates processing without server
- **Mock Progress Updates**: Shows realistic progress simulation

## Requirements

### For Testing Mode
- Nuke (any recent version)
- Python (included with Nuke)

### For Production Mode
- Python `requests` library: `pip install requests`
- Python `websocket-client` library: `pip install websocket-client`
- StereoCrafter server running on specified URL

## Troubleshooting

### Plugin Not Loading
1. Check that files are in `~/.nuke/StereoCrafter/`
2. Verify `init.py` includes StereoCrafter plugin path
3. Restart Nuke
4. Check Script Editor for error messages

### "No Source Nodes" Error
1. Create a Read node first, or
2. Use **"Create Test Setup"** for testing

### Server Connection Issues
1. Enable **"Offline Testing Mode"** for testing
2. Verify server URL is correct
3. Ensure server is running and accessible
4. Check firewall settings

### Performance Issues
- Use smaller frame ranges for testing
- Reduce tile count for lower resolution
- Process in batches for long sequences

## API Reference

### StereoCrafterClient
Main client class for server communication:
- `check_health()`: Verify server connection
- `submit_job(frames, settings)`: Submit processing job
- `get_job_status(job_id)`: Get processing status
- `get_job_result(job_id)`: Retrieve results

### MockStereoCrafterClient
Testing client for offline mode:
- Same interface as real client
- Generates mock data for testing
- No network requirements

## File Structure
```
StereoCrafter/
├── __init__.py           # Main plugin module
├── stereo_plugin.py      # Core plugin functionality
├── test_setup.py         # Testing utilities
└── README.md            # This documentation
```

## Support
For issues or questions:
1. Check the Script Editor for error messages
2. Try **"Offline Testing Mode"** first
3. Verify all requirements are installed
4. Test with the provided test setup

## Version History
- **v1.0.0**: Initial release with testing support
  - Offline testing mode
  - UI improvements
  - Better error handling
  - Test environment setup