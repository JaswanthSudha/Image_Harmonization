"""
StereoCrafter Nuke Plugin
AI-powered stereo generation plugin for Nuke with offline testing support
Place this file in ~/.nuke/StereoCrafter/ directory
"""

import nuke
import nukescripts
import os
import sys
import json
import base64
import threading
import queue
from datetime import datetime
from typing import List, Dict, Any, Optional
import tempfile
import time

# Add requests for API communication (optional for testing)
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("StereoCrafter: Requests not available - running in offline mode")

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("StereoCrafter: WebSocket not available - running without real-time updates")

# Configuration
DEFAULT_SERVER_URL = os.getenv("STEREOCRAFTER_SERVER", "http://localhost:8000")
DEFAULT_WS_URL = DEFAULT_SERVER_URL.replace("http://", "ws://").replace("https://", "wss://")

class MockStereoCrafterClient:
    """Mock client for testing without server connection"""
    
    def __init__(self, server_url: str = DEFAULT_SERVER_URL):
        self.server_url = server_url
        self.current_job_id = None
        self.mock_processing = False
        
    def check_health(self) -> bool:
        """Mock health check - always returns True for testing"""
        return True
    
    def submit_job(self, frames: List[bytes], settings: Dict[str, Any]) -> str:
        """Mock job submission"""
        import uuid
        job_id = str(uuid.uuid4())
        self.current_job_id = job_id
        print(f"Mock job submitted: {job_id}")
        print(f"Settings: {settings}")
        print(f"Frame count: {len(frames)}")
        return job_id
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Mock job status"""
        if not self.mock_processing:
            return {"status": "completed", "progress": 100}
        return {"status": "processing", "progress": 50}
    
    def get_job_result(self, job_id: str, format: str = "sbs") -> Dict[str, Any]:
        """Mock job result with placeholder data"""
        # Create some mock stereo frames (base64 encoded placeholder)
        mock_frames = []
        # This is a tiny placeholder - in real use this would be actual stereo image data
        placeholder_data = b"Mock stereo frame data"
        for i in range(3):  # Mock 3 frames
            mock_frames.append(base64.b64encode(placeholder_data).decode('utf-8'))
        
        return {
            "stereo_frames": mock_frames,
            "format": format,
            "metadata": {
                "frame_count": len(mock_frames),
                "processing_time": "30s",
                "disparity_range": 20.0
            }
        }
    
    def connect_websocket(self, job_id: str, on_message=None):
        """Mock WebSocket connection"""
        print(f"Mock WebSocket connected for job: {job_id}")
        if on_message:
            # Simulate progress updates
            def mock_updates():
                for progress in [25, 50, 75, 100]:
                    message = json.dumps({
                        "progress": progress,
                        "status": "processing",
                        "stage": f"Processing frame {progress//25}/4"
                    })
                    on_message(None, message)
                    time.sleep(1)
            
            update_thread = threading.Thread(target=mock_updates)
            update_thread.daemon = True
            update_thread.start()
    
    def disconnect_websocket(self):
        """Mock WebSocket disconnect"""
        print("Mock WebSocket disconnected")

class StereoCrafterClient:
    """Real client for communicating with StereoCrafter server"""

    def __init__(self, server_url: str = DEFAULT_SERVER_URL):
        self.server_url = server_url.rstrip('/')
        self.ws_url = DEFAULT_WS_URL
        self.current_job_id = None
        self.ws_connection = None
        self.progress_callback = None

    def check_health(self) -> bool:
        """Check if server is healthy"""
        if not REQUESTS_AVAILABLE:
            return False
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def submit_job(self, frames: List[bytes], settings: Dict[str, Any]) -> str:
        """Submit a stereo generation job"""
        if not REQUESTS_AVAILABLE:
            raise Exception("Requests library not available")
            
        # Encode frames to base64
        frames_b64 = []
        for frame_data in frames:
            frame_b64 = base64.b64encode(frame_data).decode('utf-8')
            frames_b64.append(frame_b64)

        # Prepare request
        payload = {
            "frames": frames_b64,
            "settings": settings
        }

        # Submit job
        response = requests.post(
            f"{self.server_url}/api/v1/jobs/submit",
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        result = response.json()
        self.current_job_id = result['job_id']
        return self.current_job_id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status"""
        if not REQUESTS_AVAILABLE:
            raise Exception("Requests library not available")
            
        response = requests.get(
            f"{self.server_url}/api/v1/jobs/{job_id}/status",
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    def get_job_result(self, job_id: str, format: str = "sbs") -> Dict[str, Any]:
        """Get job result"""
        if not REQUESTS_AVAILABLE:
            raise Exception("Requests library not available")
            
        response = requests.get(
            f"{self.server_url}/api/v1/jobs/{job_id}/result",
            params={"format": format},
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def connect_websocket(self, job_id: str, on_message=None):
        """Connect to WebSocket for real-time updates"""
        if not WEBSOCKET_AVAILABLE:
            print("WebSocket not available - using polling instead")
            return
            
        try:
            ws_url = f"{self.ws_url}/ws/{job_id}"
            self.ws_connection = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=self._on_ws_error,
                on_close=self._on_ws_close
            )
            # Run in separate thread
            ws_thread = threading.Thread(target=self.ws_connection.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
        except Exception as e:
            print(f"WebSocket connection failed: {e}")

    def _on_ws_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def _on_ws_close(self, ws):
        print("WebSocket connection closed")

    def disconnect_websocket(self):
        """Disconnect WebSocket"""
        if self.ws_connection:
            self.ws_connection.close()
            self.ws_connection = None

class StereoCrafterPanel(nukescripts.PythonPanel):
    """Nuke panel for StereoCrafter"""

    def __init__(self):
        nukescripts.PythonPanel.__init__(self, "StereoCrafter - AI Stereo Generator")

        # Create knobs
        self.server_url = nuke.String_Knob("server_url", "Server URL", DEFAULT_SERVER_URL)
        self.addKnob(self.server_url)
        
        # Testing mode
        self.test_mode = nuke.Boolean_Knob("test_mode", "Offline Testing Mode", True)
        self.test_mode.setTooltip("Enable to test UI without server connection")
        self.addKnob(self.test_mode)

        self.divider1 = nuke.Text_Knob("divider1", "")
        self.addKnob(self.divider1)

        # Input settings
        self.source_node = nuke.Enumeration_Knob("source_node", "Source Node", self._get_read_nodes())
        self.addKnob(self.source_node)

        self.frame_range = nuke.String_Knob("frame_range", "Frame Range", f"{nuke.root().firstFrame()}-{nuke.root().lastFrame()}")
        self.addKnob(self.frame_range)
        
        self.refresh_nodes_btn = nuke.PyScript_Knob("refresh_nodes", "Refresh Nodes")
        self.addKnob(self.refresh_nodes_btn)

        self.divider2 = nuke.Text_Knob("divider2", "")
        self.addKnob(self.divider2)

        # Processing settings
        self.max_disp = nuke.Double_Knob("max_disp", "Max Disparity")
        self.max_disp.setValue(20.0)
        self.max_disp.setRange(1, 100)
        self.addKnob(self.max_disp)

        self.tile_num = nuke.Int_Knob("tile_num", "Tile Count")
        self.tile_num.setValue(1)
        self.tile_num.setRange(1, 4)
        self.tile_num.setTooltip("Use higher values for 2K+ resolution")
        self.addKnob(self.tile_num)

        self.output_format = nuke.Enumeration_Knob("output_format", "Output Format", ["Side-by-Side", "Anaglyph", "Both"])
        self.addKnob(self.output_format)

        self.divider3 = nuke.Text_Knob("divider3", "")
        self.addKnob(self.divider3)

        # Status
        self.status_text = nuke.Multiline_Eval_String_Knob("status", "Status", "Ready - Select source node to begin")
        self.status_text.setEnabled(False)
        self.addKnob(self.status_text)

        self.progress_bar = nuke.Int_Knob("progress", "Progress")
        self.progress_bar.setRange(0, 100)
        self.addKnob(self.progress_bar)

        # Buttons
        self.check_server_btn = nuke.PyScript_Knob("check_server", "Check Server")
        self.addKnob(self.check_server_btn)

        self.process_btn = nuke.PyScript_Knob("process", "Generate Stereo")
        self.addKnob(self.process_btn)

        self.cancel_btn = nuke.PyScript_Knob("cancel", "Cancel")
        self.cancel_btn.setEnabled(False)
        self.addKnob(self.cancel_btn)
        
        # Info button
        self.info_btn = nuke.PyScript_Knob("info", "About")
        self.addKnob(self.info_btn)

        # Client
        self.client = None
        self.current_job = None
        self.processing_thread = None

    def _get_read_nodes(self) -> List[str]:
        """Get list of Read nodes in the script"""
        read_nodes = []
        for node in nuke.allNodes():
            if node.Class() == "Read":
                read_nodes.append(node.name())
        return read_nodes if read_nodes else ["None"]

    def knobChanged(self, knob):
        """Handle knob changes"""
        if knob == self.check_server_btn:
            self._check_server()
        elif knob == self.process_btn:
            self._start_processing()
        elif knob == self.cancel_btn:
            self._cancel_processing()
        elif knob == self.refresh_nodes_btn:
            self._refresh_nodes()
        elif knob == self.info_btn:
            self._show_info()

    def _refresh_nodes(self):
        """Refresh the list of available nodes"""
        new_nodes = self._get_read_nodes()
        self.source_node.setValues(new_nodes)
        self.status_text.setValue("Node list refreshed")

    def _show_info(self):
        """Show plugin information"""
        info = """StereoCrafter - AI Stereo Generator

Features:
• Convert 2D images/videos to stereo 3D
• Multiple output formats (Side-by-Side, Anaglyph)
• Real-time progress tracking
• Offline testing mode

Usage:
1. Load images using Read node
2. Select source node in dropdown
3. Set frame range and parameters
4. Click 'Generate Stereo'

Testing Mode:
Enable 'Offline Testing Mode' to test the UI
without connecting to a server.

Requirements:
• Python requests (for server mode)
• Python websocket-client (for real-time updates)
• StereoCrafter server (for actual processing)
        """
        nuke.message(info)

    def _check_server(self):
        """Check server connection"""
        try:
            if self.test_mode.value():
                self.client = MockStereoCrafterClient(self.server_url.value())
                self.status_text.setValue("Using offline testing mode")
                nuke.message("Offline testing mode active!\nServer connection skipped.")
            else:
                self.client = StereoCrafterClient(self.server_url.value())
                if self.client.check_health():
                    self.status_text.setValue("Server connected successfully!")
                    nuke.message("Server is healthy and ready!")
                else:
                    self.status_text.setValue("Server not responding")
                    nuke.message("Cannot connect to server. Please check the URL and ensure server is running.")
        except Exception as e:
            self.status_text.setValue(f"Connection failed: {str(e)}")
            nuke.message(f"Error: {str(e)}")

    def _start_processing(self):
        """Start stereo generation"""
        # Initialize client based on mode
        if self.test_mode.value():
            self.client = MockStereoCrafterClient(self.server_url.value())
        elif not self.client:
            self.client = StereoCrafterClient(self.server_url.value())

        # Get source node
        source_name = self.source_node.value()
        if source_name == "None":
            nuke.message("Please select a source node or create a Read node first")
            return

        source_node = nuke.toNode(source_name)
        if not source_node:
            nuke.message(f"Cannot find node: {source_name}")
            return

        # Parse frame range
        try:
            frame_range = self.frame_range.value()
            if "-" in frame_range:
                start, end = map(int, frame_range.split("-"))
            else:
                start = end = int(frame_range)
        except:
            nuke.message("Invalid frame range. Use format: 1-100 or single frame number")
            return

        # Validate frame range
        if start > end:
            nuke.message("Start frame cannot be greater than end frame")
            return
        
        if end - start > 100:
            result = nuke.ask("Processing more than 100 frames may take a long time. Continue?")
            if not result:
                return

        # Disable controls during processing
        self.process_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status_text.setValue("Preparing for processing...")
        self.progress_bar.setValue(0)

        # Start processing in separate thread
        self.processing_thread = threading.Thread(
            target=self._process_frames,
            args=(source_node, start, end)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _process_frames(self, source_node, start_frame, end_frame):
        """Process frames in background thread"""
        try:
            # Extract frames
            frames = []
            frame_count = end_frame - start_frame + 1

            for i, frame_num in enumerate(range(start_frame, end_frame + 1)):
                progress = int((i / frame_count) * 30)  # Use 30% for frame extraction
                nuke.executeInMainThread(lambda p=progress: self.progress_bar.setValue(p))
                nuke.executeInMainThread(lambda fn=frame_num, ec=end_frame: 
                    self.status_text.setValue(f"Extracting frame {fn}/{ec}"))

                # Get frame data
                if self.test_mode.value():
                    # Mock frame data for testing
                    frames.append(b"Mock frame data")
                else:
                    frame_data = self._extract_frame(source_node, frame_num)
                    if frame_data:
                        frames.append(frame_data)

            if not frames:
                nuke.executeInMainThread(lambda: nuke.message("Failed to extract frames"))
                return

            # Prepare settings
            format_map = {"Side-by-Side": "sbs", "Anaglyph": "anaglyph", "Both": "both"}
            settings = {
                "max_disp": self.max_disp.value(),
                "tile_num": int(self.tile_num.value()),
                "output_format": format_map[self.output_format.value()]
            }

            # Submit job
            nuke.executeInMainThread(lambda: self.status_text.setValue("Submitting job to server..."))
            nuke.executeInMainThread(lambda: self.progress_bar.setValue(40))
            
            job_id = self.client.submit_job(frames, settings)
            self.current_job = job_id

            # Connect WebSocket for progress updates
            def on_ws_message(ws, message):
                try:
                    data = json.loads(message)
                    progress = data.get('progress', 0)
                    status = data.get('status', '')
                    stage = data.get('stage', '')

                    # Map progress to 40-95 range (saving 5% for final steps)
                    mapped_progress = 40 + int((progress / 100) * 55)
                    nuke.executeInMainThread(lambda p=mapped_progress: self.progress_bar.setValue(p))
                    nuke.executeInMainThread(lambda s=status, st=stage: 
                        self.status_text.setValue(f"Status: {s}\nStage: {st}"))
                except:
                    pass

            self.client.connect_websocket(job_id, on_ws_message)

            # Poll for completion
            while True:
                status = self.client.get_job_status(job_id)

                if status['status'] == 'completed':
                    nuke.executeInMainThread(lambda: self.status_text.setValue("Processing completed! Loading results..."))
                    nuke.executeInMainThread(lambda: self.progress_bar.setValue(95))
                    self._handle_result(job_id)
                    break
                elif status['status'] == 'failed':
                    error = status.get('error', 'Unknown error')
                    nuke.executeInMainThread(lambda e=error: self.status_text.setValue(f"Failed: {e}"))
                    nuke.executeInMainThread(lambda e=error: nuke.message(f"Processing failed: {e}"))
                    break

                time.sleep(2)

        except Exception as e:
            error_msg = str(e)
            nuke.executeInMainThread(lambda: self.status_text.setValue(f"Error: {error_msg}"))
            nuke.executeInMainThread(lambda: nuke.message(f"Error: {error_msg}"))
        finally:
            # Clean up
            if hasattr(self.client, 'disconnect_websocket'):
                self.client.disconnect_websocket()
            nuke.executeInMainThread(lambda: self.process_btn.setEnabled(True))
            nuke.executeInMainThread(lambda: self.cancel_btn.setEnabled(False))

    def _extract_frame(self, node, frame_num) -> bytes:
        """Extract frame data from node"""
        try:
            # Create temp file
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"stereo_frame_{frame_num}_{os.getpid()}.jpg")

            # Write frame
            write_node = nuke.createNode("Write", inpanel=False)
            write_node.setInput(0, node)
            write_node['file'].setValue(temp_file)
            write_node['file_type'].setValue('jpg')
            write_node['_jpeg_quality'].setValue(0.95)

            # Execute write
            nuke.execute(write_node, frame_num, frame_num)

            # Read file data
            with open(temp_file, 'rb') as f:
                data = f.read()

            # Clean up
            nuke.delete(write_node)
            if os.path.exists(temp_file):
                os.remove(temp_file)

            return data

        except Exception as e:
            print(f"Failed to extract frame {frame_num}: {e}")
            return None

    def _handle_result(self, job_id: str):
        """Handle job result"""
        try:
            # Get result
            result = self.client.get_job_result(job_id)
            
            if self.test_mode.value():
                # In test mode, create a simple test setup
                nuke.executeInMainThread(lambda: self.progress_bar.setValue(100))
                nuke.executeInMainThread(lambda: self.status_text.setValue("Test completed! Mock stereo data generated."))
                nuke.executeInMainThread(lambda: nuke.message(
                    f"Stereo generation test complete!\n\n"
                    f"Job ID: {job_id}\n"
                    f"Output Format: {self.output_format.value()}\n"
                    f"Max Disparity: {self.max_disp.value()}\n"
                    f"Tile Count: {self.tile_num.value()}\n\n"
                    f"In real mode, this would create Read nodes with the stereo result."
                ))
                return

            # Create Read node for result (real mode)
            stereo_frames = result.get('stereo_frames', [])
            if stereo_frames:
                # Save frames to temp directory
                temp_dir = os.path.join(tempfile.gettempdir(), f"stereo_{job_id}")
                os.makedirs(temp_dir, exist_ok=True)

                for i, frame_b64 in enumerate(stereo_frames):
                    frame_data = base64.b64decode(frame_b64)
                    frame_path = os.path.join(temp_dir, f"stereo_{i:04d}.jpg")
                    with open(frame_path, 'wb') as f:
                        f.write(frame_data)

                # Create Read node
                read_node = nuke.createNode("Read")
                read_node['file'].setValue(os.path.join(temp_dir, "stereo_####.jpg"))
                read_node['first'].setValue(0)
                read_node['last'].setValue(len(stereo_frames) - 1)
                read_node['label'].setValue(f"StereoCrafter Result\n{self.output_format.value()}")

                nuke.executeInMainThread(lambda: self.progress_bar.setValue(100))
                nuke.executeInMainThread(lambda: self.status_text.setValue("Results loaded successfully!"))
                nuke.executeInMainThread(lambda rn=read_node: 
                    nuke.message(f"Stereo generation complete!\nResult loaded as Read node: {rn.name()}"))

        except Exception as e:
            error_msg = str(e)
            nuke.executeInMainThread(lambda: nuke.message(f"Failed to load result: {error_msg}"))

    def _cancel_processing(self):
        """Cancel current processing"""
        if self.current_job and self.client:
            try:
                if hasattr(self.client, 'server_url') and REQUESTS_AVAILABLE:
                    response = requests.delete(f"{self.client.server_url}/api/v1/jobs/{self.current_job}")
                self.status_text.setValue("Job cancelled")
                if hasattr(self.client, 'disconnect_websocket'):
                    self.client.disconnect_websocket()
            except:
                pass
        
        self.status_text.setValue("Processing cancelled")
        self.progress_bar.setValue(0)
        self.process_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

def show_stereocrafter_panel():
    """Show the StereoCrafter panel"""
    try:
        panel = StereoCrafterPanel()
        panel.showModalDialog()
    except Exception as e:
        nuke.message(f"Error opening StereoCrafter panel: {str(e)}")
        print(f"StereoCrafter error: {e}")

# Register menu
def register_stereocrafter():
    """Register StereoCrafter in Nuke menu"""
    try:
        # Create menu in multiple locations for easy access
        toolbar = nuke.menu("Nodes")
        
        # Add to main menu
        if not toolbar.findItem("StereoCrafter"):
            stereo_menu = toolbar.addMenu("StereoCrafter", icon="Viewer.png")
            stereo_menu.addCommand("Generate Stereo...", show_stereocrafter_panel)
            stereo_menu.addCommand("About", lambda: nuke.message(
                "StereoCrafter v1.0.0\n\n"
                "AI-powered stereo generation for Nuke\n"
                "Convert 2D images/videos to stereo 3D\n\n"
                "Features:\n"
                "• Multiple output formats\n"
                "• Real-time progress tracking\n"
                "• Offline testing mode\n"
                "• Batch processing support"
            ))
        
        # Add to main Nuke menu bar
        main_menu = nuke.menu("Nuke")
        if not main_menu.findItem("StereoCrafter"):
            main_menu.addCommand("StereoCrafter", show_stereocrafter_panel)
        
        print("StereoCrafter plugin registered successfully")
        
    except Exception as e:
        print(f"Failed to register StereoCrafter: {e}")

# Auto-register on import
if __name__ != "__main__":
    register_stereocrafter()