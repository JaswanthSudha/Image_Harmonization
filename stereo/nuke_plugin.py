"""
StereoCrafter Nuke Plugin
Simple client for the StereoCrafter server
Place this file in ~/.nuke/ or your Nuke plugin directory
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

# Add requests for API communication
try:
    import requests
except ImportError:
    nuke.message("Please install requests: pip install requests")
    sys.exit(1)

try:
    import websocket
except ImportError:
    nuke.message("Please install websocket-client: pip install websocket-client")
    sys.exit(1)

# Configuration
DEFAULT_SERVER_URL = os.getenv("STEREOCRAFTER_SERVER", "http://localhost:8000")
DEFAULT_WS_URL = DEFAULT_SERVER_URL.replace("http://", "ws://").replace("https://", "wss://")

class StereoCrafterClient:
    """Client for communicating with StereoCrafter server"""

    def __init__(self, server_url: str = DEFAULT_SERVER_URL):
        self.server_url = server_url.rstrip('/')
        self.ws_url = DEFAULT_WS_URL
        self.current_job_id = None
        self.ws_connection = None
        self.progress_callback = None

    def check_health(self) -> bool:
        """Check if server is healthy"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def submit_job(self, frames: List[bytes], settings: Dict[str, Any]) -> str:
        """Submit a stereo generation job"""
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
        response = requests.get(
            f"{self.server_url}/api/v1/jobs/{job_id}/status",
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    def get_job_result(self, job_id: str, format: str = "sbs") -> Dict[str, Any]:
        """Get job result"""
        response = requests.get(
            f"{self.server_url}/api/v1/jobs/{job_id}/result",
            params={"format": format},
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def connect_websocket(self, job_id: str, on_message=None):
        """Connect to WebSocket for real-time updates"""
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

        self.divider1 = nuke.Text_Knob("divider1", "")
        self.addKnob(self.divider1)

        # Input settings
        self.source_node = nuke.Enumeration_Knob("source_node", "Source Node", self._get_read_nodes())
        self.addKnob(self.source_node)

        self.frame_range = nuke.String_Knob("frame_range", "Frame Range", f"{nuke.root().firstFrame()}-{nuke.root().lastFrame()}")
        self.addKnob(self.frame_range)

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
        self.status_text = nuke.Multiline_Eval_String_Knob("status", "Status", "Ready")
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

    def _check_server(self):
        """Check server connection"""
        try:
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
        if not self.client:
            self.client = StereoCrafterClient(self.server_url.value())

        # Get source node
        source_name = self.source_node.value()
        if source_name == "None":
            nuke.message("Please select a source node")
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

        # Disable controls during processing
        self.process_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status_text.setValue("Extracting frames...")

        # Start processing in separate thread
        self.processing_thread = threading.Thread(
            target=self._process_frames,
            args=(source_node, start, end)
        )
        self.processing_thread.start()

    def _process_frames(self, source_node, start_frame, end_frame):
        """Process frames in background thread"""
        try:
            # Extract frames
            frames = []
            frame_count = end_frame - start_frame + 1

            for frame_num in range(start_frame, end_frame + 1):
                nuke.executeInMainThread(lambda: self.status_text.setValue(f"Extracting frame {frame_num}/{end_frame}"))

                # Get frame data
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
            job_id = self.client.submit_job(frames, settings)
            self.current_job = job_id

            # Connect WebSocket for progress updates
            def on_ws_message(ws, message):
                try:
                    data = json.loads(message)
                    progress = data.get('progress', 0)
                    status = data.get('status', '')
                    stage = data.get('stage', '')

                    nuke.executeInMainThread(lambda: self.progress_bar.setValue(int(progress)))
                    nuke.executeInMainThread(lambda: self.status_text.setValue(f"Status: {status}\nStage: {stage}"))
                except:
                    pass

            self.client.connect_websocket(job_id, on_ws_message)

            # Poll for completion
            while True:
                status = self.client.get_job_status(job_id)

                if status['status'] == 'completed':
                    nuke.executeInMainThread(lambda: self.status_text.setValue("Processing completed!"))
                    nuke.executeInMainThread(lambda: self.progress_bar.setValue(100))
                    self._handle_result(job_id)
                    break
                elif status['status'] == 'failed':
                    error = status.get('error', 'Unknown error')
                    nuke.executeInMainThread(lambda: self.status_text.setValue(f"Failed: {error}"))
                    nuke.executeInMainThread(lambda: nuke.message(f"Processing failed: {error}"))
                    break

                import time
                time.sleep(2)

        except Exception as e:
            nuke.executeInMainThread(lambda: self.status_text.setValue(f"Error: {str(e)}"))
            nuke.executeInMainThread(lambda: nuke.message(f"Error: {str(e)}"))
        finally:
            # Clean up
            self.client.disconnect_websocket()
            nuke.executeInMainThread(lambda: self.process_btn.setEnabled(True))
            nuke.executeInMainThread(lambda: self.cancel_btn.setEnabled(False))

    def _extract_frame(self, node, frame_num) -> bytes:
        """Extract frame data from node"""
        try:
            # Create temp file
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"stereo_frame_{frame_num}.jpg")

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

            # Create Read node for result
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

                nuke.message(f"Stereo generation complete!\nResult loaded as Read node: {read_node.name()}")

        except Exception as e:
            nuke.message(f"Failed to load result: {str(e)}")

    def _cancel_processing(self):
        """Cancel current processing"""
        if self.current_job and self.client:
            try:
                response = requests.delete(f"{self.client.server_url}/api/v1/jobs/{self.current_job}")
                self.status_text.setValue("Job cancelled")
                self.client.disconnect_websocket()
            except:
                pass

def show_stereocrafter_panel():
    """Show the StereoCrafter panel"""
    panel = StereoCrafterPanel()
    panel.showModalDialog()

# Register menu
def register_stereocrafter():
    """Register StereoCrafter in Nuke menu"""
    nuke.menu("Nodes").addCommand("StereoCrafter/Generate Stereo", show_stereocrafter_panel)

# Auto-register on import
if __name__ != "__main__":
    register_stereocrafter()