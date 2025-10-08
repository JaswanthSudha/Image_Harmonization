import nuke
import requests
import base64
import tempfile
import os

def encode_image_to_base64(image_path):
    """
    Encode image file to base64 string
    """
    try:
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def run_dovenet_via_server(input_path, output_path, mask_path=None, use_patch=False):
    url = "http://127.0.0.1:5001/harmonize"
    
    # Encode images as base64
    composite_b64 = encode_image_to_base64(input_path)
    if not composite_b64:
        nuke.message("Failed to encode composite image")
        return None
    
    mask_b64 = None
    if mask_path:
        mask_b64 = encode_image_to_base64(mask_path)
        if not mask_b64:
            nuke.message("Failed to encode mask image")
            return None
    
    payload = {
        "composite_image": composite_b64,
        "mask_image": mask_b64,
        "output_path": output_path,
        "use_patch": use_patch,
        "image_format": "exr"  # Tell server what format we're sending
    }

    try:
        resp = requests.post(url, json=payload).json()
        print("Server response:", resp)
    except Exception as e:
        nuke.message("Server error: %s" % e)
        return None

    if resp.get("status") == "ok":
        # Server returns base64 encoded result - decode and save it
        if resp.get("result_image"):
            try:
                # Decode base64 result and save to output path
                result_data = base64.b64decode(resp["result_image"])
                with open(output_path, 'wb') as f:
                    f.write(result_data)
                
                return nuke.nodes.Read(file=output_path)
            except Exception as e:
                nuke.message("Failed to decode result: %s" % str(e))
                return None
        else:
            nuke.message("No result image received from server")
            return None
    else:
        nuke.message("Harmonization failed: %s" % resp.get("message"))
        return None
