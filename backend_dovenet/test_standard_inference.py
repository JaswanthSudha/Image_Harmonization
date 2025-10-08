#!/usr/bin/env python3
"""
Test script to verify standard inference method works with EXR files
"""

import requests
import base64
import json

def test_standard_inference():
    # Read test EXR files (from previous successful request)
    composite_path = r"C:\Users\JaswanthSudha\.nuke\backend_dovenet\temp\20241211_180432\composite.exr"
    mask_path = r"C:\Users\JaswanthSudha\.nuke\backend_dovenet\temp\20241211_180432\mask.exr"
    
    try:
        # Read and encode files
        with open(composite_path, 'rb') as f:
            composite_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        with open(mask_path, 'rb') as f:
            mask_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Prepare request data
        request_data = {
            "composite_image": composite_b64,
            "mask_image": mask_b64,
            "use_patch": False  # Use standard inference
        }
        
        print("Testing standard inference method...")
        print(f"Composite size: {len(composite_b64)} chars")
        print(f"Mask size: {len(mask_b64)} chars")
        
        # Send request
        response = requests.post(
            "http://127.0.0.1:5001/harmonize",
            json=request_data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Standard inference successful!")
            print(f"Status: {result['status']}")
            print(f"Result image size: {len(result['result_image'])} chars")
            print(f"Processing time: {result.get('processing_time', 'unknown')}")
            return True
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except FileNotFoundError as e:
        print(f"❌ Test files not found: {e}")
        print("Please run patch-based inference first to create test files")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_standard_inference()