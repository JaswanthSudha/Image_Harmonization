#!/usr/bin/env python3
"""
Test standard inference with sample data
"""

import requests
import base64
import json
import tempfile
import numpy as np
import os
try:
    import cv2
except ImportError:
    print("OpenCV not available - creating minimal test")
    cv2 = None

def create_test_exr_files():
    """Create sample EXR files for testing"""
    temp_dir = tempfile.mkdtemp()
    print(f"Creating test files in: {temp_dir}")
    
    if cv2 is not None:
        # Create a simple test image (1920x1080)
        test_img = np.random.rand(1080, 1920, 3).astype(np.float32)
        test_mask = np.random.rand(1080, 1920).astype(np.float32)
        
        composite_path = os.path.join(temp_dir, "composite.exr")
        mask_path = os.path.join(temp_dir, "mask.exr")
        
        cv2.imwrite(composite_path, test_img)
        cv2.imwrite(mask_path, test_mask)
        
        return composite_path, mask_path
    else:
        # Fallback: use any existing EXR files
        print("Creating mock EXR files (empty)")
        composite_path = os.path.join(temp_dir, "composite.exr")
        mask_path = os.path.join(temp_dir, "mask.exr")
        
        # Create minimal EXR-like files (just binary data)
        with open(composite_path, 'wb') as f:
            f.write(b'\x76\x2f\x31\x01' + b'\x00' * 1000)  # EXR magic + minimal data
        with open(mask_path, 'wb') as f:
            f.write(b'\x76\x2f\x31\x01' + b'\x00' * 100)
            
        return composite_path, mask_path

def test_standard_inference():
    try:
        composite_path, mask_path = create_test_exr_files()
        
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
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_standard_inference()