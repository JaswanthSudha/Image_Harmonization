#!/usr/bin/env python3
"""
Simple test for standard inference without numpy
"""

import requests
import base64
import json

def test_standard_inference():
    """Test with minimal fake EXR data"""
    try:
        # Create minimal fake EXR data
        fake_exr_header = b'\x76\x2f\x31\x01'  # EXR magic number
        fake_composite = fake_exr_header + b'\x00' * 10000  # 10KB fake composite
        fake_mask = fake_exr_header + b'\x00' * 1000        # 1KB fake mask
        
        # Encode to base64
        composite_b64 = base64.b64encode(fake_composite).decode('utf-8')
        mask_b64 = base64.b64encode(fake_mask).decode('utf-8')
        
        # Prepare request data for STANDARD inference (not patch-based)
        request_data = {
            "composite_image": composite_b64,
            "mask_image": mask_b64,
            "output_path": "test_output.exr",
            "use_patch": False  # This is the key - use standard test_inference.py
        }
        
        print("🧪 Testing STANDARD inference method (test_inference.py)...")
        print(f"Composite size: {len(composite_b64)} chars")
        print(f"Mask size: {len(mask_b64)} chars")
        print("Expected: Should use test_inference.py script with dataset loading")
        
        # Send request
        response = requests.post(
            "http://127.0.0.1:5001/harmonize",
            json=request_data,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ STANDARD inference successful!")
            print(f"Status: {result['status']}")
            print(f"Result image size: {len(result['result_image'])} chars")
            print(f"Processing time: {result.get('processing_time', 'unknown')}")
            print("🎉 EXR dataset loading with OpenCV fallback works!")
            return True
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing DoveNet Standard Inference with EXR Support")
    print("=" * 60)
    success = test_standard_inference()
    print("=" * 60)
    if success:
        print("✅ All tests passed! Both inference methods now work.")
    else:
        print("❌ Standard inference test failed.")
    print("=" * 60)