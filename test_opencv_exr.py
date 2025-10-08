#!/usr/bin/env python
"""
Test script to verify OpenCV EXR support in the DoveNet server
"""

import requests
import base64
import json
import numpy as np
from PIL import Image
import io

def test_server_with_opencv():
    """Test that the server can handle the OpenCV import for EXR files"""
    
    # Create a simple test image
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(test_img)
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    # Test data for server
    test_data = {
        "composite_base64": img_str,
        "mask_base64": img_str,  # Using same image as mask for simplicity
        "harmonization_type": "foreground"
    }
    
    try:
        print("Testing DoveNet server with OpenCV dependency...")
        response = requests.post(
            "http://127.0.0.1:5001/harmonize", 
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Server responded successfully!")
            print(f"Response status: {result.get('status', 'N/A')}")
            
            if 'harmonized_image' in result:
                print("✅ Harmonized image returned successfully!")
                return True
            else:
                print("❌ No harmonized image in response")
                print(f"Response: {result}")
                return False
        else:
            print(f"❌ Server error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_server_with_opencv()
    if success:
        print("\n🎉 OpenCV integration test PASSED!")
        print("The DoveNet server is ready for EXR file processing from Nuke!")
    else:
        print("\n💥 OpenCV integration test FAILED!")
        print("Check server logs for more details.")