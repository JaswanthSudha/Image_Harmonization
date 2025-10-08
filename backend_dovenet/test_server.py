import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import base64
import tempfile
from typing import Optional
from PIL import Image
import shutil

app = FastAPI()

class HarmonizeRequest(BaseModel):
    composite_image: str  # Base64 encoded composite image
    mask_image: Optional[str] = None  # Base64 encoded mask image
    output_path: str     # Where client wants the result saved
    use_patch: bool = False
    image_format: str = "exr"  # Format of the images being sent

@app.post("/harmonize")
def harmonize_test(req: HarmonizeRequest):
    """
    Test harmonization endpoint that processes images without DoveNet
    This verifies the base64 pipeline works before fixing NumPy issues
    """
    print(f"Received test harmonization request:")
    print(f"  Composite image: {len(req.composite_image)} characters")
    print(f"  Mask image: {'Present' if req.mask_image else 'None'}")
    print(f"  Output path: {req.output_path}")
    print(f"  Use patch: {req.use_patch}")
    print(f"  Image format: {req.image_format}")
    if not req.mask_image:
        return {"status": "error", "message": "Mask image is required for harmonization"}
    try:
        # Step 1: Decode base64 images and save to temporary files
        temp_dir = tempfile.mkdtemp(prefix="dovenet_test_")
        # Decode composite image
        composite_data = base64.b64decode(req.composite_image)
        composite_temp_path = os.path.join(temp_dir, f"composite.{req.image_format}")
        with open(composite_temp_path, 'wb') as f:
            f.write(composite_data)
        # Decode mask image
        mask_data = base64.b64decode(req.mask_image)
        mask_temp_path = os.path.join(temp_dir, f"mask.{req.image_format}")
        with open(mask_temp_path, 'wb') as f:
            f.write(mask_data)
        print(f"Successfully decoded images to: {temp_dir}")
        print(f"Composite size: {len(composite_data)} bytes")
        print(f"Mask size: {len(mask_data)} bytes")
        # Step 2: Simple processing - just copy composite as "harmonized" result
        # In real implementation, this would be DoveNet processing
        result_path = os.path.join(temp_dir, f"harmonized.{req.image_format}")
        shutil.copy2(composite_temp_path, result_path)
        # Step 3: Encode result as base64
        with open(result_path, 'rb') as f:
            result_data = f.read()
            result_b64 = base64.b64encode(result_data).decode('utf-8')
        # Step 4: Save to requested output path
        shutil.copy2(result_path, req.output_path)
        # Step 5: Cleanup
        shutil.rmtree(temp_dir)
        print(f"Test harmonization completed successfully")
        return {
            "status": "ok",
            "output_path": req.output_path,
            "result_image": result_b64,
            "message": "Test harmonization completed (just copied composite as result)"
        }
    except Exception as e:
        print(f"Error during test harmonization: {str(e)}")
        try:
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir)
        except:
            pass
        
        return {
            "status": "error",
            "message": f"Test harmonization failed: {str(e)}"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5001)