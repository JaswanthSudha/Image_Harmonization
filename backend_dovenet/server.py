import os
import subprocess
import tempfile
import base64
import time
import shutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from PIL import Image
import io
import uuid

app = FastAPI()

# Configuration
PYTHON_EXE = r"C:\Users\JaswanthSudha\anaconda3\envs\pytorch12\python.exe"
DOVENET_SCRIPT = r"C:\Users\JaswanthSudha\.nuke\backend_dovenet\DoveNet\patch_based_4k.py"

class HarmonizationRequest(BaseModel):
    composite_image: str  # base64 encoded
    mask_image: str       # base64 encoded
    output_path: str
    use_patch: bool = True
    image_format: str = "exr"

class DoveNetDataPreparer:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.custom_data_dir = os.path.join(base_dir, 'custom_data')
        
    def prepare_data(self, composite_path, mask_path):
        """Prepare data in DoveNet expected format"""
        try:
            print(f"DoveNetDataPreparer: Starting data preparation")
            print(f"Base dir: {self.base_dir}")
            print(f"Composite path: {composite_path}")
            print(f"Mask path: {mask_path}")
            
            # Check if input files exist
            if not os.path.exists(composite_path):
                raise Exception(f"Composite file does not exist: {composite_path}")
            if not os.path.exists(mask_path):
                raise Exception(f"Mask file does not exist: {mask_path}")
            
            print(f"Input files verified to exist")
            
            # Create directory structure
            composite_dir = os.path.join(self.custom_data_dir, 'composite_images')
            mask_dir = os.path.join(self.custom_data_dir, 'masks')
            
            os.makedirs(composite_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            
            print(f"Created directories:")
            print(f"  Composite dir: {composite_dir}")
            print(f"  Mask dir: {mask_dir}")
            
            # Generate unique sample name
            sample_name = f"sample_{uuid.uuid4().hex[:8]}"
            print(f"Using sample name: {sample_name}")
            
            # Convert and copy composite image (DoveNet expects JPG for composite)
            composite_dest = os.path.join(composite_dir, f"{sample_name}_2.jpg")
            actual_composite_path = self._convert_to_jpg(composite_path, composite_dest)
            print(f"Composite saved to: {actual_composite_path}")
            
            # Convert and copy mask (DoveNet expects PNG for masks)
            mask_dest = os.path.join(mask_dir, f"{sample_name}.png")
            actual_mask_path = self._convert_to_png(mask_path, mask_dest)
            print(f"Mask saved to: {actual_mask_path}")
            
            # Verify files were created (check actual paths)
            if not os.path.exists(actual_composite_path):
                raise Exception(f"Failed to create composite file: {actual_composite_path}")
            if not os.path.exists(actual_mask_path):
                raise Exception(f"Failed to create mask file: {actual_mask_path}")
            
            print(f"Files verified to exist after conversion")
            
            # Create IHD_test.txt with actual filename
            actual_composite_name = os.path.basename(actual_composite_path)
            test_file_path = os.path.join(self.custom_data_dir, 'IHD_test.txt')
            with open(test_file_path, 'w') as f:
                # Write relative path from custom_data directory
                relative_path = f"composite_images/{actual_composite_name}"
                f.write(f"{relative_path}\n")
            
            print(f"Created test file: {test_file_path}")
            
            return self.custom_data_dir
            
        except Exception as e:
            print(f"DoveNetDataPreparer error: {e}")
            raise
    
    def _convert_to_jpg(self, input_path, output_path):
        """Convert image to JPG format"""
        try:
            print(f"Converting {input_path} to JPG: {output_path}")
            
            # Handle EXR files (they need special handling)
            if input_path.lower().endswith('.exr'):
                try:
                    img = Image.open(input_path)
                    # Convert to RGB if needed
                    if img.mode in ('RGBA', 'LA'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'RGBA':
                            background.paste(img, mask=img.split()[-1])
                        else:
                            background.paste(img)
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img.save(output_path, 'JPEG', quality=95)
                    print(f"EXR converted successfully")
                    return output_path
                    
                except Exception as exr_error:
                    print(f"PIL EXR conversion failed: {exr_error}")
                    # Fallback: copy as EXR and update the expected path
                    fallback_path = output_path.replace('.jpg', '.exr')
                    shutil.copy2(input_path, fallback_path)
                    print(f"Copied EXR as-is to: {fallback_path}")
                    # Return the actual path created, not the original expected path
                    return fallback_path
            else:
                img = Image.open(input_path)
                
                # Convert RGBA to RGB with white background
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        background.paste(img, mask=img.split()[-1])
                    else:
                        background.paste(img)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img.save(output_path, 'JPEG', quality=95)
                print(f"Image converted to JPG successfully")
                return output_path
            
        except Exception as e:
            print(f"Error converting to JPG: {e}")
            try:
                shutil.copy2(input_path, output_path)
                print(f"Fallback: copied file directly")
                return output_path
            except Exception as copy_error:
                raise Exception(f"Failed to convert image: {e}")
    
    def _convert_to_png(self, input_path, output_path):
        """Convert image to PNG format"""
        try:
            print(f"Converting {input_path} to PNG: {output_path}")
            
            if input_path.lower().endswith('.exr'):
                try:
                    img = Image.open(input_path)
                    if img.mode not in ('L', 'LA'):
                        img = img.convert('L')
                    img.save(output_path, 'PNG')
                    print(f"EXR mask converted successfully")
                    return output_path
                    
                except Exception as exr_error:
                    print(f"PIL EXR mask conversion failed: {exr_error}")
                    fallback_path = output_path.replace('.png', '.exr')
                    shutil.copy2(input_path, fallback_path)
                    print(f"Copied mask EXR as-is to: {fallback_path}")
                    # Return the actual path created
                    return fallback_path
            else:
                img = Image.open(input_path)
                
                if img.mode != 'L':
                    img = img.convert('L')
                
                img.save(output_path, 'PNG')
                print(f"Mask converted to PNG successfully")
                return output_path
            
        except Exception as e:
            print(f"Error converting to PNG: {e}")
            try:
                shutil.copy2(input_path, output_path)
                print(f"Fallback: copied mask file directly")
                return output_path
            except Exception as copy_error:
                raise Exception(f"Failed to convert mask: {e}")

def setup_conda_environment():
    """Set up proper conda environment variables for subprocess"""
    
    # Get conda environment paths
    conda_env_path = os.path.dirname(PYTHON_EXE)
    conda_root = os.path.dirname(os.path.dirname(conda_env_path))  # Go up two levels to anaconda3
    
    # Create environment copy
    env = os.environ.copy()
    
    # Set conda-specific environment variables
    env['CONDA_DEFAULT_ENV'] = 'pytorch12'
    env['CONDA_PREFIX'] = conda_env_path
    env['CONDA_PREFIX_1'] = conda_root
    env['CONDA_PYTHON_EXE'] = PYTHON_EXE
    
    # Update PATH to include conda environment paths
    conda_paths = [
        conda_env_path,                           # Main env path
        os.path.join(conda_env_path, 'Scripts'),  # Scripts
        os.path.join(conda_env_path, 'Library', 'bin'),  # Library bin
        os.path.join(conda_env_path, 'Library', 'usr', 'bin'),  # Library usr bin
        os.path.join(conda_env_path, 'Library', 'mingw-w64', 'bin'),  # MinGW
        conda_root,                               # Conda root
        os.path.join(conda_root, 'Scripts'),      # Conda root scripts
    ]
    
    # Prepend conda paths to system PATH
    current_path = env.get('PATH', '')
    env['PATH'] = ';'.join(conda_paths) + ';' + current_path
    
    # Set library paths for DLL loading
    env['CONDA_DLL_SEARCH_MODIFICATION_ENABLE'] = '1'
    
    # Set PYTHONPATH to include conda site-packages
    site_packages = os.path.join(conda_env_path, 'Lib', 'site-packages')
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = site_packages + ';' + env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = site_packages
    
    # NumPy specific environment variables (to help with DLL loading)
    env['MKL_THREADING_LAYER'] = 'GNU'
    env['OMP_NUM_THREADS'] = '1'
    
    return env

def run_dovenet_harmonization(composite_path, mask_path, output_dir, use_patch=True):
    """Run DoveNet harmonization with proper environment setup"""
    
    try:
        print(f"Setting up DoveNet data structure...")
        
        # Prepare data in DoveNet format
        data_prep = DoveNetDataPreparer(output_dir)
        dataset_root = data_prep.prepare_data(composite_path, mask_path)
        
        print(f"Dataset prepared at: {dataset_root}")
        
        # Set up conda environment for subprocess
        env = setup_conda_environment()
        
        # Print environment info for debugging
        print(f"Using Python: {PYTHON_EXE}")
        print(f"CONDA_DEFAULT_ENV: {env.get('CONDA_DEFAULT_ENV')}")
        print(f"CONDA_PREFIX: {env.get('CONDA_PREFIX')}")
        print(f"PATH (first few): {env.get('PATH', '')[:200]}...")
        
        # Build DoveNet command
        script_name = 'patch_based_4k.py' if use_patch else 'test_inference.py'
        dovenet_script = os.path.join(os.path.dirname(DOVENET_SCRIPT), script_name)
        
        cmd = [
            PYTHON_EXE,
            dovenet_script,
            '--dataset_root', dataset_root,
            '--name', 'experiment_name_pretrain',
            '--model', 'dovenet',
            '--dataset_mode', 'inference',
            '--netG', 's2ad',
            '--is_train', '0',
            '--norm', 'batch',
            '--gpu_ids', '-1'
        ]
        
        # Add specific arguments for test_inference.py
        if not use_patch:
            cmd.extend([
                '--no_flip',
                '--preprocess', 'none'
            ])
        
        print(f"Running DoveNet command: {' '.join(cmd)}")
        
        # Run DoveNet with proper environment (Python 3.6 compatible)
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(DOVENET_SCRIPT),  # Run from DoveNet directory
            env=env,                              # Use conda environment
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=300
        )
        
        print(f"DoveNet return code: {result.returncode}")
        if result.stdout:
            print(f"DoveNet stdout: {result.stdout}")
        if result.stderr:
            print(f"DoveNet stderr: {result.stderr}")
        
        if result.returncode != 0:
            raise Exception(f"DoveNet failed with return code {result.returncode}: {result.stderr}")
        
        # Find the result file - different scripts save to different locations
        if use_patch:
            # patch_based_4k.py saves to ./results/experiment_name_pretrain/patch_based_4k/
            result_dir = os.path.join(os.path.dirname(DOVENET_SCRIPT), 'results', 'experiment_name_pretrain', 'patch_based_4k')
        else:
            # test_inference.py saves to ./results/experiment_name_pretrain/test_latest/images/
            result_dir = os.path.join(os.path.dirname(DOVENET_SCRIPT), 'results', 'experiment_name_pretrain', 'test_latest', 'images')
        
        if not os.path.exists(result_dir):
            # Try alternative locations
            alternative_dirs = [
                os.path.join(os.path.dirname(DOVENET_SCRIPT), 'results', 'experiment_name_pretrain', 'patch_based_4k'),
                os.path.join(os.path.dirname(DOVENET_SCRIPT), 'results', 'experiment_name_pretrain', 'test_latest', 'images'),
                os.path.join(dataset_root, 'results', 'experiment_name_pretrain', 'test_latest', 'images')
            ]
            
            for alt_dir in alternative_dirs:
                if os.path.exists(alt_dir):
                    result_dir = alt_dir
                    break
            else:
                raise Exception(f"Result directory not found. Tried: {', '.join(alternative_dirs)}")
        
        print(f"Looking for results in: {result_dir}")
        
        # Look for harmonized result
        result_files = [f for f in os.listdir(result_dir) if 'harmonized' in f.lower()]
        if not result_files:
            # Fallback: look for any image file (including EXR)
            result_files = [f for f in os.listdir(result_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.exr'))]
        
        if not result_files:
            raise Exception(f"No result images found in {result_dir}")
        
        result_file = os.path.join(result_dir, result_files[0])
        print(f"Found result: {result_file}")
        
        return result_file
        
    except subprocess.TimeoutExpired:
        raise Exception("DoveNet processing timed out (5 minutes)")
    except Exception as e:
        print(f"DoveNet error: {e}")
        raise Exception(f"DoveNet processing failed: {str(e)}")
    finally:
        # Cleanup temporary data
        try:
            if 'dataset_root' in locals() and os.path.exists(dataset_root):
                shutil.rmtree(dataset_root)
                print("Cleaned up temporary DoveNet data")
        except:
            pass

def decode_base64_image(base64_string, output_path):
    """Decode base64 string to image file"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Save to file
        with open(output_path, 'wb') as f:
            f.write(image_data)
        
        print(f"Decoded image saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return False

def encode_image_to_base64(image_path):
    """Encode image file to base64 string"""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        base64_string = base64.b64encode(image_data).decode('utf-8')
        return base64_string
        
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None

@app.post("/harmonize")
async def harmonize_image(request: HarmonizationRequest):
    temp_dir = None
    try:
        print("=== Received harmonization request ===")
        
        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp(prefix="dovenet_")
        print(f"Created temp directory: {temp_dir}")
        
        # Decode composite image
        composite_temp_path = os.path.join(temp_dir, "composite.exr")
        print(f"Decoding composite image to: {composite_temp_path}")
        
        if not decode_base64_image(request.composite_image, composite_temp_path):
            raise HTTPException(status_code=400, detail="Failed to decode composite image")
        
        # Verify composite file was created
        if not os.path.exists(composite_temp_path):
            raise HTTPException(status_code=400, detail=f"Composite file was not created: {composite_temp_path}")
        
        composite_size = os.path.getsize(composite_temp_path)
        print(f"Composite file created successfully: {composite_size} bytes")
        
        # Decode mask image
        mask_temp_path = os.path.join(temp_dir, "mask.exr")  # Use EXR for consistency
        print(f"Decoding mask image to: {mask_temp_path}")
        
        if not decode_base64_image(request.mask_image, mask_temp_path):
            raise HTTPException(status_code=400, detail="Failed to decode mask image")
        
        # Verify mask file was created
        if not os.path.exists(mask_temp_path):
            raise HTTPException(status_code=400, detail=f"Mask file was not created: {mask_temp_path}")
        
        mask_size = os.path.getsize(mask_temp_path)
        print(f"Mask file created successfully: {mask_size} bytes")
        
        print(f"=== Starting DoveNet processing ===")
        print(f"Composite: {composite_temp_path}")
        print(f"Mask: {mask_temp_path}")
        
        # Run DoveNet harmonization with proper environment
        result_path = run_dovenet_harmonization(
            composite_temp_path, 
            mask_temp_path, 
            temp_dir, 
            request.use_patch
        )
        
        print(f"=== DoveNet processing completed ===")
        print(f"Result path: {result_path}")
        
        # Verify result file exists
        if not os.path.exists(result_path):
            raise HTTPException(status_code=500, detail=f"Result file not found: {result_path}")
        
        # Encode result as base64
        result_base64 = encode_image_to_base64(result_path)
        if not result_base64:
            raise HTTPException(status_code=500, detail="Failed to encode result image")
        
        # Copy result to requested output path
        try:
            os.makedirs(os.path.dirname(request.output_path), exist_ok=True)
            shutil.copy2(result_path, request.output_path)
            print(f"Result saved to: {request.output_path}")
        except Exception as e:
            print(f"Warning: Could not copy to output path {request.output_path}: {e}")
        
        print(f"=== Harmonization completed successfully ===")
        
        return {
            "status": "ok",
            "result_image": result_base64,
            "output_path": request.output_path,
            "message": "Harmonization completed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"=== Harmonization error: {e} ===")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temp directory: {temp_dir}")
            except Exception as cleanup_error:
                print(f"Warning: Could not cleanup temp directory: {cleanup_error}")

if __name__ == "__main__":
    print("Starting DoveNet Harmonization Server...")
    print(f"Using Python: {PYTHON_EXE}")
    print(f"DoveNet script: {DOVENET_SCRIPT}")
    
    # Test environment setup
    env = setup_conda_environment()
    print(f"Environment configured for: {env.get('CONDA_DEFAULT_ENV')}")
    
    uvicorn.run(app, host="127.0.0.1", port=5001)
