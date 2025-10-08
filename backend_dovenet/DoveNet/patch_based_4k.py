import os
import torch
import numpy as np
import time
from PIL import Image
import torchvision.transforms.functional as tf
from options.test_options import TestOptions
from models import create_model


def main():
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.dataset_mode = 'inference'
    
    model = create_model(opt)
    model.setup(opt)
    
    if opt.eval:
        model.eval()
    
    test_file_path = os.path.join(opt.dataset_root, "IHD_test.txt")
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Test file not found: {test_file_path}")
    
    with open(test_file_path, 'r') as f:
        composite_filename = f.read().strip()
    
    print(f"Reading from test file: {composite_filename}")
    
    base_name = os.path.splitext(composite_filename)[0]
    if base_name.endswith('_2'):
        mask_base = base_name[:-2]
    else:
        mask_base = base_name
    
    comp_path = os.path.join(opt.dataset_root, composite_filename)
    
    mask_extensions = ['.png', '.exr']
    mask_path = None
    
    for ext in mask_extensions:
        potential_mask = os.path.join(opt.dataset_root, "masks", f"{os.path.basename(mask_base)}{ext}")
        if os.path.exists(potential_mask):
            mask_path = potential_mask
            break
    
    if mask_path is None:
        raise FileNotFoundError(f"Mask file not found for base name: {mask_base}")
    
    print(f"Using composite: {comp_path}")
    print(f"Using mask: {mask_path}")
    
    try:
        comp = Image.open(comp_path).convert('RGB')
        print(f"Loaded composite image: {comp.size}")
    except Exception as e:
        print(f"Warning: Could not load composite as standard image: {e}")
        if comp_path.lower().endswith('.exr'):
            try:
                import cv2
                print("Attempting to load EXR with OpenCV...")
                
                exr_img = cv2.imread(comp_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if exr_img is None:
                    raise Exception("Could not load EXR file with OpenCV")
                
                exr_img = cv2.cvtColor(exr_img, cv2.COLOR_BGR2RGB)
                exr_img = np.clip(exr_img * 255, 0, 255).astype(np.uint8)
                comp = Image.fromarray(exr_img)
                print(f"Successfully loaded EXR composite: {comp.size}")
                
            except ImportError:
                raise Exception("OpenCV not available for EXR loading")
            except Exception as cv_error:
                raise Exception(f"Could not load EXR file: {cv_error}")
        else:
            raise e
    
    try:
        mask = Image.open(mask_path).convert('L')
        print(f"Loaded mask image: {mask.size}")
    except Exception as e:
        print(f"Warning: Could not load mask as standard image: {e}")
        if mask_path.lower().endswith('.exr'):
            try:
                import cv2
                print("Attempting to load EXR mask with OpenCV...")
                
                exr_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
                if exr_mask is None:
                    raise Exception("Could not load EXR mask file with OpenCV")
                
                exr_mask = np.clip(exr_mask * 255, 0, 255).astype(np.uint8)
                mask = Image.fromarray(exr_mask).convert('L')
                print(f"Successfully loaded EXR mask: {mask.size}")
                
            except ImportError:
                raise Exception("OpenCV not available for EXR mask loading")
            except Exception as cv_error:
                raise Exception(f"Could not load EXR mask file: {cv_error}")
        else:
            raise e
    
    target_size = (256, 256)
    # Use older PIL compatibility for resampling
    try:
        # Try newer PIL syntax first
        comp_resized = comp.resize(target_size, Image.Resampling.LANCZOS)
        mask_resized = mask.resize(target_size, Image.Resampling.LANCZOS)
    except AttributeError:
        # Fall back to older PIL syntax
        comp_resized = comp.resize(target_size, Image.LANCZOS)
        mask_resized = mask.resize(target_size, Image.LANCZOS)
    
    comp_tensor = tf.to_tensor(comp_resized).unsqueeze(0)
    mask_tensor = tf.to_tensor(mask_resized).unsqueeze(0)
    
    input_tensor = torch.cat([comp_tensor, mask_tensor], dim=1)
    
    print(f"Starting harmonization inference...")
    start_time = time.time()
    
    with torch.no_grad():
        # Use the correct input format expected by DoveNet model
        model.set_input({
            'comp': comp_tensor,
            'mask': mask_tensor,
            'inputs': input_tensor,
            'A': input_tensor,
            'B': comp_tensor
        })
        model.test()
        
        # Get output using the correct method
        visuals = model.get_current_visuals()
        if 'output' in visuals:
            result_tensor = visuals['output']
            print(f"Successfully retrieved output tensor: {result_tensor.shape}")
        else:
            print(f"Available visual keys: {list(visuals.keys())}")
            # Try other common keys
            if 'fake_B' in visuals:
                result_tensor = visuals['fake_B']
            elif 'harmonized' in visuals:
                result_tensor = visuals['harmonized']
            else:
                raise Exception(f"Could not find output in visuals. Available keys: {list(visuals.keys())}")
        
        # Convert tensor to image using util function if available
        try:
            from util import util
            result_np = util.tensor2im(result_tensor)
            result_img = Image.fromarray(result_np)
        except ImportError:
            # Fallback to manual conversion
            result = result_tensor.squeeze(0)
            if result.dim() == 3:
                result_np = result.permute(1, 2, 0).numpy()
            else:
                result_np = result.numpy()
            result_np = np.clip(result_np * 255, 0, 255).astype(np.uint8)
            result_img = Image.fromarray(result_np)
    
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.2f}s")
    
    orig_size = comp.size
    # Use older PIL compatibility for resampling
    try:
        # Try newer PIL syntax first
        result_final = result_img.resize(orig_size, Image.Resampling.LANCZOS)
    except AttributeError:
        # Fall back to older PIL syntax
        result_final = result_img.resize(orig_size, Image.LANCZOS)
    
    output_dir = os.path.join(opt.results_dir, opt.name, "patch_based_4k")
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"{os.path.basename(mask_base)}_4k_harmonized.jpg"
    output_path = os.path.join(output_dir, output_filename)
    result_final.save(output_path, quality=95)
    
    print(f"Harmonized image saved to: {output_path}")
    print(f"Output size: {result_final.size[0]}x{result_final.size[1]}")


if __name__ == '__main__':
    main()
