import os
import torch
import numpy as np
import time
from PIL import Image
import torchvision.transforms.functional as tf
from options.test_options import TestOptions
from models import create_model
import torchvision.transforms as transforms
from util import util


class PatchBasedInference:
    def __init__(self, opt, patch_size=512, overlap=64):
        """
        Initialize patch-based inference for high-resolution images
        
        Args:
            opt: Options object
            patch_size: Size of each patch (default: 512)
            overlap: Overlap between patches (default: 64)
        """
        self.opt = opt
        self.patch_size = patch_size
        self.overlap = overlap
        self.model = create_model(opt)
        self.model.setup(opt)
        
        # Transforms for patches
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def load_image_with_exr_support(self, path, mode='RGB'):
        """
        Load image with EXR support fallback
        
        Args:
            path: Path to image
            mode: PIL image mode ('RGB' or 'L')
            
        Returns:
            PIL Image
        """
        try:
            if mode == 'RGB':
                return Image.open(path).convert('RGB')
            else:
                return Image.open(path).convert('L')
        except Exception as e:
            print(f"Warning: Could not load image as standard format: {e}")
            if path.lower().endswith('.exr'):
                try:
                    import cv2
                    print(f"Attempting to load EXR with OpenCV...")
                    
                    if mode == 'RGB':
                        exr_img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                        if exr_img is None:
                            raise Exception("Could not load EXR file with OpenCV")
                        exr_img = cv2.cvtColor(exr_img, cv2.COLOR_BGR2RGB)
                        exr_img = np.clip(exr_img * 255, 0, 255).astype(np.uint8)
                        return Image.fromarray(exr_img)
                    else:
                        exr_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
                        if exr_img is None:
                            raise Exception("Could not load EXR file with OpenCV")
                        exr_img = np.clip(exr_img * 255, 0, 255).astype(np.uint8)
                        return Image.fromarray(exr_img).convert('L')
                        
                except ImportError:
                    raise Exception("OpenCV not available for EXR loading")
                except Exception as cv_error:
                    raise Exception(f"Could not load EXR file: {cv_error}")
            else:
                raise e
        
    def process_image(self, comp_path, mask_path):
        """
        Process a high-resolution image using patch-based inference
        
        Args:
            comp_path: Path to composite image
            mask_path: Path to mask image
            
        Returns:
            harmonized_image: PIL Image of the result
        """
        # Load images with EXR support
        comp = self.load_image_with_exr_support(comp_path, 'RGB')
        mask = self.load_image_with_exr_support(mask_path, 'L').convert('1')
        
        # Ensure both images have the same size before processing
        if comp.size != mask.size:
            print(f"Resizing mask from {mask.size} to {comp.size} to match composite")
            # Use older PIL compatibility for resampling
            try:
                # Try newer PIL syntax first
                mask = mask.resize(comp.size, Image.Resampling.NEAREST)  # preserves binary mask edges
            except AttributeError:
                # Fall back to older PIL syntax
                mask = mask.resize(comp.size, Image.NEAREST)  # preserves binary mask edges
        
        # Get original size
        orig_w, orig_h = comp.size
        print(f"Processing {orig_w}x{orig_h} image using {self.patch_size}x{self.patch_size} patches")
        
        # Convert to tensors for processing
        comp_tensor = tf.to_tensor(comp)
        mask_tensor = tf.to_tensor(mask)
        
        # Initialize output tensor
        output_tensor = torch.zeros_like(comp_tensor)
        weight_map = torch.zeros((1, orig_h, orig_w))
        
        # Calculate patch positions
        stride = self.patch_size - self.overlap
        patches_h = (orig_h - self.overlap + stride - 1) // stride
        patches_w = (orig_w - self.overlap + stride - 1) // stride
        
        total_patches = patches_h * patches_w
        processed_patches = 0
        
        print(f"Total patches to process: {total_patches}")
        
        for i in range(patches_h):
            for j in range(patches_w):
                # Calculate patch coordinates
                start_h = i * stride
                start_w = j * stride
                end_h = min(start_h + self.patch_size, orig_h)
                end_w = min(start_w + self.patch_size, orig_w)
                
                # Adjust start if patch would be too small
                if end_h - start_h < self.patch_size:
                    start_h = max(0, end_h - self.patch_size)
                if end_w - start_w < self.patch_size:
                    start_w = max(0, end_w - self.patch_size)
                    
                # Extract patches
                comp_patch = comp_tensor[:, start_h:start_h+self.patch_size, start_w:start_w+self.patch_size]
                mask_patch = mask_tensor[:, start_h:start_h+self.patch_size, start_w:start_w+self.patch_size]
                
                # Ensure patch is exactly patch_size
                if comp_patch.shape[1] != self.patch_size or comp_patch.shape[2] != self.patch_size:
                    comp_patch = tf.resize(comp_patch, [self.patch_size, self.patch_size])
                    mask_patch = tf.resize(mask_patch, [self.patch_size, self.patch_size])
                
                # Normalize patch
                comp_patch = tf.normalize(comp_patch, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                
                # Final safety check before concatenation
                # assert comp_patch.shape[1] == self.patch_size and comp_patch.shape[2] == self.patch_size, f"comp_patch wrong size: {comp_patch.shape}"
                # assert mask_patch.shape[1] == self.patch_size and mask_patch.shape[2] == self.patch_size, f"mask_patch wrong size: {mask_patch.shape}"
                
                # Prepare input for model
                inputs = torch.cat([comp_patch, mask_patch], 0).unsqueeze(0)
                
                # Create data dictionary
                data = {'inputs': inputs, 'comp': comp_patch.unsqueeze(0)}
                
                # Run inference on patch
                self.model.set_input(data)
                self.model.test()
                
                # Get output
                visuals = self.model.get_current_visuals()
                output_patch = visuals['output']
                
                # Convert back to image space
                output_patch_np = util.tensor2im(output_patch)
                output_patch_tensor = tf.to_tensor(output_patch_np)
                
                # Resize back if needed and place in output tensor
                if output_patch_tensor.shape[1] != end_h - start_h or output_patch_tensor.shape[2] != end_w - start_w:
                    output_patch_tensor = tf.resize(output_patch_tensor, [end_h - start_h, end_w - start_w])
                
                # Apply feathering for smooth blending
                weight = torch.ones((1, end_h - start_h, end_w - start_w))
                if self.overlap > 0:
                    # Create feathering mask
                    fade_size = self.overlap // 2
                    if start_h > 0:  # Not top edge
                        weight[:, :fade_size, :] = torch.linspace(0, 1, fade_size).unsqueeze(-1)
                    if start_w > 0:  # Not left edge  
                        weight[:, :, :fade_size] = torch.linspace(0, 1, fade_size).unsqueeze(0)
                    if end_h < orig_h:  # Not bottom edge
                        weight[:, -fade_size:, :] = torch.linspace(1, 0, fade_size).unsqueeze(-1) 
                    if end_w < orig_w:  # Not right edge
                        weight[:, :, -fade_size:] = torch.linspace(1, 0, fade_size).unsqueeze(0)
                
                # Accumulate output with weights
                output_tensor[:, start_h:end_h, start_w:end_w] += output_patch_tensor * weight
                weight_map[:, start_h:end_h, start_w:end_w] += weight
                
                processed_patches += 1
                if processed_patches % 10 == 0:
                    print(f"Processed {processed_patches}/{total_patches} patches...")
        
        # Normalize by weights
        output_tensor = output_tensor / torch.clamp(weight_map, min=1e-8)
        
        # Convert to PIL Image - use standard 8-bit conversion for all formats
        output_np = (output_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        output_image = Image.fromarray(output_np)
        
        return output_image


def main():
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.dataset_mode = 'inference'  # Use our inference dataset
    
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
    
    # Initialize patch-based processor
    processor = PatchBasedInference(opt, patch_size=512, overlap=64)
    
    print(f"Starting patch-based 4K inference...")
    start_time = time.time()
    
    result = processor.process_image(comp_path, mask_path)
    
    inference_time = time.time() - start_time
    print(f"Total inference time: {inference_time:.2f}s")
    
    # Save result
    output_dir = os.path.join(opt.results_dir, opt.name, "patch_based_4k")
    os.makedirs(output_dir, exist_ok=True)
    
    # Choose output format based on input
    if comp_path.lower().endswith('.exr'):
        output_filename = f"{os.path.basename(mask_base)}_4k_harmonized.exr"
        output_path = os.path.join(output_dir, output_filename)
        
        # Try to save as EXR using OpenCV for better HDR support
        try:
            import cv2
            # Convert PIL image to float range for EXR
            result_np = np.array(result).astype(np.float32) / 255.0  # Convert uint8 to float range
            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr)
            print(f"4K harmonized EXR saved using OpenCV: {output_path}")
        except Exception as e:
            print(f"OpenCV EXR save failed: {e}, falling back to standard JPG")
            # Fall back to JPG if EXR saving fails
            output_filename = f"{os.path.basename(mask_base)}_4k_harmonized.jpg"
            output_path = os.path.join(output_dir, output_filename)
            result.save(output_path, quality=95)
            print(f"4K harmonized image saved as JPG: {output_path}")
    else:
        output_filename = f"{os.path.basename(mask_base)}_4k_harmonized.jpg"
        output_path = os.path.join(output_dir, output_filename)
        result.save(output_path, quality=95)
        print(f"4K harmonized image saved to: {output_path}")
    
    print(f"Output size: {result.size[0]}x{result.size[1]}")


if __name__ == '__main__':
    main()
