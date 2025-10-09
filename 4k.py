import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as tf
from options.test_options import TestOptions
from models import create_model
import torchvision.transforms as transforms
from util import util
import time

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
        
    def process_image(self, comp_path, mask_path):
        """
        Process a high-resolution image using patch-based inference
        
        Args:
            comp_path: Path to composite image
            mask_path: Path to mask image
            
        Returns:
            harmonized_image: PIL Image of the result
        """
        # Load images
        comp = Image.open(comp_path).convert('RGB')
        mask = Image.open(mask_path).convert('1')
        
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
        
        # Convert to PIL Image
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
    
    # Initialize patch-based processor
    processor = PatchBasedInference(opt, patch_size=512, overlap=64)
    
    # Process your image
    comp_path = os.path.join(opt.dataset_root, "composite_images", "sample_1_2.jpg")
    mask_path = os.path.join(opt.dataset_root, "masks", "sample_1.png")
    
    print(f"Starting patch-based 4K inference...")
    start_time = time.time()
    
    result = processor.process_image(comp_path, mask_path)
    
    inference_time = time.time() - start_time
    print(f"Total inference time: {inference_time:.2f}s")
    
    # Save result
    output_dir = os.path.join(opt.results_dir, opt.name, "patch_based_4k")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sample_1_2_4k_harmonized.jpg")
    result.save(output_path, quality=95)
    
    print(f"4K harmonized image saved to: {output_path}")
    print(f"Output size: {result.size[0]}x{result.size[1]}")

if __name__ == '__main__':
    main()
