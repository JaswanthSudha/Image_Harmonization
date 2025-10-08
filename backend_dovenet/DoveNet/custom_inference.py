#!/usr/bin/env python3
"""
Custom inference script for DoveNet with your own composite image + mask data
"""

import os
import sys
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from options.test_options import TestOptions
from models import create_model
from util import util

class CustomDoveNetInference:
    def __init__(self, model_path, device='cpu'):
        """
        Initialize the DoveNet model for inference
        
        Args:
            model_path (str): Path to the pre-trained model
            device (str): Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        
        # Create options for the model
        self.opt = TestOptions().parse([])
        self.opt.model = 'dovenet'
        self.opt.dataset_mode = 'iharmony4'
        self.opt.netG = 's2ad'
        self.opt.is_train = False
        self.opt.norm = 'batch'  # Use batch normalization as mentioned in README
        self.opt.no_flip = True
        self.opt.preprocess = 'none'
        self.opt.load_size = 256
        self.opt.crop_size = 256
        self.opt.serial_batches = True
        self.opt.batch_size = 1
        self.opt.num_threads = 0
        self.opt.display_id = -1
        
        # Create the model
        self.model = create_model(self.opt)
        self.model.setup(self.opt)
        
        # Load the pre-trained model
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model.load_networks(model_path)
        else:
            print(f"Warning: Model file not found at {model_path}")
            print("Please download the pre-trained model and place it at the specified path")
        
        # Set to evaluation mode
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    
    def preprocess_image(self, composite_path, mask_path):
        """
        Preprocess composite image and mask for inference
        
        Args:
            composite_path (str): Path to composite image
            mask_path (str): Path to mask image
            
        Returns:
            dict: Preprocessed data ready for model input
        """
        # Load images
        comp_img = Image.open(composite_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('L')  # Convert to grayscale
        
        # Apply transforms
        comp_tensor = self.transform(comp_img)
        mask_tensor = self.mask_transform(mask_img)
        
        # Ensure mask is binary (0 or 1)
        mask_tensor = (mask_tensor > 0.5).float()
        
        # Concatenate composite image and mask as input
        inputs = torch.cat([comp_tensor, mask_tensor], dim=0)
        
        # Add batch dimension
        inputs = inputs.unsqueeze(0)
        comp_tensor = comp_tensor.unsqueeze(0)
        
        return {
            'inputs': inputs.to(self.device),
            'comp': comp_tensor.to(self.device),
            'img_path': composite_path
        }
    
    def harmonize_image(self, composite_path, mask_path, output_path=None):
        """
        Harmonize a composite image using the provided mask
        
        Args:
            composite_path (str): Path to composite image
            mask_path (str): Path to mask image
            output_path (str, optional): Path to save harmonized image
            
        Returns:
            PIL.Image: Harmonized image
        """
        # Preprocess input
        data = self.preprocess_image(composite_path, mask_path)
        
        # Run inference
        with torch.no_grad():
            self.model.set_input(data)
            self.model.test()
            
            # Get the harmonized output
            visuals = self.model.get_current_visuals()
            harmonized_tensor = visuals['output']
            
            # Convert tensor back to PIL Image
            harmonized_img = util.tensor2im(harmonized_tensor)
            
            # Save output if path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                harmonized_img.save(output_path)
                print(f"Harmonized image saved to: {output_path}")
            
            return harmonized_img
    
    def harmonize_batch(self, composite_dir, mask_dir, output_dir, file_extensions=('.jpg', '.jpeg', '.png')):
        """
        Harmonize a batch of composite images with their corresponding masks
        
        Args:
            composite_dir (str): Directory containing composite images
            mask_dir (str): Directory containing mask images
            output_dir (str): Directory to save harmonized images
            file_extensions (tuple): Supported image file extensions
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of composite images
        composite_files = []
        for ext in file_extensions:
            composite_files.extend([f for f in os.listdir(composite_dir) if f.lower().endswith(ext)])
        
        print(f"Found {len(composite_files)} composite images to process")
        
        for comp_file in composite_files:
            # Construct paths
            comp_path = os.path.join(composite_dir, comp_file)
            
            # Try to find corresponding mask file
            mask_file = None
            for ext in file_extensions:
                # Try different naming conventions
                possible_names = [
                    comp_file.replace(ext, '.png'),
                    comp_file.replace(ext, '.jpg'),
                    comp_file.replace(ext, '.jpeg'),
                    'mask_' + comp_file,
                    comp_file.replace(ext, '_mask.png'),
                    comp_file.replace(ext, '_mask.jpg')
                ]
                
                for name in possible_names:
                    mask_path = os.path.join(mask_dir, name)
                    if os.path.exists(mask_path):
                        mask_file = name
                        break
                
                if mask_file:
                    break
            
            if not mask_file:
                print(f"Warning: No mask found for {comp_file}, skipping...")
                continue
            
            mask_path = os.path.join(mask_dir, mask_file)
            output_path = os.path.join(output_dir, f"harmonized_{comp_file}")
            
            try:
                print(f"Processing: {comp_file} with mask: {mask_file}")
                self.harmonize_image(comp_path, mask_path, output_path)
            except Exception as e:
                print(f"Error processing {comp_file}: {str(e)}")
                continue

def main():
    parser = argparse.ArgumentParser(description='DoveNet Custom Inference')
    parser.add_argument('--model_path', type=str, 
                       default='checkpoints/experiment_name_pretrain/latest_net_G.pth',
                       help='Path to pre-trained model')
    parser.add_argument('--composite', type=str, required=True,
                       help='Path to composite image')
    parser.add_argument('--mask', type=str, required=True,
                       help='Path to mask image')
    parser.add_argument('--output', type=str, default='harmonized_output.jpg',
                       help='Path to save harmonized image')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Device to use for inference')
    parser.add_argument('--batch_mode', action='store_true',
                       help='Process a batch of images')
    parser.add_argument('--composite_dir', type=str,
                       help='Directory containing composite images (for batch mode)')
    parser.add_argument('--mask_dir', type=str,
                       help='Directory containing mask images (for batch mode)')
    parser.add_argument('--output_dir', type=str, default='harmonized_outputs',
                       help='Directory to save harmonized images (for batch mode)')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    # Initialize the inference class
    inferencer = CustomDoveNetInference(args.model_path, args.device)
    
    if args.batch_mode:
        if not args.composite_dir or not args.mask_dir:
            print("Error: --composite_dir and --mask_dir are required for batch mode")
            return
        
        print("Starting batch processing...")
        inferencer.harmonize_batch(args.composite_dir, args.mask_dir, args.output_dir)
        print("Batch processing completed!")
    else:
        if not os.path.exists(args.composite):
            print(f"Error: Composite image not found at {args.composite}")
            return
        
        if not os.path.exists(args.mask):
            print(f"Error: Mask image not found at {args.mask}")
            return
        
        print("Starting single image harmonization...")
        harmonized_img = inferencer.harmonize_image(args.composite, args.mask, args.output)
        print("Harmonization completed!")

if __name__ == '__main__':
    main()
