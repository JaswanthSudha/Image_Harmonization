from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import torch

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html,util

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    
    # Print input resolution info
    print(f"High-Resolution Inference Mode")
    print(f"Max size limit: {getattr(opt, 'max_size', 'unlimited')}")
    print(f"Preserve aspect ratio: {getattr(opt, 'preserve_aspect_ratio', False)}")
    print(f"GPU Memory: {'CPU' if len(opt.gpu_ids) == 0 or opt.gpu_ids[0] == -1 else f'GPU {opt.gpu_ids}'}")
    
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    if opt.eval:
        model.eval()
        
    total_inference_time = 0
    
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
            
        # Get original size if available
        original_size = data.get('original_size', None)
        if original_size:
            print(f"Processing image {i+1}: Original size {original_size[0]}x{original_size[1]}")
        
        # Check input tensor size
        input_tensor = data['inputs']
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Monitor memory usage for large images
        if len(opt.gpu_ids) > 0 and opt.gpu_ids[0] != -1:
            torch.cuda.empty_cache()  # Clear GPU cache
            if torch.cuda.is_available():
                print(f"GPU memory before inference: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        start_time = time.time()
        
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        print(f"Inference time: {inference_time:.2f}s")
        
        if len(opt.gpu_ids) > 0 and opt.gpu_ids[0] != -1 and torch.cuda.is_available():
            print(f"GPU memory after inference: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        visuals = model.get_current_visuals()  # get image results
        img_path = str(data['img_path'])
        raw_name = img_path.replace(('[\''),'')
        raw_name = raw_name.replace(('.jpg\']'),'.jpg')
        raw_name = raw_name.split('/')[-1]
        image_name = '%s' % raw_name
        save_path = os.path.join(web_dir,'images/',image_name)
        
        # Save all outputs
        for label, im_data in visuals.items():
            if label=='output':
                output = util.tensor2im(im_data)
                
                # If we have original size info and it's different, we could resize back
                # For now, save at the processed resolution
                output_path = save_path
                if original_size and hasattr(opt, 'preserve_aspect_ratio') and opt.preserve_aspect_ratio:
                    # Optionally resize back to original dimensions
                    # output_pil = Image.fromarray(output)
                    # output_pil = output_pil.resize(original_size, Image.BICUBIC)
                    # output = np.array(output_pil)
                    output_path = save_path.replace('.jpg', '_highres.jpg')
                
                util.save_image(output, output_path, aspect_ratio=opt.aspect_ratio)
                print(f'{image_name} | High-res harmonized image saved successfully to {output_path}')
                print(f'Output resolution: {output.shape[1]}x{output.shape[0]}')
                
            if label=='comp':
                comp = util.tensor2im(im_data)
                comp_path = save_path.replace('.jpg', '_composite.jpg')
                util.save_image(comp, comp_path, aspect_ratio=opt.aspect_ratio)

    print(f"\nTotal inference time: {total_inference_time:.2f}s")
    print(f"Average per image: {total_inference_time/max(1, i+1):.2f}s")
    webpage.save()  # save the HTML
    print(f"High-resolution inference complete! Results saved to: {web_dir}")
