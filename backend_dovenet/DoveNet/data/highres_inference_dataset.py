import os.path
import torch
import torchvision.transforms.functional as tf
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class HighResInferenceDataset(BaseDataset):
    """Dataset for high-resolution inference (including 4K) where real images are not available."""
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--is_train', type=bool, default=True, help='whether in the training phase')
        parser.add_argument('--preserve_aspect_ratio', action='store_true', help='preserve aspect ratio during inference')
        parser.add_argument('--max_size', type=int, default=4096, help='maximum dimension for high-res inference')
        parser.set_defaults(max_dataset_size=float("inf"), new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        This dataset supports high-resolution inference without forced downsizing.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.image_paths = []
        self.isTrain = opt.isTrain
        
        if opt.isTrain==True:
            print('loading training file: ')
            self.trainfile = os.path.join(opt.dataset_root, 'IHD_train.txt')
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        self.image_paths.append(os.path.join(opt.dataset_root,line.rstrip()))
        elif opt.isTrain==False:
            print('loading test file for high-resolution inference')
            self.trainfile = os.path.join(opt.dataset_root, 'IHD_test.txt')
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        self.image_paths.append(os.path.join(opt.dataset_root,line.rstrip()))
        
        # For high-res inference, we may not want standard transforms
        self.use_original_size = hasattr(opt, 'preserve_aspect_ratio') and opt.preserve_aspect_ratio
        if not self.use_original_size:
            self.transform = get_transform(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. Preserves original resolution for high-res inference.
        """
        path = self.image_paths[index]
        name_parts=os.path.basename(path).split('_')
        mask_path = path.replace('composite_images','masks')
        mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')

        comp = Image.open(path).convert('RGB')
        mask = Image.open(mask_path).convert('1')

        # Store original dimensions for later restoration
        original_size = comp.size  # (width, height)
        
        if self.use_original_size:
            # For high-res inference, ensure dimensions are divisible by 4 (network requirement)
            w, h = comp.size
            new_w = ((w + 3) // 4) * 4
            new_h = ((h + 3) // 4) * 4
            
            if new_w != w or new_h != h:
                comp = tf.resize(comp, [new_h, new_w], interpolation=Image.BICUBIC)
                mask = tf.resize(mask, [new_h, new_w], interpolation=Image.NEAREST)
            
            # Manual transforms for high-res (no random transforms)
            comp = tf.to_tensor(comp)
            comp = tf.normalize(comp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            mask = tf.to_tensor(mask)
        else:
            # Use standard pipeline with configurable size
            if hasattr(self, 'transform'):
                comp = self.transform(comp) 
                mask = tf.to_tensor(mask)
            else:
                # Fallback to 256x256 if no transform
                comp = tf.resize(comp, [256, 256])
                mask = tf.resize(mask, [256, 256])
                comp = tf.to_tensor(comp)
                comp = tf.normalize(comp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                mask = tf.to_tensor(mask)
        
        #concatenate the composite and mask as the input of generator
        inputs=torch.cat([comp,mask],0)

        # Return data with original size info for potential restoration
        return {
            'inputs': inputs, 
            'comp': comp, 
            'img_path': path,
            'original_size': original_size
        }

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
