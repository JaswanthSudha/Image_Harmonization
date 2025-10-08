import os.path
import torch
import torchvision.transforms.functional as tf
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class InferenceDataset(BaseDataset):
    """Dataset for inference-only mode where real images are not available."""
    
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
        parser.set_defaults(max_dataset_size=float("inf"), new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        This dataset only requires composite images and masks - no real images needed.
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
            print('loading test file for inference')
            self.trainfile = os.path.join(opt.dataset_root, 'IHD_test.txt')
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        self.image_paths.append(os.path.join(opt.dataset_root,line.rstrip()))
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. Only composite image and mask are loaded.
        """
        path = self.image_paths[index]
        name_parts=os.path.basename(path).split('_')
        mask_path = path.replace('composite_images','masks')
        mask_path_png = mask_path.replace(('_'+name_parts[-1]),'.png')
        mask_path_exr = mask_path.replace(('_'+name_parts[-1]),'.exr')
        
        # Check which mask file actually exists
        if os.path.exists(mask_path_png):
            mask_path = mask_path_png
        elif os.path.exists(mask_path_exr):
            mask_path = mask_path_exr
        else:
            # Default to PNG for error reporting
            mask_path = mask_path_png

        comp = None
        try:
            comp = Image.open(path).convert('RGB')
        except Exception as e:
            # Try OpenCV for EXR files
            if path.lower().endswith('.exr'):
                try:
                    import cv2
                    import numpy as np
                    exr_img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                    if exr_img is None:
                        raise Exception("Could not load EXR file with OpenCV")
                    
                    exr_img = cv2.cvtColor(exr_img, cv2.COLOR_BGR2RGB)
                    exr_img = np.clip(exr_img * 255, 0, 255).astype(np.uint8)
                    comp = Image.fromarray(exr_img)
                except ImportError:
                    raise Exception("OpenCV not available for EXR loading")
                except Exception as cv_error:
                    raise Exception(f"Could not load EXR file: {cv_error}")
            else:
                raise e
        
        mask = None
        try:
            mask = Image.open(mask_path).convert('1')
        except Exception as e:
            # Try OpenCV for EXR mask files
            if mask_path.lower().endswith('.exr'):
                try:
                    import cv2
                    import numpy as np
                    exr_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
                    if exr_mask is None:
                        raise Exception("Could not load EXR mask file with OpenCV")
                    
                    exr_mask = np.clip(exr_mask * 255, 0, 255).astype(np.uint8)
                    mask = Image.fromarray(exr_mask).convert('1')
                except ImportError:
                    raise Exception("OpenCV not available for EXR mask loading")
                except Exception as cv_error:
                    raise Exception(f"Could not load EXR mask file: {cv_error}")
            else:
                raise e

        comp = tf.resize(comp, [256, 256])
        mask = tf.resize(mask, [256, 256])

        #apply the same transform to composite image
        comp = self.transform(comp)
        mask = tf.to_tensor(mask)
        
        #concatenate the composite and mask as the input of generator
        inputs=torch.cat([comp,mask],0)

        # Return only composite and inputs - no real image needed
        return {'inputs': inputs, 'comp': comp, 'img_path':path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
