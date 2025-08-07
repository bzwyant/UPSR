import torch
import os
import cv2
import random
import time
import numpy as np
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data
from basicsr.data.data_util import scandir
from basicsr.data.transforms import augment

@DATASET_REGISTRY.register()
class RealGalaxyMNISTDataset(data.Dataset):
    """Dataset used for GalaxyMNIST model:
    GalaxyMNIST: Test dataset for using UPSR on galaxy images.
    
    It loads the gt images and then augments them.
    
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(RealGalaxyMNISTDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']

        # file client (running on slurm so disk io backend)
        self.io_backend_opt['type'] = 'disk'
        self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        self.kernel_range = [x for x in range(3, opt['blur_kernel_size'], 2)]  # kernel size range for the first degradation
        self.kernel_range2 = [x for x in range(3, opt['blur_kernel_size2'], 2)]  # kernel size range for the second degradation

        # pulse tensor for no blurry effect 
        self.pulse_tensor = torch.zeros(opt['blur_kernel_size'], opt['blur_kernel_size']).float()
        self.pulse_tensor[opt['blur_kernel_size']//2, opt['blur_kernel_size']//2] = 1

        self.rescale_gt = opt['rescale_gt']

    
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load the image
        gt_path = self.paths[index]

        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warning(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes, float32=True)

        # Augmentation for training: flip, rotation
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        h, w = img_gt.shape[:2]

        crop_pad_size = self.opt['crop_pad_size']

        if h > crop_pad_size or w > crop_pad_size:
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size]

        if self.rescale_gt and crop_pad_size != self.opt['gt_size']:
            img_gt = cv2.resize(img_gt, dsize=(self.opt['gt_size'],)*2, interpolation=cv2.INTER_AREA)
        

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # sinc filter for kernels ranging from 3 to 15
            if kernel_size < 9:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-np.pi, np.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (self.blur_kernel_size - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range2)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-np.pi, np.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (self.blur_kernel_size2 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))


        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range2)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=self.blur_kernel_size2)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {
            'gt': img_gt,
            'kernel1': kernel,
            'kernel2': kernel2,
            'sinc_kernel': sinc_kernel,
            'gt_path': gt_path
        }
        return return_d
    
    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.paths)
       