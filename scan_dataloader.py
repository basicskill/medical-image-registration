import os
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage import median_filter


class CTPET_Dataset(Dataset):
    """Class for holding data of PET and CT scans of body parts.

    Args:
        Dataset (Dataset): Pytorch base dataset class.
    """

    def __init__(self, data_location: str) -> None:
        super(CTPET_Dataset, self).__init__()

        self.ct_paths = []
        self.pet_paths = []

        for sts in os.listdir(data_location):
            
            ct_dir = os.path.join(data_location, sts, "ct")
            pet_dir = os.path.join(data_location, sts, "pet")

            for file_name in os.listdir(ct_dir):

                self.ct_paths.append(os.path.join(ct_dir, file_name))
                self.pet_paths.append(os.path.join(pet_dir, file_name))

        self.len = len(self.ct_paths)


    def __getitem__(self, index) -> dict:
        # Load and normalize CT image
        ct_img = torch.from_numpy(np.load(self.ct_paths[index])).to(torch.float32)
        # ct_img[ct_img < 0] = 0
        ct_img -= torch.min(ct_img)
        ct_img /= torch.max(ct_img)
        
        # Load and normalize PET image
        pet_img = zoom(np.load(self.pet_paths[index]), 4, order=0)
        # pet_img[pet_img < 0] = 0
        pet_img -= np.min(pet_img)
        pet_img /= np.max(pet_img)

        ct_img *= 512
        pet_img *= 512

        pet_img[pet_img < 100] = 0
        pet_img = median_filter(pet_img, 20)
        pet_img = torch.from_numpy(pet_img).to(torch.float32)

        # Filter out PET image
        
        # Stack images for input of autoencoder
        stacked = torch.stack([ct_img, pet_img])

        return {
            "CT": ct_img,
            "PET": pet_img,
            "stacked": stacked
        }

    def __len__(self):
        """Get dataset length."""
        return self.len