import os
from torch.utils.data import Dataset
import numpy as np

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
        ct_img = torch.from_numpy(np.load(self.ct_paths[index]))
        pet_img = torch.from_numpy(np.load(self.pet_paths[index])  

        return {
            "CT": ct_img,
            "PET": pet_img,
        }

    def __len__(self):
        """Get dataset length."""
        return self.len