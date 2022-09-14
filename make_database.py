import os
from os.path import join, exists
import numpy as np
from rt_utils import RTStructBuilder

def create_ct_pet_database(src: str, dest: str):
    
    if not exists(dest):
        os.makedirs(dest)

    # Loop over STS folders in src directory
    for folder in os.listdir(src):

        # Patient dir
        patient_dir = join(src, folder)

        print(f"Extracting data from: {patient_dir}")

        # Skip if directory already parsed
        if exists(join(dest, folder)):
            print(f"Skipping {patient_dir}, directory exists!")
            continue

        # Make folder for patient
        os.makedirs(join(dest, folder))

        # Take dir with more files
        dir1, dir2 = os.listdir(patient_dir)

        if len(os.listdir(join(patient_dir, dir1))) > len(os.listdir(join(patient_dir, dir1))):
            petct_dir = dir1
        else:
            petct_dir = dir2
    
        petct_dir = join(patient_dir, petct_dir)

        # Find RT struct for PET scan
        for name in os.listdir(petct_dir):
            if "RTstructPET" in name.split("-"):
                pet_rt_path = join(petct_dir, name)
                pet_rt_path = join(pet_rt_path, os.listdir(pet_rt_path)[0])
                break

        # Find RT struct for CT scan
        for name in os.listdir(petct_dir):
            if "RTstructCT" in name.split("-"):
                ct_rt_path = join(petct_dir, name)
                ct_rt_path = join(ct_rt_path, os.listdir(ct_rt_path)[0])
                break

        # Find PET and CT dcm files
        dir_names = os.listdir(petct_dir)
        num_of_files = [len(os.listdir(join(petct_dir, x))) for x in dir_names]
        
        ct_dir = dir_names[np.argmax(num_of_files)]
        num_of_files[np.argmax(num_of_files)] = 0
        pet_dir = dir_names[np.argmax(num_of_files)]

        if "StandardFull" in pet_dir.split("-") or pet_dir.find("CT") == -1:
            ct_dir, pet_dir = pet_dir, ct_dir

        ct_dir = join(petct_dir, ct_dir)
        pet_dir = join(petct_dir, pet_dir)

        # Create dest dirs
        ct_dest = join(dest, folder, "ct")
        pet_dest = join(dest, folder, "pet")
        os.makedirs(ct_dest)
        os.makedirs(pet_dest)

        print(f"\t{ct_dir}")
        print(f"\t{pet_dir}")
        # Load RT structs
        ct_scan = RTStructBuilder.create_from(
            dicom_series_path=ct_dir,
            rt_struct_path=ct_rt_path
        )
        ct_mask = ct_scan.get_roi_mask_by_name(ct_scan.get_roi_names()[0])
        ct_imgs = ct_scan.series_data


        pet_scan = RTStructBuilder.create_from(
            dicom_series_path=pet_dir,
            rt_struct_path=pet_rt_path
        )
        pet_imgs = pet_scan.series_data

        # Find maximal number of cancer pixels per scan
        max_pixels = max([np.sum(x) for x in ct_mask.T])
        threshold = .1 * max_pixels

        # Save scans with 10% of maximal number of cancer pixels
        for idx in range(ct_scan.shape[-1]):
            if np.sum(ct_scan[:, :, idx]) > threshold:
                
                with open(join(ct_dest, f"{idx:03}.npy"), "wb") as f:
                    np.save(f, ct_imgs[idx].pixel_data)

                with open(join(pet_dest, f"{idx:03}.npy"), "wb") as f:
                    np.save(f, pet_imgs[idx].pixel_data)


    print("Finished!")
    
if __name__ == "__main__":
    create_ct_pet_database("database_tryout", "dest_tryout")