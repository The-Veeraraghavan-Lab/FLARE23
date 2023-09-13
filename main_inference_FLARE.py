import argparse
import os

 

import nibabel as nib
import numpy as np
import torch

#from flare_swinunetr import FlareSwinUNETR
from monai import data, transforms
from monai.data import load_decathlon_datalist, decollate_batch

from utils_slidingwindow import sliding_window_inference

import smit_mini, configs_smit

from tqdm import tqdm
 
def list_of_strings(arg):
    return arg.split(',')
 

# def main():
    

if __name__ == "__main__":
    print("Working now...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.AddChanneld(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=-250, a_max=250, b_min=0, b_max=1, clip=True),
            transforms.CropForegroundd(keys=["image"], source_key="image"),
            transforms.SpatialPadd(keys=["image"], spatial_size=[96, 96, 96]),
            transforms.ToTensord(keys=["image"]),
        ]
    )

    post_transforms = transforms.Compose([
        transforms.EnsureTyped(keys="pred"),
        transforms.Invertd(
            keys="pred",
            transform=test_transform,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
        )
        ])

    post_transforms_softmax = transforms.AsDiscreted(keys="pred", argmax=True)
    
    test_files = load_decathlon_datalist("files2run.json", True, "validation")
    test_ds = data.Dataset(test_files, transform=test_transform)
    test_loader = data.DataLoader(test_ds, batch_size=1, shuffle=False)
    
    print("Finished loading and process transforming files...")
    
    
    config = configs_smit.get_SMIT_small_128_bias_True()
    model = smit_mini.SMIT_3D_Seg_mini(config,out_channels=15)
    #Fine-tuning weights would go here - in the next two lines
    #model_dict = torch.load('weights/run1_ce.pt', map_location="cpu")["state_dict"]
    #model.load_state_dict(model_dict)
    model.eval()
    model.to(device)
    
    print("Finished loading model dictionary...")
    
    output_directory = 'outputs'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
            
    with torch.no_grad():    
        for i, batch in enumerate(tqdm(test_loader)):
            
            val_inputs = batch["image"].to(device)
            
            #print("Val inputs shape:")
            #print(val_inputs.shape)
            
            img_name = batch["image_meta_dict"]["filename_or_obj"][0]
            img_name_tosave = img_name.replace("inputs/","").replace("_0000","")
            #print(img_name_tosave)
            
            batch["pred"]= sliding_window_inference(
                inputs = val_inputs,
                roi_size = (96, 96, 96),
                sw_batch_size = 1,
                predictor = model,
                overlap=0.5,
                mode="gaussian",
                progress=False)
            
            batch = [post_transforms(i) for i in decollate_batch(batch)]
            
            seg_batch = post_transforms_softmax(batch[0])
            seg_ori_size=seg_batch['pred'].cpu().numpy().astype(np.uint8)
            seg_ori_size=np.squeeze(seg_ori_size)
            val_labels_ori_save = nib.Nifti1Image(seg_ori_size,np.eye(4))
            nib.save(val_labels_ori_save, os.path.join(output_directory, img_name_tosave))
            
            
            
            
            
            
            
