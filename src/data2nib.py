import os
import json
import glob
import numpy as np
import tifffile
import nibabel as nib
from tqdm import tqdm

# ============================================================
# CẤU HÌNH (SỬA ĐƯỜNG DẪN TẠI ĐÂY)
# ============================================================
CONFIG = {
    "src_img": "data/train_images", 
    "src_lbl": "data/train_labels",
    "nnunet_raw": "nnUNet_raw",
    "dataset_id": 501,
    "dataset_name": "Vesuvius320"
}

def save_nifti(arr, path):
    # Tiff (D, H, W) -> NIfTI (W, H, D)
    arr = arr.transpose(2, 1, 0)
    nib.save(nib.Nifti1Image(arr, np.eye(4)), path)

def main():
    folder_name = f"Dataset{CONFIG['dataset_id']}_{CONFIG['dataset_name']}"
    out_dir = os.path.join(CONFIG['nnunet_raw'], folder_name)
    
    img_tr = os.path.join(out_dir, "imagesTr")
    lbl_tr = os.path.join(out_dir, "labelsTr")
    os.makedirs(img_tr, exist_ok=True)
    os.makedirs(lbl_tr, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(CONFIG['src_img'], "*.tif")))
    lbls = sorted(glob.glob(os.path.join(CONFIG['src_lbl'], "*.tif")))

    print(f"--> Start converting {len(imgs)} files for nnU-Net V2...")

    # 1. CONVERT DATA (Batch Processing)
    for i, (imp, lbp) in tqdm(enumerate(zip(imgs, lbls)), total=len(imgs)):
        case_id = f"Vesuvius_{i:03d}"
        
        # Load & Cast -> RAM usage thấp do load tuần tự
        img = tifffile.imread(imp).astype(np.float32)
        lbl = tifffile.imread(lbp).astype(np.uint8)

        # Save NIfTI
        save_nifti(img, os.path.join(img_tr, f"{case_id}_0000.nii.gz"))
        save_nifti(lbl, os.path.join(lbl_tr, f"{case_id}.nii.gz"))

    # 2. CREATE DATASET.JSON (Chuẩn nnU-Net V2)
    # V2 dùng "channel_names" thay vì "modality" cũ
    json_info = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "ink": 1
        },
        "numTraining": len(imgs),
        "file_ending": ".nii.gz",
        "name": CONFIG['dataset_name'],
        "reference": "Vesuvius Challenge",
        "release": "1.0",
        "description": "Vesuvius 3D Segmentation",
    }

    with open(os.path.join(out_dir, "dataset.json"), 'w') as f:
        json.dump(json_info, f, indent=4)

    print("\n--> Done. Dataset ready for nnU-Net V2.")

if __name__ == "__main__":
    main()