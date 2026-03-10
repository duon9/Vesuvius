import os
import json
import glob
import subprocess
import numpy as np
import tifffile
import nibabel as nib
from tqdm import tqdm


# ===== DEFAULT CONFIG =====
SRC_IMG = "train_images"
SRC_LBL = "train_labels"

DATASET_ID = 501
DATASET_NAME = "Vesuvius320"

NNUNET_RAW = "nnunet/nnUNet_raw"
NNUNET_PREPROCESSED = "nnunet/nnUNet_preprocessed"
# ==========================


def load_settings():
    with open("SETTINGS.json") as f:
        return json.load(f)


def save_nifti(arr, path):
    # TIFF (D,H,W) -> NIfTI (W,H,D)
    arr = arr.transpose(2, 1, 0)
    nib.save(nib.Nifti1Image(arr, np.eye(4)), path)


def build_nnunet_raw(raw_data_dir):

    dataset_folder = f"Dataset{DATASET_ID}_{DATASET_NAME}"
    out_dir = os.path.join(NNUNET_RAW, dataset_folder)

    imagesTr = os.path.join(out_dir, "imagesTr")
    labelsTr = os.path.join(out_dir, "labelsTr")

    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(raw_data_dir, SRC_IMG, "*.tif")))
    lbls = sorted(glob.glob(os.path.join(raw_data_dir, SRC_LBL, "*.tif")))

    print(f"--> Converting {len(imgs)} samples to nnU-Net format")

    for i, (imp, lbp) in tqdm(enumerate(zip(imgs, lbls)), total=len(imgs)):

        case_id = f"{DATASET_NAME}_{i:03d}"

        img = tifffile.imread(imp).astype(np.float32)
        lbl = tifffile.imread(lbp).astype(np.uint8)

        save_nifti(img, os.path.join(imagesTr, f"{case_id}_0000.nii.gz"))
        save_nifti(lbl, os.path.join(labelsTr, f"{case_id}.nii.gz"))

    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "surface": 1
        },
        "numTraining": len(imgs),
        "file_ending": ".nii.gz",
        "name": DATASET_NAME,
        "reference": "Vesuvius Challenge",
        "release": "1.0",
        "description": "Vesuvius 3D Segmentation",
    }

    with open(os.path.join(out_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)

    print("--> nnUNet_raw dataset ready")


def run_preprocess():

    print("\n--> Running nnU-Net preprocessing")

    os.environ["nnUNet_raw"] = NNUNET_RAW
    os.environ["nnUNet_preprocessed"] = NNUNET_PREPROCESSED

    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d",
        str(DATASET_ID),
        "-pl",
        "nnUNetPlannerResEncM"
    ]

    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    print("--> Preprocessing completed")


def main():

    cfg = load_settings()
    raw_data_dir = cfg["RAW_DATA_DIR"]

    os.makedirs(NNUNET_RAW, exist_ok=True)
    os.makedirs(NNUNET_PREPROCESSED, exist_ok=True)

    build_nnunet_raw(raw_data_dir)
    run_preprocess()


if __name__ == "__main__":
    main()