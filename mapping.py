import os
import json
import glob
from tqdm import tqdm

CONFIG = {
    "src_img": "data/train_images",
    "dataset_id": 501,
    "dataset_name": "Vesuvius320",
    "out_dir": "nnUNet_raw"
}

def main():
    folder_name = f"Dataset{CONFIG['dataset_id']}_{CONFIG['dataset_name']}"
    out_dir = os.path.join(CONFIG['out_dir'], folder_name)
    os.makedirs(out_dir, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(CONFIG['src_img'], "*.tif")))

    name_mapping = {}

    print(f"--> Creating mapping for {len(imgs)} files...")

    for i, imp in tqdm(enumerate(imgs), total=len(imgs)):
        case_id = f"Vesuvius_{i:03d}"
        old_name = os.path.splitext(os.path.basename(imp))[0]
        name_mapping[old_name] = case_id

    with open(os.path.join(out_dir, "name_mapping.json"), "w") as f:
        json.dump(name_mapping, f, indent=4)

    print("--> Done. Mapping saved.")

if __name__ == "__main__":
    main()
