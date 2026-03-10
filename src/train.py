import os
import subprocess


NNUNET_RAW = "nnunet/nnUNet_raw"
NNUNET_PREPROCESSED = "nnunet/nnUNet_preprocessed"
NNUNET_RESULTS = "nnunet/nnUNet_results"


def run_nnunet_train():

    # set environment variables
    os.environ["nnUNet_raw"] = NNUNET_RAW
    os.environ["nnUNet_preprocessed"] = NNUNET_PREPROCESSED
    os.environ["nnUNet_results"] = NNUNET_RESULTS

    # đảm bảo folder tồn tại
    os.makedirs(NNUNET_RAW, exist_ok=True)
    os.makedirs(NNUNET_PREPROCESSED, exist_ok=True)
    os.makedirs(NNUNET_RESULTS, exist_ok=True)

    # command training
    cmd = [
        "nnUNetv2_train",
        "501",
        "3d_fullres",
        "all",
        "-p",
        "nnUNetResEncUNetMPlans"
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print("===== STDOUT =====")
        print(result.stdout)

        print("===== STDERR =====")
        print(result.stderr)

        if result.returncode == 0:
            print("Training started successfully ✅")
        else:
            print(f"Command failed with code {result.returncode} ❌")

    except Exception as e:
        print("Error while running command:", str(e))


if __name__ == "__main__":
    run_nnunet_train()