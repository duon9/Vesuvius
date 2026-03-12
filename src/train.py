import os
import subprocess

# Paths (Using absolute paths is recommended to avoid issues)
NNUNET_RAW = "nnunet/nnUNet_raw"
NNUNET_PREPROCESSED = "nnunet/nnUNet_preprocessed"
NNUNET_RESULTS = "nnunet/nnUNet_results"

def run_nnunet_train():
    # Set environment variables
    os.environ["nnUNet_raw"] = os.path.abspath(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = os.path.abspath(NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = os.path.abspath(NNUNET_RESULTS)

    # Ensure directories exist
    os.makedirs(NNUNET_RAW, exist_ok=True)
    os.makedirs(NNUNET_PREPROCESSED, exist_ok=True)
    os.makedirs(NNUNET_RESULTS, exist_ok=True)

    # List of configurations to run sequentially
    # Ensure "3d_fullres_ps128" and "3d_fullres_ps160" exist in your plans file
    configs_to_run = ["3d_fullres_ps128", "3d_fullres_ps160"]
    dataset_id = "501"
    fold = "all"
    plans_identifier = "nnUNetResEncUNetMPlans"

    for config in configs_to_run:
        print(f"\n" + "="*50)
        print(f"🚀 STARTING TRAINING: Configuration {config}")
        print("="*50 + "\n")

        # Training command for each configuration
        cmd = [
            "nnUNetv2_train",
            dataset_id,
            config,
            fold,
            "-p",
            plans_identifier
        ]

        try:
            # Running without subprocess.PIPE to allow real-time console output (loss, epochs)
            # This prevents the terminal from appearing "frozen" during long training sessions
            result = subprocess.run(
                cmd,
                check=True,  # Raises CalledProcessError if the command fails
                text=True
            )

            if result.returncode == 0:
                print(f"✅ Training completed successfully for {config}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error occurred while running {config}: {e}")
            # Decide whether to stop or continue to the next config
            # Use 'break' to stop entirely, or 'continue' to move to the next task
            continue 

if __name__ == "__main__":
    run_nnunet_train()