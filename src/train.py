import subprocess

def run_nnunet_train():
    # Lệnh cần chạy
    cmd = [
        "nnUNetv2_train",
        "501",
        "3d_fullres",
        "all",
        "-p",
        "nnUNetResEncUNetMPlans"
    ]

    try:
        # Chạy command
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # In kết quả
        print("===== STDOUT =====")
        print(result.stdout)

        print("===== STDERR =====")
        print(result.stderr)

        # Kiểm tra exit code
        if result.returncode == 0:
            print("Training started successfully ✅")
        else:
            print(f"Command failed with code {result.returncode} ❌")

    except Exception as e:
        print("Error while running command:", str(e))


if __name__ == "__main__":
    run_nnunet_train()