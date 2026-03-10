Hello!

Below you can find a outline of how to reproduce my solution for the Vesuvius Challenge - Surface Detection competition.
If you run into any trouble with the setup/code or have any questions please contact me at ngducduong305@gmail.com

# ARCHIVE CONTENTS
kaggle_model.tgz          : original kaggle model upload - contains original code, additional training examples, corrected labels, etc
train.py                  : code to rebuild models from scratch
prepare_data.py           : code to prepare and preprocess data
predict.py                : code to generate predictions from model binaries

# HARDWARE: (The following specs were used to create the original solution)
Ubuntu 22.04.5 LTS x86_64
Intel Xeon Bronze 3104 (6) @ 1.700GHz
32GB RAM
1 x NVIDIA RTX A4000

# SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.10.12
CUDA 12.9
Driver Version: 575.57.08
cuddn 9.10.2
-- Equivalent Dockerfile for the GPU installs: https://hub.docker.com/layers/nvidia/cuda/12.9.0-cudnn-devel-ubuntu22.04/images/sha256-ef33852f3d321c9aedee5103f57b247114407d2e8382fe291a7ea5b2e6cb94ce

# DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
mkdir -p data
cd data
kaggle competitions download -c vesuvius-challenge-surface-detection && unzip *.zip


# DATA PROCESSING
python ./src/prepare_data.py

# MODEL BUILD: Run this command.
python ./src/train.py

# INFERENCE: Run this command, a submission.zip will be created on specific folder.
python ./src/predict.py

