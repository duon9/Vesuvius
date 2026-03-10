# Vesuvius Challenge - Surface Detection Solution

This repository contains the source code and instructions to reproduce the winning solution for the **Vesuvius Challenge - Surface Detection** competition.

---

## 📂 Archive Contents

* **`kaggle_model.zip`**: Original Kaggle model upload. Includes original source code, additional training examples, corrected labels, and weight files.
* **`train.py`**: Script to rebuild models from scratch.
* **`prepare_data.py`**: Script for data ingestion and preprocessing.
* **`predict.py`**: Script to generate inference predictions from model binaries.

---

## 💻 System Requirements

### Hardware
The following specifications were used to develop and train the original solution:
* **OS:** Ubuntu 22.04.5 LTS x86_64
* **CPU:** Intel Xeon Bronze 3104 (6 @ 1.700GHz)
* **RAM:** 32GB
* **GPU:** 1 x NVIDIA RTX A4000

### Software & Environment
* **Python:** 3.10.12
* **CUDA:** 12.9
* **cuDNN:** 9.10.2
* **Driver Version:** 575.57.08
* **Docker Base:** [nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04](https://hub.docker.com/layers/nvidia/cuda/12.9.0-cudnn-devel-ubuntu22.04/images/sha256-ef33852f3d321c9aedee5103f57b247114407d2e8382fe291a7ea5b2e6cb94ce)

> **Note:** Python package dependencies are listed in `requirements.txt`.

---

## 🚀 Step-by-Step Setup

### 1. Data Acquisition
Ensure the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed and configured. Run the following to download the competition data:

```bash
mkdir -p data
cd data
kaggle competitions download -c vesuvius-challenge-surface-detection
unzip *.zip
cd ..
```
### 2. Data Processing
Preprocess the raw data into the required format for training:
 
```bash
python ./src/prepare_data.py
```

### 3. Model Training
To train the models from scratch, execute:

```bash
python ./src/train.py
```
### 4. Inference & Submission
To generate predictions using the trained weights, run:

```bash
python ./src/predict.py
```
A submission.zip file will be generated in the output directory.

📧 Contact & Support
If you encounter any issues with the setup, code execution, or have general questions, please feel free to reach out:
Email: marius_heuser@web.de
Email: ngducduong305@gmail.com
