# Kaggle Competition Solution

## 1. Overview
This repository contains the complete solution used to generate the final submission for the Kaggle competition.  
All required code, configuration files, and the trained model are included to allow the host to reproduce the results.

The pipeline consists of three main stages:

1. Data preparation
2. Model training
3. Prediction generation

---

## 2. Hardware

The solution was developed and trained using the following hardware:

- CPU: Intel Core i7 (8 cores)
- RAM: 16 GB
- GPU: NVIDIA RTX 3060 (1 GPU)

---

## 3. Software Environment

- Operating System: Ubuntu 22.04.5 LTS x86_64
- Python Version: Python 3.10.12

Install required dependencies:

pip install -r requirements.txt

All library versions are specified in `requirements.txt`.

---

## 4. Project Structure

.
├── README.md  
├── requirements.txt  
├── SETTINGS.json  
├── directory_structure.txt  
├── entry_points.md  

├── config/  
│   └── keras.json  

├── data/  
│   ├── raw/  
│   └── clean/  

├── src/  
│   ├── prepare_data.py  
│   ├── train.py  
│   └── predict.py  

├── model/  
│   └── trained_model.pkl  

└── outputs/  
    └── submission.csv  

---

## 5. Configuration

All directory paths are specified in the `SETTINGS.json` file.

Example:

{
  "RAW_DATA_DIR": "./data/raw",
  "CLEAN_DATA_DIR": "./data/clean",
  "MODEL_DIR": "./model",
  "SUBMISSION_DIR": "./outputs"
}

All scripts performing input/output operations read paths from this configuration file.

---

## 6. Data

The dataset is provided by the Kaggle competition and must be downloaded separately.

Place the downloaded dataset into:

data/raw/

---

## 7. Data Preparation

Run the following command to preprocess the dataset:

python src/prepare_data.py

This script performs the following tasks:

- Reads raw data from `RAW_DATA_DIR`
- Applies preprocessing and feature engineering
- Saves processed data into `CLEAN_DATA_DIR`

---

## 8. Model Training

To train the model, run:

python src/train.py

This script:

- Loads processed training data
- Trains the machine learning model
- Saves the trained model into `MODEL_DIR`

Checkpoint files may also be saved during training.

---

## 9. Prediction

To generate predictions for the test dataset:

python src/predict.py

This script:

- Loads the trained model from `MODEL_DIR`
- Reads test data
- Generates predictions
- Saves the submission file into `SUBMISSION_DIR`

---

## 10. Trained Model

A serialized copy of the trained model is stored in:

model/trained_model.pkl

This allows predictions to be generated without retraining the model.

---

## 11. Important Notes

- The `outputs` directory should be empty before running prediction.
- Data preprocessing may overwrite existing cleaned data.
- All scripts rely on the directory paths defined in `SETTINGS.json`.

---

## 12. Reproducibility

To fully reproduce the solution:

1. Install dependencies
2. Download the Kaggle dataset
3. Run data preprocessing
4. Train the model
5. Generate predictions

Commands summary:

python src/prepare_data.py  
python src/train.py  
python src/predict.py