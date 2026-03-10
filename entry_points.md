1. `python prepare_data.py`, which would
   - Read training data from `RAW_DATA_DIR` (specified in `SETTINGS.json`)
   - Run any preprocessing steps

2. `python train.py`, which would
   - Train your model.
   - Save your model to `nnunet/nnUNet_results`

3. `python predict.py`, which would
   - Read test data from `TEST_DATA_DIR` (specified in `SETTINGS.json`)
   - Auto load your model from `nnunet/nnUNet_results`
   - Use your model to make predictions on new samples
   - Save your predictions to `OUTPUT_DIR` (specified in `SETTINGS.json`)