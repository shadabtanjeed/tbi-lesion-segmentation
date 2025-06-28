## v1 Progress Overview

### Model
- **Architecture:** 3D U-Net

### Hyperparameters

- **Model Architecture:**
  - UNet3D
    - Input channels: 1
    - Output channels: 1
    - Features: [32, 64, 128, 256] (encoder/decoder layers)

- **Loss Function:**
  - Combined Loss:
    - 0.5 × BCEWithLogitsLoss (binary cross-entropy with logits)
    - 0.5 × SoftDiceLoss (soft Dice coefficient loss)
    - BCE Pos Weight: 5.0 (to handle class imbalance)

- **Optimizer:**
  - Adam
    - Learning rate: 1e-4

- **Learning Rate Scheduler:**
  - ReduceLROnPlateau
    - Mode: max (maximize validation Dice score)
    - Factor: 0.5 (reduce LR by half)
    - Patience: 3 epochs (wait for improvement)

- **Training Parameters:**
  - Batch Size: 2
  - Epochs: 20
  - Patience for Early Stopping: 10 epochs (stop if no improvement)

- **Metrics:**
  - Dice Coefficient (Threshold: 0.5 for binary segmentation)

### Preprocessing Pipeline
1. **File Listing:** Identify all `.nii` files in the dataset directory.
2. **Scan ID Extraction:** Remove suffixes (e.g., `_T1.nii`) to pair scans and masks.
3. **Target Shape:** Resize scans and masks to `(128, 128, 128)` for uniformity.
4. **Skip Processed Files:** Check for existing preprocessed files and skip if present.
5. **Data Loading:** Use `nibabel` to load scans and masks as NumPy arrays.
6. **Scan Normalization:** Scale scan intensities to `[0, 1]`.
7. **Mask Binarization:** Convert lesion masks to binary (0 or 1).
8. **Resampling:** 
    - Scans: Trilinear interpolation
    - Masks: Nearest-neighbor interpolation
9. **Statistics Computation:** Calculate mean, standard deviation, and lesion voxel count.
10. **Noise Detection:** Flag scans with low statistics or no lesion voxels.
11. **Tensor Saving:** Store preprocessed scans and masks as `.pt` files.

### Training Workflow
1. **Model Initialization:** Define the 3D U-Net for segmentation.
2. **Loss Function:** Combine `BCEWithLogitsLoss` and Dice loss.
3. **Data Preparation:** Split preprocessed data into training and validation sets.
4. **Training Loop:** Forward pass, loss computation, and weight updates.
5. **Validation:** Evaluate on validation data and compute metrics.
6. **Model Checkpointing:** Save the model with the highest validation Dice score.
7. **Early Stopping:** Stop training if no improvement after a set patience.
8. **Logging:** Record training/validation loss and Dice scores for analysis.

### Training Statistics (First 7 Epochs)
| Epoch | Train Loss | Train Dice | Val Loss | Val Dice | Best Model | Patience Counter |
|-------|------------|------------|----------|----------|------------|------------------|
| 1     | 0.6157     | 0.1512     | 0.5997   | 0.2143   | ✅         | 0                |
| 2     | 0.5879     | 0.3085     | 0.5767   | 0.2321   | ✅         | 0                |
| 3     | 0.5680     | 0.3790     | 0.5614   | 0.3036   | ✅         | 0                |
| 4     | 0.5532     | 0.4194     | 0.5495   | 0.3036   |            | 1                |
| 5     | 0.5421     | 0.4194     | 0.5379   | 0.3036   |            | 2                |
| 6     | 0.5337     | 0.4254     | 0.5302   | 0.3036   |            | 3                |
| 7     | ...        | ...        | ...      | ...      |            | ...              |

- **Note:** Best model saved at epochs 1–3. No improvement from epoch 4 onward; patience counter incremented.

### Submission Metrics
- **Mean Position:** 1.8
- **Absolute Volume Difference:** 7.1870 (Rank: 4)
- **Absolute Lesion Count Difference:** 3.8200 (Rank: 1)
- **Lesion-wise F1 Score:** 0.4700 (Rank: 1)


**Notebook:** `Training_AIMS_TBI_Lesion_Segmentation.ipynb`
**Model Name:** `best_3dunet_v2_epoch_20.pt`

---