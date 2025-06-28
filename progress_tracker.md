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

## v2 Progress Overview

### Model
- **Architecture:** Improved 3D U-Net with Attention Gates and Deep Supervision

### Hyperparameters

- **Model Architecture:**
  - ImprovedUNet3D
    - Input channels: 1
    - Output channels: 1
    - Features: [32, 64, 128, 256, 512] (encoder/decoder layers)
    - Attention gates and deep supervision outputs

- **Loss Function:**
  - Combined Loss:
    - 0.3 × BCEWithLogitsLoss (binary cross-entropy with logits, pos_weight=10.0)
    - 0.4 × SoftDiceLoss (soft Dice coefficient loss)
    - 0.3 × FocalTverskyLoss (advanced overlap loss)
    - Deep supervision: additional Dice loss on intermediate outputs

- **Optimizer:**
  - AdamW
    - Learning rate: 1e-4
    - Weight decay: 1e-5

- **Learning Rate Scheduler:**
  - ReduceLROnPlateau
    - Mode: max (maximize validation Dice score)
    - Factor: 0.5 (reduce LR by half)
    - Patience: 5 epochs (wait for improvement)
    - Min LR: 1e-7

- **Training Parameters:**
  - Batch Size: 1
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
6. **Scan Normalization:** Z-score normalization on non-zero voxels, after percentile clipping.
7. **Mask Binarization:** Convert lesion masks to binary (0 or 1).
8. **Resampling:** 
    - Scans: Trilinear interpolation
    - Masks: Nearest-neighbor interpolation
9. **Statistics Computation:** Calculate mean, standard deviation, and lesion voxel count.
10. **Noise Detection:** Flag scans with low statistics or no lesion voxels.
11. **Tensor Saving:** Store preprocessed scans and masks as `.pt` files.

### Training Workflow
1. **Model Initialization:** Define the improved 3D U-Net for segmentation.
2. **Loss Function:** Combine BCEWithLogitsLoss, Dice loss, and FocalTverskyLoss (with deep supervision).
3. **Data Preparation:** Split preprocessed data into training and validation sets.
4. **Training Loop:** Forward pass, loss computation, and weight updates (mixed precision).
5. **Validation:** Evaluate on validation data and compute metrics.
6. **Model Checkpointing:** Save the model with the highest validation Dice score.
7. **Early Stopping:** Stop training if no improvement after a set patience.
8. **Logging:** Record training/validation loss and Dice scores for analysis.

### Training Statistics (First 11 Epochs)

| Epoch | Train Loss | Train Dice | Val Loss | Val Dice | Best Model | Patience Counter |
|-------|------------|------------|----------|----------|------------|------------------|
| 1     | 0.9529     | 0.3596     | 0.7431   | 0.4107   | ✅         | 0                |
| 2     | 0.9317     | 0.3960     | 0.7251   | 0.4107   |            | 1                |
| 3     | 0.9184     | 0.3745     | 0.7176   | 0.0319   |            | 2                |
| 4     | 0.9106     | 0.1146     | 0.7127   | 0.0197   |            | 3                |
| 5     | 0.9046     | 0.0570     | 0.7066   | 0.0329   |            | 4                |
| 6     | 0.8968     | 0.0628     | 0.7040   | 0.0422   |            | 5                |
| 7     | 0.8895     | 0.0608     | 0.7047   | 0.0936   |            | 6                |
| 8     | 0.8779     | 0.0673     | 0.6979   | 0.1988   |            | 7                |
| 9     | 0.8709     | 0.0713     | 0.6924   | 0.0717   |            | 8                |
| 10    | 0.8630     | 0.0839     | 0.6822   | 0.1146   |            | 9                |
| 11    | 0.8572     | 0.0825     | 0.6930   | 0.2930   |            | 10               |

- **Note:** Best model saved at epoch 1. No improvement from epoch 2 onward; patience counter incremented. Early stopping triggered at epoch 11.  
- **Best Dice score:** 0.4107


### Submission Metrics
- **Mean Position:** 
- **Absolute Volume Difference:** 
- **Absolute Lesion Count Difference:** 
- **Lesion-wise F1 Score:** 

**Notebook:** `adv-preproc-unet-aims-tbi-lesion-segmentation.ipynb`  
**Model Name:** `improved_processing_unet_v1.pt`