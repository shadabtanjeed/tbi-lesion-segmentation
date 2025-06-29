## v1 Progress Overview

## v4 Progress Overview

### Model
- **Architecture:** Improved 3D U-Net with Attention Gates and Deep Supervision + Optuna Hyperparameter Optimization

### Hyperparameters

- **Model Architecture:**
  - ImprovedUNet3D
    - Input channels: 1
    - Output channels: 1
    - Features: [32, 64, 128, 256, 512] (encoder/decoder layers)
    - Attention gates and deep supervision outputs
    - Bottleneck: 50% dropout

- **Loss Function:**
  - Combined Loss:
    - 0.3 × BCEWithLogitsLoss (binary cross-entropy with logits, pos_weight=75.0)
    - 0.4 × SoftDiceLoss (soft Dice coefficient loss)
    - 0.3 × FocalTverskyLoss (alpha=0.7, beta=0.3, gamma=2.0)
    - Deep supervision: 0.2 × additional Dice loss on intermediate outputs

- **Optimizer:**
  - AdamW (Optuna optimized)
    - Learning rate: (Optuna optimized)
    - Weight decay: (Optuna optimized)

- **Learning Rate Scheduler:**
  - ReduceLROnPlateau
    - Mode: max (maximize validation Dice score)
    - Factor: 0.5 (reduce LR by half)
    - Patience: 5 epochs (wait for improvement)
    - Min LR: 1e-7

- **Training Parameters:**
  - Batch Size: (Optuna optimized - options: 1, 2, 4)
  - Epochs: 30
  - Patience for Early Stopping: 10 epochs (stop if no improvement)
  - Mixed Precision Training: Enabled with GradScaler

- **Metrics:**
  - Dice Coefficient (Threshold: 0.3 for binary segmentation)

### Preprocessing Pipeline
1. **File Listing:** Identify all `.nii` files in the dataset directory.
2. **Scan ID Extraction:** Remove suffixes (e.g., `_T1.nii`) to pair scans and masks.
3. **Target Shape:** Resize scans and masks to `(128, 128, 128)` for uniformity.
4. **Skip Processed Files:** Check for existing preprocessed files and skip if present.
5. **Data Loading:** Use `nibabel` to load scans and masks as NumPy arrays.
6. **Advanced Scan Normalization:**
    - Percentile clipping (1st-99th percentile) to remove extreme outliers
    - Z-score normalization on non-zero voxels only
7. **Mask Binarization:** Convert lesion masks to binary (0 or 1).
8. **Resampling:** 
    - Scans: Trilinear interpolation
    - Masks: Nearest-neighbor interpolation
9. **Statistics Computation:** Calculate mean, standard deviation, and lesion voxel count.
10. **Noise Detection:** Flag scans with low statistics or no lesion voxels.
11. **Tensor Saving:** Store preprocessed scans and masks as `.pt` files.

### Training Workflow
1. **Hyperparameter Optimization:** Use Optuna to optimize learning rate, weight decay, and batch size (5 trials).
2. **Model Initialization:** Define the improved 3D U-Net with attention gates for segmentation.
3. **Loss Function:** Combine BCEWithLogitsLoss, Dice loss, and FocalTverskyLoss with deep supervision.
4. **Data Preparation:** Split preprocessed data into training (90%) and validation (10%) sets.
5. **Training Loop:** Forward pass, loss computation, and weight updates with mixed precision training.
6. **Validation:** Evaluate on validation data and compute metrics.
7. **Model Checkpointing:** Save the model with the highest validation Dice score.
8. **Early Stopping:** Stop training if no improvement after 10 epochs patience.
9. **Learning Rate Scheduling:** Reduce LR by half if no improvement for 5 epochs.
10. **Logging:** Record training/validation loss and Dice scores for analysis.

### Training Statistics (First 23 Epochs)

| Epoch | Train Loss | Train Dice | Val Loss | Val Dice | Best Model | Patience Counter |
|-------|------------|------------|----------|----------|------------|------------------|
| 1     | 0.9758     | 0.0481     | 0.7312   | 0.0172   | ✅         | 0                |
| 2     | 0.9296     | 0.0192     | 0.7169   | 0.0270   | ✅         | 0                |
| 3     | 0.9161     | 0.0692     | 0.7147   | 0.0079   |            | 1                |
| 4     | 0.9076     | 0.0317     | 0.6986   | 0.0158   |            | 2                |
| 5     | 0.9056     | 0.0296     | 0.7186   | 0.0065   |            | 3                |
| 6     | 0.9100     | 0.0279     | 0.7012   | 0.0134   |            | 4                |
| 7     | 0.8968     | 0.0448     | 0.6950   | 0.0438   | ✅         | 0                |
| 8     | 0.8945     | 0.0414     | 0.7016   | 0.0142   |            | 1                |
| 9     | 0.8901     | 0.0758     | 0.6926   | 0.0304   |            | 2                |
| 10    | 0.8919     | 0.0518     | 0.6889   | 0.0279   |            | 3                |
| 11    | 0.8915     | 0.0572     | 0.6859   | 0.1035   | ✅         | 0                |
| 12    | 0.8873     | 0.0688     | 0.6867   | 0.0281   |            | 1                |
| 13    | 0.8784     | 0.0633     | 0.6963   | 0.2986   | ✅         | 0                |
| 14    | 0.8882     | 0.1325     | 0.6790   | 0.0593   |            | 1                |
| 15    | 0.8727     | 0.0842     | 0.6801   | 0.2819   |            | 2                |
| 16    | 0.8796     | 0.0851     | 0.6890   | 0.0250   |            | 3                |
| 17    | 0.8658     | 0.0941     | 0.8281   | 0.0054   |            | 4                |
| 18    | 0.9015     | 0.0537     | 0.7077   | 0.0158   |            | 5                |
| 19    | 0.8863     | 0.0718     | 0.6811   | 0.0539   |            | 6                |
| 20    | 0.8573     | 0.1166     | 0.6807   | 0.1971   |            | 7                |
| 21    | 0.8414     | 0.1338     | 0.6777   | 0.1254   |            | 8                |
| 22    | 0.8491     | 0.1058     | 0.6663   | 0.2779   |            | 9                |
| 23    | 0.8385     | 0.1316     | 0.6686   | 0.0554   |            | 10               |

- **Note:** Best model saved at epochs 1, 2, 7, 11, and 13. No improvement for 10 epochs after epoch 13; early stopping triggered at epoch 23.
- **Best Dice score:** 0.2986

#### Validation Segmentation Stats

- **True Positives:** 10,996
- **False Positives:** 208,129
- **False Negatives:** 17,027
- **True Negatives:** 117,204,360
- **Mean Dice:** 0.0633
- **Mean Vol Diff:** 6,825.07 voxels
- **Total Vol Pred:** 219,125
- **Total Vol True:** 28,023

### Submission Metrics
- **Mean Position:** 
- **Absolute Volume Difference:** 
- **Absolute Lesion Count Difference:** 
- **Lesion-wise F1 Score:** 

**Notebook:** `all-mask-lesion-fix-optuna-unet-tbi-lesion.ipynb`  
**Model Name:** `all_masks_lesion_fix_optuna_unet3d_model.pt`
