### Prerequisites:
1. Python 3.7 or higher
2. CUDA-capable GPU or CPU

### Required Libraries Installation:
```
pip install torch torchvision
pip install medmnist
pip install timm
pip install scikit-learn
pip install numpy
```

### Running Instructions:

1. File Organization:(Both training and testing are in same file)
   - Save the first code as `Chest.py` or by using jupyter notebook save `chest.ipynb`
   - Save the second code as `Breast&Pneumonia.py` or by using jupyter notebook save `Breast&Pneumonia.ipynb`
   - Place both files in the same directory

2. **Running the Chest X-ray Classification:**
   ```
   python Chest.py
   ```
   using jupyter notebook, copy the program in the cells and execute each cell.
   This will:
   - Train ResNet18, ViT, and DenseNet models on chest X-ray dataset
   - Save the best models as `ResNet18_chest_best.pth`, `ViT_chest_best.pth`, and `DenseNet_chest_best.pth`
   - Display training progress and final results

3. **Running the Breast and Pneumonia Classification:**
   ```
   python Breast\&Pneumonia.py
   ```
   using jupyter notebook, copy the program in the cells and execute each cell.
   This will:
   - Train ResNet18, ViT, and DenseNet models on both breast and pneumonia datasets
   - Save the best models with corresponding names (e.g., `ResNet18_breast_best.pth`, `ViT_pneumonia_best.pth`)
   - Display training progress and final results

### Notes:
- Default sample sizes:
  - Chest: 1000 samples
  - Breast & Pneumonia: 100 samples
- Training runs for 5 epochs by default
- Models are automatically saved in the same directory
- Results include AUC scores for each model-dataset combination
- The code will automatically use GPU if available, otherwise CPU
