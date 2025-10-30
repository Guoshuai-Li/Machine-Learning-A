# MLA 2025 Final Exam - README

**Student Submission for Machine Learning A (2025-2026)**  
**Sections Completed: 4.1 - 4.4 (Alzheimer's Disease Diagnosis)**

---

## Overview

This README describes how to run the source code for the Alzheimer's Disease Diagnosis task (Section 4) of the MLA 2025 Final Exam. The implementation covers data preprocessing, PCA analysis, clustering, and classification tasks.

---

## Requirements

### Software and Libraries

The code is implemented in **Python 3.12.3** and requires the following libraries:

- **pandas** - Data loading and manipulation
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning algorithms (PCA, clustering, classification)
- **matplotlib** - Plotting and visualization

### Installation

If you don't have these libraries installed, you can install them using:

```bash
pip install pandas numpy scikit-learn matplotlib
```

Or using conda:

```bash
conda install pandas numpy scikit-learn matplotlib
```

---

## Data Files

The code expects the following data files to be in the same directory as the notebook (or update the file paths in the code):

- `trainInput.csv` - Training features (2430 samples × 252 features)
- `trainTarget.csv` - Training labels
- `testInput.csv` - Test features (270 samples × 252 features)
- `testTarget.csv` - Test labels

**Data Source:** https://github.com/christian-igel/ML/tree/main/data/AD

---

## Running the Code

### Option 1: Using Jupyter Notebook (Recommended)

1. **Open the notebook:**
   ```bash
   jupyter notebook exam.ipynb
   ```

2. **Update file paths (if necessary):**
   
   In the first code cell, update the file paths to match your directory structure:
   ```python
   train_input = pd.read_csv('path/to/trainInput.csv', header=None)
   train_target = pd.read_csv('path/to/trainTarget.csv', header=None)
   test_input = pd.read_csv('path/to/testInput.csv', header=None)
   test_target = pd.read_csv('path/to/testTarget.csv', header=None)
   ```

3. **Run all cells:**
   - Click "Cell" → "Run All" in the menu, or
   - Press `Shift + Enter` to run each cell sequentially

### Option 2: Running as a Python Script

If you prefer to run the code as a script:

1. Convert the notebook to a Python script:
   ```bash
   jupyter nbconvert --to script exam.ipynb
   ```

2. Run the generated Python file:
   ```bash
   python exam.py
   ```

---

## Code Structure

The notebook is organized into the following sections corresponding to the exam tasks:

### **Section 4.1: Data Understanding and Preprocessing**
- **Cell 1:** Loads the training and test data
- **Outputs:**
  - Dataset dimensions
  - Class frequencies for training and test sets

### **Section 4.2: Principal Component Analysis (PCA)**
- **Cell 2:** Performs PCA on training data
- **Outputs:**
  - Eigenspectrum plot (`pca_eigenspectrum.png`)
  - Number of components explaining 90% of variance
  - Scatter plot of first 2 principal components (`pca_scatter.png`)

### **Section 4.3: Clustering**
- **Cell 3:** Implements k-means and k-means++ clustering
- **Initialization:** First data point from each class (as specified in exam)
  - Class 0: First occurrence in training data
  - Class 1: First occurrence in training data
- **Outputs:**
  - Cluster centers projected onto first 2 PCs
  - Visualization with cluster centers overlaid (`clustering_visualization.png`)
  - Discussion of clustering quality

### **Section 4.4: Classification**
- **Cell 4:** Trains and evaluates multiple classifiers

#### 4.4.1: Logistic Regression
- Uses L2 regularization (default in scikit-learn)
- Reports training and test 0-1 loss

#### 4.4.2: Random Forests
- **Configuration 1:** 200 trees, max_features=sqrt(n_features) ≈ 16
- **Configuration 2:** 200 trees, max_features=n_features (252)
- Reports training loss, test loss, and OOB error for both configurations

#### 4.4.3: K-Nearest Neighbors
- Uses 5-fold cross-validation to select k
- Tests k values from 1 to 50
- Reports best k value and corresponding errors
- Outputs CV error plot (`knn_cv_error.png`)

---

## Expected Outputs

### Figures Generated

The code generates the following figure files (saved in the working directory):

1. `pca_eigenspectrum.png` - Eigenvalue spectrum from PCA
2. `pca_scatter.png` - Data projected onto first 2 principal components
3. `clustering_visualization.png` - Clustering results with centers
4. `knn_cv_error.png` - Cross-validation error vs. k for KNN

### Console Output

The code prints comprehensive results including:

- Class frequencies for training and test sets
- Number of PCA components for 90% variance
- Cluster centers (before and after projection)
- Training and test errors for all classifiers
- OOB errors for Random Forests
- Cross-validation results for KNN
- Summary table comparing all methods

---

## Methodology Details

### Software Used
- **Python version:** 3.12.3
- **Environment:** Jupyter Notebook / Conda
- **Core libraries:** 
  - scikit-learn 1.x for all ML algorithms
  - pandas for data handling
  - numpy for numerical operations
  - matplotlib for visualization

### Key Implementation Choices

**PCA:**
- Centered but not standardized (as per standard PCA)
- Eigenvalues computed from covariance matrix via SVD

**Clustering:**
- Lloyd's algorithm (standard k-means)
- k-means++ uses D² weighting for initialization
- Cluster centers initialized as specified in exam (first point from each class)

**Logistic Regression:**
- Solver: 'lbfgs' (default)
- Regularization: L2 (Ridge), default C=1.0
- Max iterations: 10000 to ensure convergence

**Random Forest:**
- 200 trees (as specified)
- Bootstrap samples with replacement
- OOB error computed during training
- Split criterion: Gini impurity

**K-Nearest Neighbors:**
- Distance metric: Euclidean (default)
- Cross-validation: 5-fold stratified
- k range: 1 to 50
- Ties broken by first occurrence

---

## Troubleshooting

### Common Issues

**Issue 1: File not found error**
- **Solution:** Update file paths in Cell 1 to match your directory structure

**Issue 2: Module not found error**
- **Solution:** Install missing libraries using pip or conda (see Requirements section)

**Issue 3: Memory error**
- **Solution:** The dataset is small (2430 samples), but if you encounter memory issues, try closing other applications

**Issue 4: Figures not displaying**
- **Solution:** Ensure you're using `%matplotlib inline` in Jupyter or check that `plt.show()` is called

**Issue 5: Random Forest OOB error is NaN**
- **Solution:** Ensure `bootstrap=True` (default) and `oob_score=True` are set

---

## Notes

- All random processes use fixed random seeds where applicable for reproducibility
- The code follows exam requirements: training on training set only, test set used only for final evaluation
- Cross-validation for model selection is performed on training data only
- Plots are saved with high resolution (300 dpi) for report inclusion

---

## Contact

If you encounter any issues running this code, please verify:
1. All required libraries are installed
2. Data files are in the correct location
3. Python version is 3.8 or higher

---

## Submission Contents

This submission includes:
1. **exam.ipynb** - Jupyter notebook with complete implementation
2. **README.md** - This file, explaining how to run the code
3. **Report.pdf** - Detailed answers and analysis (separate file)
4. Generated figures (*.png files)

---

**Last Updated:** October 31, 2025  
**Exam:** Machine Learning A (2025-2026) Final Exam  
**University of Copenhagen - Department of Computer Science**
