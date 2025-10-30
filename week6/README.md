# MLA 2025 Final Exam - README

**Sections Completed: 4.1 - 4.4 (Alzheimer's Disease Diagnosis)**


---

## Requirements

### Software and Libraries

The code is implemented in **Python 3.12.3** and requires the following libraries:

- **pandas** - Data loading and manipulation
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning algorithms (PCA, clustering, classification)
- **matplotlib** - Plotting and visualization

## Data Files

The code expects the following data files to be in the same directory as the notebook (or update the file paths in the code):

- `trainInput.csv` - Training features (2430 samples × 252 features)
- `trainTarget.csv` - Training labels
- `testInput.csv` - Test features (270 samples × 252 features)
- `testTarget.csv` - Test labels

**Data Source:** https://github.com/christian-igel/ML/tree/main/data/AD

---

## Running the Code

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

## Notes

- All random processes use fixed random seeds where applicable for reproducibility
- The code follows exam requirements: training on training set only, test set used only for final evaluation
- Cross-validation for model selection is performed on training data only
- Plots are saved with high resolution (300 dpi) for report inclusion

---


