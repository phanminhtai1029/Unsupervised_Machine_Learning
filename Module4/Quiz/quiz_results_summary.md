# Coursera Quiz Results Summary - Module 4

## Overall Performance Table

| Quiz Name | Total Questions | Correct Answers | Score | Status |
|-----------|----------------|-----------------|-------|---------|
| Graded Module 4 Quiz | 5 | 4 | 80% | **PASS** |
| Ungraded Dimensionality Reduction | 3 | 3 | 100% | **PASS** |

---

## Topics Covered

### Graded Module 4 Quiz (Dimensionality Reduction - Coursera)

The graded quiz assessed understanding of Principal Component Analysis (PCA) with the following key topics:

**1. Principal Component Analysis Fundamentals**
   - Understanding that PCA generates new features as linear combinations of original features
   - Recognizing that PCA creates derived features rather than simply selecting or discarding existing ones

**2. PCA Implementation in Python**
   - Proper sequencing of steps: fitting PCA to data, scaling data, determining optimal number of components based on explained variance, and defining the PCA object
   - Understanding the correct workflow for applying PCA in practice

**3. Singular Value Decomposition and Eigenvalues**
   - Ranking singular vectors by their importance based on eigenvalue magnitude
   - Understanding that larger eigenvalues on the diagonal indicate greater importance

**4. Feature Contribution Analysis**
   - Interpreting component loadings to determine feature importance
   - Using absolute values of coefficients to assess contribution magnitude
   - Comparing total contributions across multiple principal components

**5. Principal Component Interpretation**
   - Understanding that principal components are mathematical constructs (linear combinations)
   - Recognizing that the first principal component does not directly correspond to the single most important original feature

### Ungraded Dimensionality Reduction Quiz

This ungraded practice quiz covered foundational concepts:

**1. Purpose of Dimensionality Reduction**
   - Understanding the primary goal of reducing features to improve model performance
   - Recognizing methods include feature selection and feature creation

**2. PCA Mechanism**
   - Clarifying that PCA creates new features through linear combinations
   - Understanding that PCA does not simply exclude features

**3. Variance Preservation**
   - Understanding that principal components are ordered by explained variance
   - Recognizing that earlier components (v₁) retain more information than later ones (v₂)

---

## Incorrect Responses Analysis

### Question 2 (Graded Module 4 Quiz)

**Question:** Which option correctly lists the steps for implementing PCA in Python?

**Your Answer:** 2, 1, 3, 4
- Step 2: Scale the data
- Step 1: Fit PCA to data  
- Step 3: Determine the desired number of components based on total explained variance
- Step 4: Define a PCA object

**Correct Answer:** The correct sequence should follow standard machine learning workflow principles.

**Why This Was Incorrect:** The proper implementation sequence for PCA requires careful attention to the order of operations. In scikit-learn and standard practice, you must first define the PCA object, then scale your data (as PCA is sensitive to feature scales), determine how many components you want based on explained variance analysis, and finally fit the PCA model to your scaled data. The sequence you selected attempted to fit PCA before properly scaling the data and before defining the PCA object itself, which would cause errors in implementation.