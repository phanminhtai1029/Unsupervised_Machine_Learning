# Coursera Quiz Results Summary - Module 5

## Overview Table

| Quiz Name | Total Questions | Correct Answers | Score | Status |
|-----------|----------------|-----------------|-------|---------|
| Graded Module 5 Quiz | 5 | 5 | 100% | **PASS** ✓ |
| Ungraded: Kernel PCA and MDS | 3 | 2 | 66.66% | Review Needed |

---

## Detailed Analysis

### Quiz 1: Graded Module 5 Quiz (100%)

**Status:** Excellent performance - all questions answered correctly!

**Topics Covered:**

1. **Kernel PCA Fundamentals** - Understanding the core difference between Kernel PCA and Linear PCA, specifically how kernel functions enable discovery of non-linear structures by mapping data to higher dimensions through the kernel trick.

2. **Multidimensional Scaling (MDS) Theory** - The fundamental principle that MDS focuses on maintaining geometric distances between data points during dimensionality reduction.

3. **Data Suitability for Kernel PCA** - Recognition that Kernel PCA is particularly valuable when working with data that is not linearly separable, as it can identify nonlinear features through higher-dimensional mapping.

4. **MDS Applications** - Understanding that MDS aims to find embeddings that minimize the "stress" cost function while preserving the original distance relationships between points.

5. **GridSearchCV Hyperparameters** - Knowledge that n_clusters is not a valid hyperparameter for Kernel PCA when using GridSearchCV (valid parameters include n_components, gamma, and kernel type).

**Key Concepts Demonstrated:**
- The kernel trick allows transformation to higher dimensions without explicit computation
- MDS preserves distance relationships rather than variance
- Kernel PCA excels with non-linearly separable data
- Understanding of proper hyperparameter tuning methods

---

### Quiz 2: Ungraded - Kernel PCA and MDS (66.66%)

**Status:** Partial understanding - one area needs review

#### **Correct Responses (2/3):**

**Question 1: When to Use Kernel PCA vs PCA**
- Correctly identified that Kernel PCA should be used when data is not linearly separable
- Understanding: When data cannot be clearly separated in its original lower dimension, a kernel function maps it to a higher dimension first before applying PCA

**Question 2: MDS vs PCA Goals**
- Correctly recognized that MDS tries to maintain geometric distances between data points, whereas PCA tries to preserve variance within data
- Key distinction: MDS is distance-preserving, PCA is variance-preserving

#### **Incorrect Response (1/3):**

**Question 3: Kernel PCA Reconstruction**

**Question:** If the number of components equals the dimension of the original features, kernel PCA will reconstruct the data, returning the original.

**Your Answer:** True (Incorrect)

**Correct Answer:** False

**Why This Matters:** This reveals an important conceptual difference between linear PCA and kernel PCA. In linear PCA, if you use all principal components (equal to the original number of features), you can perfectly reconstruct the original data. However, kernel PCA works differently because it operates in a transformed feature space created by the kernel function. Even with a number of components equal to the original dimension, kernel PCA maps the data to a different (often higher-dimensional) space through the kernel trick, so it cannot simply return the original data by using all components.

**What to Review:** 
- The fundamental difference between the feature space in linear PCA versus the kernel-induced feature space in kernel PCA
- How the kernel trick implicitly maps data to a higher (possibly infinite) dimensional space
- Why reconstruction in kernel PCA is more complex than in linear PCA
- Practice lab: Kernel PCA (as suggested by the feedback)