# Coursera Quiz Results Summary - Module 6

## Results Overview Table

| Quiz Name | Total Questions | Correct Answers | Score | Status |
|-----------|----------------|-----------------|-------|--------|
| Module 6 Quiz - Non-Negative Matrix Factorization (Graded) | 5 | 5 | 100% | **PASS** ✓ |
| Non-Negative Matrix Factorization (Ungraded Practice) | 3 | 3 | 100% | **PASS** ✓ |

---

## Topics Covered

### Module 6 Graded Quiz Topics

The graded assessment covered five key areas of Non-Negative Matrix Factorization:

**1. Interpretability of NMF Features**
This question explored how NMF can create more human-interpretable latent features compared to other dimensionality reduction techniques. The key insight is that by constraining the decomposition to non-negative values, NMF produces components that are additive in nature, making them easier to interpret in many real-world applications.

**2. Feature Suitability for NMF**
You correctly identified which types of features are best suited for NMF analysis. Monthly returns of stock portfolios emerged as the most appropriate application because NMF works optimally with data that can be meaningfully represented as non-negative values. Other options like word counts, pixel values, and spectral decompositions can also work, but financial returns data particularly benefits from NMF's constraint structure.

**3. Non-Deterministic Nature of NMF**
This question addressed the important characteristic that NMF can produce different outputs depending on its initialization. Unlike deterministic methods such as SVD, NMF uses iterative optimization that can converge to different local optima based on the random initialization of the factor matrices. This makes multiple runs with different initializations a common practice in NMF applications.

**4. Sparse Matrix Representation**
You demonstrated understanding of how sparse matrices are represented in NMF contexts. The correct sparse representation using the format [[2,0,0,0], [0,3,0,0], [0,0,0,1], [0,4,0,2]] shows how most entries are zero, with only a few non-zero values positioned at specific locations. This sparse structure is common in many NMF applications, particularly in text analysis and collaborative filtering.

**5. Practical Application: Pairwise Distance in NMF**
This question tested your understanding of how to evaluate NMF reconstructions in practice. The pairwise distance calculation between the NMF-encoded version of the original dataset and the encoded query dataset serves as a similarity metric. This helps determine which existing data points are most similar to a new query point, which is particularly useful in recommendation systems and information retrieval tasks.

### Ungraded Practice Quiz Topics

The practice assessment focused on three fundamental concepts that distinguish NMF from other dimensionality reduction methods:

**1. Distinguishing NMF from PCA**
The key difference lies in the constraint that NMF requires only positive values in the input matrix. This fundamental distinction shapes how the two methods operate. While PCA can handle any real-valued data and produces orthogonal components that can have negative values, NMF's non-negativity constraint means it's adding together different values rather than subtracting them. This additive property makes NMF particularly intuitive for applications where the data naturally represents quantities that cannot be negative.

**2. When to Choose PCA Over NMF**
You correctly identified that PCA excels when working with linear combinations of features. PCA's strength is in creating uncorrelated components through orthogonal transformations, which is ideal when you want to capture variance in the data through linear combinations that can include both positive and negative coefficients. This makes PCA more flexible for general-purpose dimensionality reduction, especially when dealing with data where negative values are meaningful.

**3. Ideal Applications for NMF**
The most suitable application for NMF is reconstructing text documents with learned topics, which represents the classic use case for NMF in natural language processing. In this scenario, NMF can decompose a document-term matrix into a document-topic matrix and a topic-term matrix. Each topic becomes an interpretable collection of words, and each document is represented as a combination of these topics. The non-negativity constraint ensures that topics are additive combinations of words, which aligns perfectly with how we understand documents as mixtures of themes.

---

## Incorrect Responses Analysis

Since you achieved perfect scores on both assessments, there were **no incorrect responses** to analyze. This demonstrates excellent comprehension of:

- The mathematical foundations of Non-Negative Matrix Factorization
- The practical applications and use cases for NMF
- How NMF compares to and differs from other dimensionality reduction techniques like PCA
- The technical considerations when implementing NMF (initialization, convergence, distance metrics)
- When to choose NMF versus alternative methods based on data characteristics