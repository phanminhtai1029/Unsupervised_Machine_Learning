# Coursera Quiz Results Summary - Module 1

## Overview Table

| Quiz Name | Total Questions | Correct Answers | Score | Status |
|-----------|----------------|-----------------|-------|--------|
| Module 1 Quiz (Graded) | 5 | 5 | 100% | ✓ Pass |
| Introduction to Unsupervised Learning (Ungraded) | 3 | 3 | 100% | ✓ Pass |
| K-Means Clustering (Ungraded) | 3 | 2 | 66.66% | ✓ Pass |

**Overall Performance**: 10 correct out of 11 total questions (90.9%)

---

## Topics Covered

### Module 1 Quiz - K-Means Clustering Fundamentals
This graded quiz covered essential concepts in K-means clustering, including:

- **Standard Deviation in Clustering**: Understanding how a small standard deviation indicates that data points are tightly clustered around their centroids, which is a key measure of cluster quality.

- **Elbow Method**: Learning how to interpret the inflection point in an elbow plot to determine the ideal number of clusters for your dataset.

- **Cluster Selection Metrics**: Understanding the role of inertia and distortion as measures of entropy that help evaluate clustering performance. The quiz emphasized using both metrics to select the optimal number of clusters.

- **Distortion vs. Inertia**: Grasping when to use each metric based on the similarity of points within clusters and the distribution of cluster sizes. The key insight is that distortion is preferred when point similarity matters more, while inertia is better when clusters have similar numbers of points.

- **Elbow Method Implementation**: Understanding that the elbow method works by plotting the interpreted variation as a function of the number of clusters and selecting the point where the curve bends.

### Introduction to Unsupervised Learning
This quiz focused on foundational concepts:

- **Purpose of Unsupervised Learning**: Understanding that unsupervised algorithms are most valuable when we don't have labeled outcomes and want to discover natural structures or patterns within our data by partitioning it into meaningful groups.

- **Curse of Dimensionality**: Learning that reducing the dimensionality of data is a practical solution to improve both performance and interpretability when working with high-dimensional datasets.

- **Dimension Reduction Applications**: Recognizing image tracking as a common real-world use case where dimension reduction helps extract primary factors from complex data.

### K-Means Clustering Algorithm
This quiz tested deeper understanding of the algorithm's mechanics:

- **Iterative Process**: Understanding that K-means works by repeatedly adjusting centroids to the mean of each cluster and reassigning points until convergence is reached (when centroids no longer move).

- **K-Means++ Initialization**: Learning about the smarter initialization method that picks the first centroid as an initial point and then selects subsequent centroids by prioritizing points that are farther from existing centroids, weighted by their distance. This approach helps avoid local optima that can occur with random initialization.

---

## Incorrect Responses Analysis

### Question with Incorrect Answer

**Quiz**: K-Means Clustering (Ungraded)  
**Question 3**: What happens with our second cluster centroid when we use the probability formula?

**Your Answer** (Incorrect): "When we use the probability formula, we put more weight on the lighter centroids, because it will take more computational power to draw our clusters. So, the second cluster centroid is likely going to be less distant."

**Why This Was Incorrect**: This answer misunderstands how the probability-weighted initialization works in K-means++. The statement incorrectly suggests that using the probability formula puts more weight on lighter centroids to save computational power, which is not the purpose or effect of this method.

**Correct Concept**: When using the probability formula in K-means++ initialization, we actually put less weight on points that are close to existing centroids. The probability of selecting a point as the next centroid is proportional to the square of its distance from the nearest existing centroid. This means points that are farther away from existing centroids have a higher probability of being selected. The goal is to spread out the initial centroids to get better coverage of the data space and avoid poor local minima, not to reduce computational power. Therefore, the second cluster centroid is more likely to be farther from the first centroid, not closer.