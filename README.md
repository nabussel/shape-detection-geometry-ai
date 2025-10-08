# Shape Detection and Geometric Classification

This project detects and classifies circles, squares, rectangles, and ellipses from noisy 2D point clouds.  
It combines geometric reasoning, PCA-based alignment, and distance-based residual fitting to identify shapes even when they are rotated, translated, or noisy.

---

## Overview

This repository contains a geometric analysis pipeline that learns structure from shape data using logic and mathematical modeling rather than traditional machine learning.  
It connects multiple fitting approaches together through a ratio-based framework, numerical optimization, and PCA transformations.

---

## Key Methods and Innovations

### Circle-to-Square Ratio

A custom constant links the radius of a fitted circle to the side length of an equivalent square.  
After fitting a circle with radius `r*`, the program estimates the square’s side `s*` using:


This constant was determined experimentally by generating synthetic squares and finding the ratio that minimized residual error.  
By converting the circle radius into a square length, the algorithm can reuse a stable circle fit to bootstrap a square model, then fine-tune its angle to minimize residuals.  
Implementation: `best_rotated_square()`.

---

### Ridge-Style Radius Search

Circle fitting is performed by scanning a tight radius range between the minimum and maximum distances of the data points from the origin.  
Each candidate radius is scored using:


This method allows robust radius estimation while optionally penalizing overly large circles through α-regularization.  
Implementation: `circle_ridge()`.

---

### PCA for Alignment

Principal Component Analysis (PCA) is used to find the primary orientation of the shape.  
The data is rotated to align with this axis and translated so the geometric center sits at the origin.  
This provides rotation- and translation-invariant inputs for all shape fitting routines.  
Implementation: `find_rotation_angle()`, `fit_circle()`, `cart_to_pol()`.

---

### Residual Minimization and Parameter Search

To find the best-fitting rectangle or square, the algorithm sweeps through small steps in parameters such as width, height, and rotation angle, recording total point-to-boundary residuals.  
This interpolated search identifies global minima in noisy data where closed-form fitting might fail.  
Implementation: `find_inner_rectangle()`, `find_outer_rectangle()`, `total_residual_1()`.

---

### Shape-Specific Residual Functions

Each shape uses a tailored residual definition:
- Circles measure point-to-circumference distance  
- Rectangles and squares measure the nearest edge or corner distance  
- Ellipses use distance to the nearest boundary point after alignment  

These metrics evaluate true geometric closeness, not just average error.  
Implementation: `total_residual()`, `closest_distance_to_ellipse()`.

---

### Radial Profile Comparison

The system treats every point cloud as a signal of distance versus angle.  
It then generates ideal reference signals for each shape type and compares them to the data using cosine similarity and Euclidean distance.  
This “radial signature” approach allows the system to recognize patterns such as the four peaks of a square or the smooth curvature of a circle.  
Implementation: `finding_similarity_of_graphs()`.

---

### Ambiguity Resolution

To prevent overconfident classification, the system checks aspect ratios:
- If an ellipse has nearly equal axes, it is labeled as circular.  
- If a rectangle’s sides are nearly equal, it is labeled as square.  

This ensures stable results when shapes are close in proportion.  
Implementation: `polygon_ellipse()` and `polygon_rectangle()`.

---

## Full Pipeline

1. Translate and rotate points using PCA and fitted centers.  
2. Fit a circle using ridge-style scanning.  
3. Convert the radius to a square side using the circle-to-square ratio.  
4. Optimize rotation and side length for the square.  
5. Perform rectangular and elliptical fits through residual minimization.  
6. Compare radial profiles to reference shapes.  
7. Choose the shape with the strongest similarity or smallest residual error.

---

## Supporting Tools

- Functions to generate random noisy circles, squares, rectangles, and ellipses.  
- Visualization functions that overlay the fitted shapes on the input points.  
- A randomized test harness that generates 100+ synthetic datasets to measure classification accuracy.

Example utilities:  
`generate_random_square_points()`, `generate_random_rectangle_with_noise()`, `generate_random_ellipse_points()`, `generate_circle_coordinates_shifted()`, `plot_shapes_with_rotation()`, `generate_points_rand_shape()`, `accuracy()`.

---

## Technical Details

Language: Python  
Libraries: NumPy, SciPy, scikit-learn, Matplotlib, Seaborn  
Key algorithms: PCA, ridge-style optimization, residual minimization, radial similarity scoring

---

## Why It Works

The system does not rely on pre-trained models or labeled data.  
It instead reasons from geometry, using analytical relationships, ratios, and numerical fitting.  
This makes it explainable, robust to noise, and easily extensible to new shapes.

---

## Future Improvements

- Vectorized residual scans for faster computation.  
- Closed-form ellipse distance approximations.  
- Adaptive calibration for the circle-to-square ratio.  
- 3D extensions to detect spheres, cylinders, and ellipsoids.

---

Created by Noah Bussell  
Indiana University Bloomington  
B.S. Data Science and Statistics


