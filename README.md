# Shape Detection & Geometric Classification (Noisy 2D Point Clouds)

Detects and classifies **circles, squares, rectangles, and ellipses** from noisy 2D point clouds.  
The pipeline blends **empirical geometry**, **PCA-based alignment**, **residual minimization**, and **signal-style similarity metrics** to make robust calls even with rotation, translation, and noise.

---

## ✨ What’s novel here

### 1) Circle→Square bridge via a custom empirical ratio
**Key idea:** turn the (easier, stabler) circle fit into a square estimate by a **data-driven ratio** between the fitted circle radius and the square side length.

- After estimating the best circle radius \( r^\* \) from the noisy points, I convert it to a square side \( s^\* \) using an **empirical constant**:
  \[
  s^\* \approx \frac{r^\*}{0.56985955}
  \]
- This constant came from sweeping many synthetic squares under the project’s **specific residual metric** and choosing the radius/side ratio that minimizes error on average.
- Result: I can **reuse circle regression** to bootstrap a strong square guess, then **refine by rotation search** to find the best-fitting square.

**Code touchpoints:**  
`best_rotated_square(...)` derives `length` from `radius_ridge/0.56985955` and then searches over rotation angles to minimize point-to-square residuals.

---

### 2) “Ridge-style” radius search for circles
**Key idea:** find a robust circle radius by scanning a **tight, data-driven radius range** and (optionally) adding a **size penalty**.

- Compute a point cloud’s **min/max radial distance** from the origin to get a reasonable radius window.
- For each candidate radius \( r \) in that window, minimize:
  \[
  \text{score}(r) = \sum_i \operatorname{dist}((x_i,y_i), \text{circle}(r)) \;+\; \alpha r^2
  \]
  where \( \alpha \) is a tunable regularization (acts like ridge).
- Even with \( \alpha=0 \) (the default used in the script), the framework supports **penalizing oversized radii** to curb outlier pull.

**Why it helps:** Robust radius estimation stabilizes everything downstream (square/rectangle/ellipse comparisons).

**Code touchpoints:**  
`find_radius_range(...)`, `distance_to_radius(...)`, and `circle_ridge(points, alpha)`.

---

### 3) PCA for rotation alignment (and making translation reliable)
**Key idea:** use **PCA** to estimate the dominant orientation, rotate data to an axis-aligned frame, then do fitting/search **in that simpler frame**.

- Compute PCA on the point cloud; take the first component’s direction to get the **rotation angle**.
- Rotate points by the **negative** of that angle so rectangles/squares become **axis-aligned** (now width/height search is meaningful).
- For translation, fit a **circle** or **ellipse** to infer the **center**; translate the point cloud so that center is at the origin.  
  > PCA makes rotation correct; **fitted centers** make translation correct. Together, this yields consistent, rotation- and shift-invariant measurements.

**Code touchpoints:**  
`find_rotation_angle(...)` (PCA), `find_best_rectangle(...)` (rotate to axis, then sweep sizes), `fit_circle(...)` and `fit_ellipse(...)` + `cart_to_pol(...)` (centers for shifting).

---

### 4) Fine-resolution residual scans (quasi-interpolation)
**Key idea:** instead of single closed-form fits everywhere, **sweep** parameters at fine steps (0.01 units for widths/heights; 0.1° for angles) and track the **curvature of the residual curve** to land near the global minimum.

- For rectangles: find inner/outer bounds, then **scan** width/height pairs, computing **sum of distances to the rectangle boundary**. Keep the lowest residual.
- For squares: scan **rotation** (0→360° by 0.1°) after the circle→square side estimate.
- This dense search acts like a **numerical interpolation** of the error landscape: it’s slower than a closed form, but **robust under noise** and shape ambiguity.

**Code touchpoints:**  
`find_inner_rectangle(...)`, `find_outer_rectangle(...)`, `total_residual_1(...)`, and the angle loop in `best_rotated_square(...)`.

---

### 5) Distance-to-boundary residuals tailored per shape
**Key idea:** define **shape-specific** residuals that measure what matters:

- **Circle:** distance from each point to the closest point on the circle of radius \( r \).
- **Square/Rectangle:** distance to the **nearest edge/corner** (with piecewise logic for inside/outside/corner cases).
- **Ellipse:** minimize distance to the nearest point on an **axis-aligned ellipse** after rotating the data into that ellipse’s frame.

These residuals reflect **geometric fit** rather than just centroid/variance differences.

**Code touchpoints:**  
`total_residual(...)` (square), `total_residual_1(...)` (rectangle), `closest_distance_to_ellipse(...)` (ellipse), circle distances via `distance_to_radius(...)`.

---

### 6) Signal-style comparison of radial profiles
**Key idea:** treat each point cloud (after alignment/centering) as a **radial signal** \( d(\theta) \) and compare it to the **ideal radial signal** of each candidate shape.

- Compute angle for each point \( \theta_i = \operatorname{atan2}(y_i, x_i) \).
- Build **true radial profile** \( d_i = \sqrt{x_i^2 + y_i^2} \).
- Build **ideal** radial profile per shape at the same angles:
  - Circle: constant \( r \)
  - Square/Rectangle: ray/edge intersection distances
  - Ellipse: \(\sqrt{(a\cos\theta)^2 + (b\sin\theta)^2}\)
- Compare vectors with **cosine similarity** and **Euclidean distance**.  
  This detects **shape-like periodic signatures** (e.g., square’s four “lobes”) even when coordinates are noisy.

**Code touchpoints:**  
`gen_pnts_for_circle(...)`, `gen_pnts_for_rect(...)`, `gen_pnts_for_ellipse(...)`, `finding_similarity_of_graphs(...)`.

---

### 7) Ambiguity gating (near-square vs near-rectangle, near-circle vs near-ellipse)
**Key idea:** if aspect ratio \( \max/\min \) is close to 1 (e.g., < 1.1), treat an ellipse as a **circle-like** candidate or a rectangle as **square-like** to avoid over-confident mislabels.

**Code touchpoints:**  
Aspect-ratio checks in `polygon_ellipse(...)` and `polygon_rectangle(...)`.

---

## 🔬 Method in practice (end-to-end)

1. **Preprocess**
   - Optionally shift by fitted **circle/ellipse center** \((x_0,y_0)\) so origin is meaningful.
   - Use **PCA** to get an orientation angle; rotate to an axis frame.

2. **Circle fit**
   - Scan radius in \([r_{\min}, r_{\max}]\) with optional ridge penalty; take \( r^\* \).

3. **Square via circle**
   - Convert \( r^\* \to s^\* \) with the **empirical ratio**.
   - Sweep **rotation** to minimize boundary residuals; record best angle and error.

4. **Rectangle fit**
   - With rotated data, determine **outer/inner bounds**.
   - **Grid search** width/height with residuals; record best pair.

5. **Ellipse fit**
   - Fit algebraic ellipse (Halír–Flusser method), convert to geometric params \((x_0,y_0,a,b,\phi)\).
   - In ellipse-aligned frame, sum distances to the nearest ellipse point.

6. **Radial signal comparison**
   - Build radial profiles at the **same angles** for: circle, square, rectangle, ellipse.
   - Compute **cosine similarity** + **Euclidean distance** for each.

7. **Decision & ambiguity control**
   - Pick the shape with the **best similarity**/**lowest distance**,  
     while applying **aspect-ratio gates** to avoid false specificity.

---

## 🧪 Synthetic data, visualization, and self-tests

- **Data generators** for square/rectangle/circle/ellipse with **noise, rotation, and shifts**.
- **Overlay plots** of fitted shapes on the point cloud for sanity checks.
- **Randomized test harness** (100+ trials) to approximate accuracy across diverse cases.

**Code touchpoints:**  
`generate_random_square_points(...)`, `generate_random_rectangle_with_noise(...)`, `generate_random_ellipse_points(...)`, `generate_circle_coordinates_shifted(...)`, `plot_shapes_with_rotation(...)`, `generate_points_rand_shape(...)`, `accuracy(...)`.

---

## 🛠️ Tech & dependencies

- Python, NumPy, SciPy (optimize), scikit-learn (PCA), Matplotlib (and Seaborn for optional plots).

---

## 🚀 Quick start

```python
# 1) Generate or load points
points = generate_random_square_points(square_size=5, num_points_per_side=12, noise=0.2,
                                       x_shift=0, y_shift=0, theta=30)

# 2) Run shape scorers (returns residuals/similarities + disambiguation flags)
square_vals, circle_vals, ellipse_vals, rect_vals = find_polygon_values(points)

# 3) Choose the best shape using the project’s scorer
best_shape, score = give_best_shape_plus_sim([square_vals, circle_vals, ellipse_vals, rect_vals])
print(best_shape, score)
