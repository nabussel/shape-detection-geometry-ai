# -*- coding: utf-8 -*-
"""
Shape Classification from Noisy 2D Perimeter Point Clouds
via Multi-Hypothesis Fitting and Derivative-Based Straightness

Authors: Noah Bussell, Emily Ahern

This script implements a deterministic multi-hypothesis pipeline that classifies
noisy 2D point clouds (given only as (x,y) coordinate pairs) into one of four
shape families: circle, ellipse, square, or rectangle.

The classifier fits all four candidate models to every input and selects the
class with the lowest adjusted residual score. An auxiliary derivative-based
straightness score helps discriminate curved from rectilinear boundaries.

A Random Forest baseline trained on engineered geometric features is included
for comparison purposes.
"""

import numpy as np
import random
import math
import json
import pickle
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.spatial import ConvexHull
from collections import Counter, defaultdict


# ============================================================================
# STRAIGHTNESS AND CURVATURE ANALYSIS
# ============================================================================

def analyze_edge_derivatives(points, num_sections=4):
    """
    Analyze boundary straightness by measuring derivative consistency within sections.

    The point cloud is divided into equal sections. Within each section, finite-difference
    slopes between consecutive points are computed. A straight edge produces nearly constant
    slopes (low coefficient of variation), while a curved edge produces varying slopes
    (high coefficient of variation).

    Args:
        points: List of (x, y) coordinates, ordered around the boundary.
        num_sections: Number of sections to divide the perimeter into (default 4).

    Returns:
        dict with 'straightness_score' in [0, 1]. Higher means straighter edges.
    """
    if len(points) < 10:
        return {'straightness_score': 0.0}

    section_size = len(points) // num_sections
    if section_size < 5:
        section_size = max(len(points) // 2, 5)
        num_sections = max(len(points) // section_size, 2)

    section_scores = []

    for section_idx in range(num_sections):
        start_idx = section_idx * section_size
        end_idx = start_idx + section_size if section_idx < num_sections - 1 else len(points)
        section_points = points[start_idx:end_idx]

        if len(section_points) < 4:
            continue

        slopes = []
        is_vertical = []
        is_horizontal = []

        for i in range(len(section_points) - 1):
            p1 = section_points[i]
            p2 = section_points[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            if abs(dx) < 1e-6:
                is_vertical.append(True)
                is_horizontal.append(False)
                slopes.append(None)
            elif abs(dy) < 1e-6:
                is_vertical.append(False)
                is_horizontal.append(True)
                slopes.append(None)
            else:
                is_vertical.append(False)
                is_horizontal.append(False)
                slopes.append(dy / dx)

        num_vertical = sum(is_vertical)
        num_horizontal = sum(is_horizontal)
        num_normal = len([s for s in slopes if s is not None])
        total = len(slopes)

        if total == 0:
            continue

        if num_vertical / total > 0.8:
            section_score = 1.0
        elif num_horizontal / total > 0.8:
            section_score = 1.0
        elif num_normal / total > 0.8:
            valid_slopes = [s for s in slopes if s is not None]
            if len(valid_slopes) < 2:
                section_score = 1.0
            else:
                mean_slope = sum(valid_slopes) / len(valid_slopes)
                variance = sum((s - mean_slope)**2 for s in valid_slopes) / len(valid_slopes)
                std_dev = math.sqrt(variance)
                cv = std_dev / (abs(mean_slope) + 0.01)

                if cv < 0.5:
                    section_score = 1.0
                elif cv > 2.0:
                    section_score = 0.0
                else:
                    section_score = 1.0 - (cv - 0.5) / 1.5
        else:
            section_score = 0.0

        section_scores.append(section_score)

    if not section_scores:
        return {'straightness_score': 0.0}

    return {'straightness_score': sum(section_scores) / len(section_scores)}


def test_curvature_consistency(points):
    """
    Detect sharp corners by measuring direction changes along the boundary.

    Computes the direction angle between each consecutive pair of points and counts
    direction changes exceeding 60 degrees. Squares and rectangles produce ~4 such
    corners; circles and ellipses produce ~0.

    Args:
        points: List of (x, y) coordinates.

    Returns:
        Corner score in [0, 1]. Score of 1.0 means 4 or more sharp corners detected.
    """
    if len(points) < 5:
        return 0.0

    angles = []
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angles.append(math.atan2(dy, dx))

    sharp_corners = 0
    for i in range(len(angles)):
        angle_change = angles[(i + 1) % len(angles)] - angles[i]
        while angle_change > math.pi:
            angle_change -= 2 * math.pi
        while angle_change < -math.pi:
            angle_change += 2 * math.pi
        if abs(angle_change) > math.pi / 3:
            sharp_corners += 1

    return min(sharp_corners / 4.0, 1.0)


def compute_shape_metrics(points):
    """
    Compute straightness and corner metrics for a point cloud.

    Args:
        points: List of (x, y) coordinates.

    Returns:
        dict with 'straightness', 'corner_score', and 'is_likely_straight_edged'.
    """
    derivative_metrics = analyze_edge_derivatives(points, num_sections=4)
    straightness = derivative_metrics['straightness_score']
    corner_score = test_curvature_consistency(points)
    is_likely_straight = (straightness > 0.5) or (corner_score > 0.5)

    return {
        'straightness': straightness,
        'corner_score': corner_score,
        'is_likely_straight_edged': is_likely_straight
    }


# ============================================================================
# SHAPE GENERATION FUNCTIONS
# ============================================================================

def generate_random_ellipse_points(num_points, rad_a, rad_b, noise, center_x=0, center_y=0, angle=0):
    """
    Generate noisy points on an ellipse perimeter.

    Args:
        num_points: Number of points to sample.
        rad_a: Semi-axis in the x-direction (before rotation).
        rad_b: Semi-axis in the y-direction (before rotation).
        noise: Noise level as a fraction of each axis radius.
        center_x, center_y: Center of the ellipse.
        angle: Rotation angle in degrees.

    Returns:
        Tuple of (clean_coordinates, noisy_rotated_coordinates).
    """
    coordinates = []
    coordinates_noisy = []

    for i in range(num_points):
        theta = 2 * math.pi * i / num_points
        x = rad_a * math.cos(theta) + center_x
        y = rad_b * math.sin(theta) + center_y
        noise_x = random.uniform(-rad_a * noise, rad_a * noise)
        noise_y = random.uniform(-rad_b * noise, rad_b * noise)
        coordinates.append((x, y))
        coordinates_noisy.append((x + noise_x, y + noise_y))

    angle_rad = np.radians(angle)
    coordinates_noisy_rotated = []
    for x, y in coordinates_noisy:
        x_rel = x - center_x
        y_rel = y - center_y
        rotated_x = x_rel * np.cos(angle_rad) - y_rel * np.sin(angle_rad) + center_x
        rotated_y = x_rel * np.sin(angle_rad) + y_rel * np.cos(angle_rad) + center_y
        coordinates_noisy_rotated.append((rotated_x, rotated_y))

    return coordinates, coordinates_noisy_rotated


def generate_circle_coordinates_shifted(radius, num_points, noise, cx=0, cy=0):
    """
    Generate noisy points on a circle perimeter.

    Args:
        radius: Circle radius.
        num_points: Number of points to sample.
        noise: Noise level as a fraction of the radius.
        cx, cy: Center coordinates.

    Returns:
        Tuple of (clean_coordinates, noisy_coordinates).
    """
    coordinates = []
    coordinates_noisy = []

    for i in range(num_points):
        theta = 2 * math.pi * i / num_points
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        noise_angle = random.uniform(0, 2 * math.pi)
        noise_magnitude = random.uniform(0, radius * noise)
        noisy_x = x + noise_magnitude * math.cos(noise_angle)
        noisy_y = y + noise_magnitude * math.sin(noise_angle)
        coordinates.append((x, y))
        coordinates_noisy.append((noisy_x, noisy_y))

    return coordinates, coordinates_noisy


def generate_random_square_points(square_size, num_points_per_side, noise, x_shift, y_shift, theta):
    """
    Generate noisy points on a square perimeter.

    Points are uniformly distributed along each of the four sides, then
    noise is added and the result is rotated and shifted.

    Args:
        square_size: Side length of the square.
        num_points_per_side: Points per side (total = 4 * this).
        noise: Absolute noise magnitude added to each coordinate.
        x_shift, y_shift: Translation applied after rotation.
        theta: Rotation angle in degrees.

    Returns:
        List of (x, y) noisy, rotated, shifted coordinates.
    """
    half_size = square_size / 2
    sides = {
        "bottom": [(-half_size, -half_size), (half_size, -half_size)],
        "right": [(half_size, -half_size), (half_size, half_size)],
        "top": [(half_size, half_size), (-half_size, half_size)],
        "left": [(-half_size, half_size), (-half_size, -half_size)]
    }
    points = []
    for side, (start, end) in sides.items():
        x1, y1 = start
        x2, y2 = end
        x = np.linspace(x1, x2, num_points_per_side)
        y = np.linspace(y1, y2, num_points_per_side)
        noise_x = x + np.random.uniform(-noise, noise, num_points_per_side)
        noise_y = y + np.random.uniform(-noise, noise, num_points_per_side)
        angle = np.radians(theta)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        points.extend(list(zip(cos_a * noise_x - sin_a * noise_y,
                               sin_a * noise_x + cos_a * noise_y)))

    points_shifted = [(x + x_shift, y + y_shift) for x, y in points]
    return points_shifted


def generate_random_rectangle_with_noise(num_points, noise_level, width, height, center_x=0, center_y=0, tha=0):
    """
    Generate noisy points on a rectangle perimeter with proportional point distribution.

    Points are distributed proportionally to side length, so longer sides get more points.

    Args:
        num_points: Total number of points.
        noise_level: Absolute noise magnitude.
        width, height: Rectangle dimensions.
        center_x, center_y: Center of the rectangle.
        tha: Rotation angle in degrees.

    Returns:
        Tuple of (noisy_x_list, noisy_y_list).
    """
    x_min = center_x - width / 2
    y_min = center_y - height / 2

    points = []
    perimeter = 2 * (width + height)

    num_bottom = math.floor(num_points * (width / perimeter))
    x_bottom = np.linspace(x_min, x_min + width, num_bottom)
    y_bottom = np.full_like(x_bottom, y_min)
    points.extend(list(zip(x_bottom, y_bottom)))

    num_right = math.floor(num_points * (height / perimeter))
    y_right = np.linspace(y_min, y_min + height, num_right)
    x_right = np.full_like(y_right, x_min + width)
    points.extend(list(zip(x_right, y_right)))

    num_top = math.floor(num_points * (width / perimeter))
    x_top = np.linspace(x_min + width, x_min, num_top)
    y_top = np.full_like(x_top, y_min + height)
    points.extend(list(zip(x_top, y_top)))

    num_left = math.floor(num_points * (height / perimeter))
    y_left = np.linspace(y_min + height, y_min, num_left)
    x_left = np.full_like(y_left, x_min)
    points.extend(list(zip(x_left, y_left)))

    noisy_x = [x + np.random.uniform(-noise_level, noise_level) for x, y in points]
    noisy_y = [y + np.random.uniform(-noise_level, noise_level) for x, y in points]

    angle = np.radians(tha)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    noisy_x = np.array(noisy_x)
    noisy_y = np.array(noisy_y)

    rot_x = cos_a * noisy_x - sin_a * noisy_y
    rot_y = sin_a * noisy_x + cos_a * noisy_y

    return [round(x, 2) for x in rot_x], [round(y, 2) for y in rot_y]


# ============================================================================
# CIRCLE FITTING (Kasa Algebraic Least Squares)
# ============================================================================

def fit_circle(x, y):
    """
    Fit a circle using the Kasa algebraic least-squares method.

    Solves the linear system arising from expanding (x-a)^2 + (y-b)^2 = r^2
    into Dx + Ey + F = -(x^2 + y^2), then recovering center and radius.

    Args:
        x, y: Lists or arrays of point coordinates.

    Returns:
        Tuple (x0, y0, r): center coordinates and radius.
    """
    x = np.array(x)
    y = np.array(y)
    A = np.c_[-2*x, -2*y, np.ones_like(x)]
    x0, y0, b = np.linalg.solve(A.T @ A, A.T @ (-(x**2) - (y**2)))
    r = np.sqrt(x0**2 + y0**2 - b)
    return x0, y0, r


# ============================================================================
# ELLIPSE FITTING (Fitzgibbon-Pilu-Fisher Direct Least Squares)
# ============================================================================

def fit_ellipse(x, y):
    """
    Fit an ellipse using the Fitzgibbon-Pilu-Fisher direct least-squares method.

    Minimizes algebraic distance subject to the ellipticity constraint 4ac - b^2 = 1,
    solved via a generalized eigenvalue problem.

    Args:
        x, y: Arrays of point coordinates.

    Returns:
        Array of conic coefficients [a, b, c, d, e, f].
    """
    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0] * eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()


def cart_to_pol(coeffs):
    """
    Convert conic coefficients to geometric ellipse parameters.

    Takes the 6 conic coefficients (a, b, c, d, e, f) from ax^2+bxy+cy^2+dx+ey+f=0
    and extracts center, semi-axes, eccentricity, and rotation angle.

    Args:
        coeffs: Array of 6 conic coefficients.

    Returns:
        Tuple (x0, y0, ap, bp, e, phi): center, semi-major axis, semi-minor axis,
        eccentricity, and rotation angle.
    """
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError('Coefficients do not represent an ellipse: b^2 - 4ac must be negative.')

    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)

    val1 = num / den / (fac - a - c)
    val2 = num / den / (-fac - a - c)

    if val1 < 0 or val2 < 0:
        val1 = abs(val1)
        val2 = abs(val2)

    ap = np.sqrt(val1)
    bp = np.sqrt(val2)

    if np.iscomplex(ap) or np.iscomplex(bp):
        ap = np.real(ap)
        bp = np.real(bp)

    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        phi += np.pi/2

    if np.iscomplex(phi):
        phi = np.real(phi)

    phi = float(np.real(phi)) % np.pi

    return x0, y0, ap, bp, e, phi


# ============================================================================
# ELLIPSE DISTANCE COMPUTATION
# ============================================================================

def closest_distance_to_ellipse(px, py, rad_a, rad_b):
    """
    Compute the shortest distance from a point to an axis-aligned ellipse centered at the origin.

    Uses scalar numerical minimization over the parametric angle theta.

    Args:
        px, py: Point coordinates.
        rad_a, rad_b: Semi-axes of the ellipse.

    Returns:
        Euclidean distance from (px, py) to the nearest point on the ellipse.
    """
    def objective(theta):
        ellipse_x = rad_a * math.cos(theta)
        ellipse_y = rad_b * math.sin(theta)
        return (px - ellipse_x)**2 + (py - ellipse_y)**2

    initial_guess = math.atan2(py, px)
    result = minimize(objective, initial_guess)
    closest_theta = result.x[0]
    closest_x = rad_a * math.cos(closest_theta)
    closest_y = rad_b * math.sin(closest_theta)
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)


# ============================================================================
# SQUARE FITTING (Circle-Based Radius Estimation + Rotation Search)
# ============================================================================

# Empirical constant: ratio of circle-fit radius to true side length when a circle
# is fit to points uniformly sampled on a square perimeter. Determined by fitting
# circles (Kasa method) to 1000+ known squares of varying sizes and recording
# radius/side_length. The 95% confidence interval was stable to the third decimal.
RADIUS_TO_SIDE_FACTOR = 0.56985955


def distance_to_radius(point, radius):
    """
    Compute the shortest distance from a 2D point to a circle of given radius centered at the origin.

    Args:
        point: Tuple (x, y).
        radius: Circle radius.

    Returns:
        Distance from the point to the nearest point on the circle.
    """
    x, y = point
    dist_from_origin = math.sqrt(x**2 + y**2)
    return abs(dist_from_origin - radius)


def find_radius_range(noisy_points):
    """
    Find the min and max distances from the origin across all points.

    Args:
        noisy_points: List of (x, y) coordinates.

    Returns:
        Tuple (min_distance, max_distance).
    """
    distances = [math.sqrt(x**2 + y**2) for x, y in noisy_points]
    return min(distances), max(distances)


def circle_radius_sweep(noisy_points):
    """
    Find the best-fit radius by sweeping over candidate radii.

    Sweeps from the minimum to maximum point distance in steps of 0.001,
    selecting the radius that minimizes the total absolute radial deviation.

    Args:
        noisy_points: List of (x, y) coordinates centered at the origin.

    Returns:
        Tuple (best_radius, best_total_residual).
    """
    min_radius, max_radius = find_radius_range(noisy_points)
    smallest_radius = float('inf')
    smallest_sum = float('inf')

    r = min_radius
    while r < max_radius:
        total = sum(distance_to_radius(p, r) for p in noisy_points)
        if total < smallest_sum:
            smallest_radius = r
            smallest_sum = total
        r += 0.001

    return smallest_radius, smallest_sum


def total_residual_square(length, points):
    """
    Compute the total point-to-square-boundary distance for an axis-aligned square.

    The square is centered at the origin with side length `length`. Handles all
    cases: points outside each side, outside each corner, and inside the square.

    Args:
        length: Side length of the square.
        points: List of (x, y) coordinates.

    Returns:
        Total residual (sum of distances).
    """
    h = length / 2
    residual = 0
    for x, y in points:
        if y >= h and -h <= x <= h:
            residual += abs(y - h)
        elif y <= -h and -h <= x <= h:
            residual += abs(-h - y)
        elif x >= h and -h <= y <= h:
            residual += abs(x - h)
        elif x <= -h and -h <= y <= h:
            residual += abs(-h - x)
        elif x <= -h and y >= h:
            residual += math.sqrt((x + h)**2 + (y - h)**2)
        elif x >= h and y >= h:
            residual += math.sqrt((x - h)**2 + (y - h)**2)
        elif x <= -h and y <= -h:
            residual += math.sqrt((x + h)**2 + (y + h)**2)
        elif x >= h and y <= -h:
            residual += math.sqrt((x - h)**2 + (y + h)**2)
        else:
            residual += min(abs(x + h), abs(x - h), abs(y + h), abs(y - h))
    return residual


def best_rotated_square(points):
    """
    Fit a square to centered point data.

    1. Sweep radii to find the best-fit circle radius on the centered data.
    2. Convert that radius to a side length using the empirical factor.
    3. Sweep rotation angles (0 to 360 in 0.1 degree steps) to find the angle
       that minimizes the point-to-square residual.

    Args:
        points: List of (x, y) coordinates, assumed centered near the origin.

    Returns:
        Tuple (best_angle_degrees, side_length, smallest_residual).
    """
    noise_x = np.array([x for x, y in points])
    noise_y = np.array([y for x, y in points])
    coordinates = list(zip(noise_x, noise_y))

    radius, _ = circle_radius_sweep(coordinates)
    length = radius / RADIUS_TO_SIDE_FACTOR

    smallest_error = float('inf')
    best_angle = 0
    angle = 0
    while angle < 360:
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotated = list(zip(cos_a * noise_x - sin_a * noise_y,
                           sin_a * noise_x + cos_a * noise_y))
        residual = total_residual_square(length, rotated)
        if residual < smallest_error:
            smallest_error = residual
            best_angle = angle
        angle += 0.1

    return best_angle, length, smallest_error


# ============================================================================
# RECTANGLE FITTING (PCA-Initialized Nelder-Mead Optimization)
# ============================================================================

def find_rotation_angle(x_cord, y_cord):
    """
    Estimate the principal axis orientation using PCA.

    The leading eigenvector of the covariance matrix gives the direction of
    maximum variance, which serves as the initial rectangle orientation.

    Args:
        x_cord, y_cord: Lists of point coordinates.

    Returns:
        Angle in degrees of the first principal component.
    """
    try:
        x_cord = [float(np.real(x)) for x in x_cord]
        y_cord = [float(np.real(y)) for y in y_cord]
    except:
        pass

    data = np.vstack((x_cord, y_cord)).T
    data = np.real(data).astype(float)
    pca = PCA(n_components=2)
    pca.fit(data)
    angle = np.arctan2(pca.components_[0][1], pca.components_[0][0])
    return float(np.real(np.degrees(angle)))


def find_best_rectangle(x_cord, y_cord):
    """
    Fit a rectangle using Nelder-Mead optimization with PCA-based initialization.

    The objective minimizes mean squared point-to-rectangle distance over width,
    height, and rotation angle. Multiple starting angles are tried to handle
    rotational ambiguity.

    Args:
        x_cord, y_cord: Lists of point coordinates (centered on the shape).

    Returns:
        Tuple (width, height, residual, angle_radians).
    """
    noisy_x = np.array(x_cord)
    noisy_y = np.array(y_cord)

    angle_deg_init = find_rotation_angle(x_cord, y_cord)
    angle_rad_init = -np.radians(angle_deg_init)

    def objective(params):
        base, height, angle = params
        if base <= 0 or height <= 0:
            return 1e10

        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_x = cos_a * noisy_x - sin_a * noisy_y
        rot_y = sin_a * noisy_x + cos_a * noisy_y

        half_b, half_h = base/2, height/2
        total_dist_sq = 0

        for x, y in zip(rot_x, rot_y):
            if -half_b <= x <= half_b and -half_h <= y <= half_h:
                dist = min(abs(x - half_b), abs(x + half_b),
                          abs(y - half_h), abs(y + half_h))
            elif x < -half_b and y < -half_h:
                dist = np.sqrt((x + half_b)**2 + (y + half_h)**2)
            elif x > half_b and y < -half_h:
                dist = np.sqrt((x - half_b)**2 + (y + half_h)**2)
            elif x < -half_b and y > half_h:
                dist = np.sqrt((x + half_b)**2 + (y - half_h)**2)
            elif x > half_b and y > half_h:
                dist = np.sqrt((x - half_b)**2 + (y - half_h)**2)
            elif x < -half_b:
                dist = abs(x + half_b)
            elif x > half_b:
                dist = abs(x - half_b)
            elif y < -half_h:
                dist = abs(y + half_h)
            else:
                dist = abs(y - half_h)
            total_dist_sq += dist ** 2

        return total_dist_sq / len(rot_x)

    # Initial dimensions from PCA
    cos_a = np.cos(angle_rad_init)
    sin_a = np.sin(angle_rad_init)
    rotated_x = cos_a * noisy_x - sin_a * noisy_y
    rotated_y = sin_a * noisy_x + cos_a * noisy_y
    init_base = np.max(rotated_x) - np.min(rotated_x)
    init_height = np.max(rotated_y) - np.min(rotated_y)

    # Multi-start: more starting angles for near-square shapes
    aspect_ratio = max(init_base, init_height) / min(init_base, init_height)
    if aspect_ratio < 1.5:
        start_angles = [angle_rad_init, angle_rad_init + np.pi/4,
                       angle_rad_init + np.pi/2, angle_rad_init + 3*np.pi/4]
    else:
        start_angles = [angle_rad_init, angle_rad_init + np.pi/2]

    best_result = None
    best_score = float('inf')

    for start_angle in start_angles:
        result = minimize(objective, [init_base, init_height, start_angle],
                         method='Nelder-Mead',
                         options={'xatol': 0.01, 'fatol': 0.001, 'maxiter': 300})
        if result.fun < best_score:
            best_score = result.fun
            best_result = result

    best_base, best_height, best_angle = best_result.x

    if best_base < best_height:
        best_base, best_height = best_height, best_base
        best_angle += np.pi/2

    while best_angle > np.pi:
        best_angle -= 2*np.pi
    while best_angle < -np.pi:
        best_angle += 2*np.pi

    residual = np.sqrt(best_result.fun) * len(noisy_x)
    return best_base, best_height, residual, best_angle


# ============================================================================
# MULTI-HYPOTHESIS SCORING AND CLASSIFICATION
# ============================================================================

def find_polygon_values(points):
    """
    Fit all four shape models and compute adjusted residual scores.

    For each candidate shape (circle, ellipse, square, rectangle), computes the
    raw residual and applies a straightness/curvature bonus proportional to the
    shape's size and the observed boundary character.

    Args:
        points: List of (x, y) coordinates (the only input to the classifier).

    Returns:
        List of [shape_name, [adjusted_score, raw_residual, straightness]] for each shape.
    """
    try:
        x = [float(np.real(xi)) for xi, yi in points]
        y = [float(np.real(yi)) for xi, yi in points]
    except:
        x = [xi for xi, yi in points]
        y = [yi for xi, yi in points]

    try:
        # Fit all four models
        coeffs = fit_ellipse(np.array(x), np.array(y))
        x0_e, y0_e, rad_a, rad_b, e, angle_e = cart_to_pol(coeffs)
        x0_c, y0_c, rad_c = fit_circle(x, y)

        # Center data for square fitting (around circle center)
        x_shift_c = [xi - x0_c for xi in x]
        y_shift_c = [yi - y0_c for yi in y]
        points_shifted_c = list(zip(x_shift_c, y_shift_c))

        # Center data for rectangle fitting (around ellipse center)
        x_shift_e = [xi - x0_e for xi in x]
        y_shift_e = [yi - y0_e for yi in y]

        angle_s, length, error_s = best_rotated_square(points_shifted_c)
        base, height, resid_r, angle_r = find_best_rectangle(x_shift_e, y_shift_e)

        # Compute straightness metrics
        metrics = compute_shape_metrics(points)
        straightness = metrics['straightness']
        curvature_score = 1.0 - straightness
        straight_score = straightness
        WEIGHT = 0.3

        def polygon_square():
            residual = error_s
            bonus = straight_score * length * WEIGHT
            return [residual - bonus, residual, straightness]

        def polygon_circle():
            residual = sum(abs(math.sqrt(xi**2 + yi**2) - rad_c)
                          for xi, yi in points_shifted_c)
            bonus = curvature_score * rad_c * WEIGHT
            return [residual - bonus, residual, straightness]

        def polygon_ellipse():
            try:
                x_rot = [xi * math.cos(-angle_e) - yi * math.sin(-angle_e) for xi, yi in zip(x_shift_e, y_shift_e)]
                y_rot = [xi * math.sin(-angle_e) + yi * math.cos(-angle_e) for xi, yi in zip(x_shift_e, y_shift_e)]
                residual = sum(closest_distance_to_ellipse(xi, yi, rad_a, rad_b)
                              for xi, yi in zip(x_rot, y_rot))
                bonus = curvature_score * (rad_a + rad_b) / 2 * WEIGHT
                return [residual - bonus, residual, straightness]
            except:
                return [float('inf'), float('inf'), straightness]

        def polygon_rectangle():
            try:
                residual = resid_r
                bonus = straight_score * (base + height) / 2 * WEIGHT
                return [residual - bonus, residual, straightness]
            except:
                return [float('inf'), float('inf'), straightness]

        return [
            ["square", polygon_square()],
            ["circle", polygon_circle()],
            ["ellipse", polygon_ellipse()],
            ["rectangle", polygon_rectangle()]
        ]

    except Exception as ex:
        return [
            ["square", [float('inf'), float('inf'), 0]],
            ["circle", [float('inf'), float('inf'), 0]],
            ["ellipse", [float('inf'), float('inf'), 0]],
            ["rectangle", [float('inf'), float('inf'), 0]]
        ]


def classify_shape(polygon_values):
    """
    Select the shape with the lowest adjusted score.

    All four shapes compete equally; none are skipped.

    Args:
        polygon_values: Output of find_polygon_values.

    Returns:
        Tuple (best_shape_name, best_adjusted_score).
    """
    best_score = float('inf')
    best_shape = ""
    for shape, val in polygon_values:
        adjusted_score = val[0]
        if adjusted_score < best_score:
            best_score = adjusted_score
            best_shape = shape
    return best_shape, best_score


# ============================================================================
# FAMILY MATCHING (Evaluation Only)
# ============================================================================

def is_family_match(true_shape, predicted_shape):
    """
    Check if two shapes belong to the same geometric family.

    Curved family: {circle, ellipse}
    Rectilinear family: {square, rectangle}

    This is used only for evaluation, not during classification.

    Args:
        true_shape: Ground-truth shape label.
        predicted_shape: Predicted shape label.

    Returns:
        True if shapes are in the same family.
    """
    if true_shape == predicted_shape:
        return True
    if true_shape in {"circle", "ellipse"} and predicted_shape in {"circle", "ellipse"}:
        return True
    if true_shape in {"square", "rectangle"} and predicted_shape in {"square", "rectangle"}:
        return True
    return False


# ============================================================================
# DATA GENERATION AND TESTING
# ============================================================================

def generate_test_shape(noise_level):
    """
    Generate a single random shape with the given noise level.

    Randomly selects one of four shape types, generates random parameters,
    and returns the noisy point cloud with its ground-truth label.

    Args:
        noise_level: Noise intensity.

    Returns:
        Tuple (true_shape_label, point_list).
    """
    shape_type = random.randint(1, 4)
    rand_x_shift = random.uniform(-10, 10)
    rand_y_shift = random.uniform(-10, 10)
    rand_angle = random.uniform(-180, 180)

    if shape_type == 1:
        rand_length = random.uniform(3, 8)
        points = generate_random_square_points(rand_length, 12, noise_level,
                                               rand_x_shift, rand_y_shift, rand_angle)
        return "square", points

    elif shape_type == 2:
        rand_rad = random.uniform(3, 8)
        points = generate_circle_coordinates_shifted(rand_rad, 50, noise_level,
                                                     rand_x_shift, rand_y_shift)[1]
        return "circle", points

    elif shape_type == 3:
        rand_rad_a = random.uniform(3, 8)
        rand_rad_b = random.uniform(3, 8)
        points = generate_random_ellipse_points(50, rand_rad_a, rand_rad_b, noise_level,
                                                rand_x_shift, rand_y_shift, rand_angle)[1]
        if max(rand_rad_a, rand_rad_b) / min(rand_rad_a, rand_rad_b) < 1.1:
            return "circle", points
        else:
            return "ellipse", points

    else:
        rand_base = random.uniform(3, 8)
        rand_height = random.uniform(3, 8)
        x_r, y_r = generate_random_rectangle_with_noise(50, noise_level, rand_base, rand_height,
                                                        rand_x_shift, rand_y_shift, rand_angle)
        if max(rand_base, rand_height) / min(rand_base, rand_height) < 1.1:
            return "square", list(zip(x_r, y_r))
        else:
            return "rectangle", list(zip(x_r, y_r))


def accuracy_at_noise_level(noise_level, num_samples=100):
    """
    Test family-matching accuracy at a specific noise level.

    Args:
        noise_level: Noise intensity.
        num_samples: Number of shapes to test.

    Returns:
        dict with 'overall' accuracy, 'per_shape' accuracies, etc.
    """
    correct_d = {"square": [0, 0], "circle": [0, 0], "ellipse": [0, 0], "rectangle": [0, 0]}
    shape_points = [generate_test_shape(noise_level) for _ in range(num_samples)]
    shapes = [s for s, pts in shape_points]
    points_list = [pts for s, pts in shape_points]

    polygon_values = []
    for shape_data in points_list:
        try:
            polygon_values.append(find_polygon_values(shape_data))
        except:
            polygon_values.append([["square", [float('inf'), 0, 0]],
                                  ["circle", [float('inf'), 0, 0]],
                                  ["ellipse", [float('inf'), 0, 0]],
                                  ["rectangle", [float('inf'), 0, 0]]])

    predictions = [classify_shape(pv) for pv in polygon_values]

    correct = 0
    for i in range(len(predictions)):
        if is_family_match(shapes[i], predictions[i][0]):
            correct += 1
            correct_d[shapes[i]][0] += 1
        else:
            correct_d[shapes[i]][1] += 1

    per_shape = {}
    for name in ["square", "circle", "ellipse", "rectangle"]:
        total = correct_d[name][0] + correct_d[name][1]
        per_shape[name] = correct_d[name][0] / total if total > 0 else 0

    return {
        'overall': correct / len(shapes) if shapes else 0,
        'per_shape': per_shape,
        'total': len(shapes),
        'correct': correct
    }


def run_noise_analysis(noise_levels=None, samples_per_level=100):
    """
    Run accuracy analysis across multiple noise levels.

    Args:
        noise_levels: List of noise values to test.
        samples_per_level: Number of shapes per noise level.

    Returns:
        dict with arrays of accuracy results per noise level.
    """
    if noise_levels is None:
        noise_levels = [round(x * 0.02, 2) for x in range(0, 36)]

    print("=" * 70)
    print("NOISE ANALYSIS")
    print("=" * 70)
    print(f"Testing {len(noise_levels)} noise levels, {samples_per_level} samples each")
    print(f"Total: {len(noise_levels) * samples_per_level} tests\n")

    results = {
        'noise_levels': noise_levels,
        'overall_accuracy': [],
        'square_accuracy': [],
        'circle_accuracy': [],
        'ellipse_accuracy': [],
        'rectangle_accuracy': []
    }

    for noise in noise_levels:
        print(f"Noise {noise:.2f}...", end=' ')
        result = accuracy_at_noise_level(noise, num_samples=samples_per_level)
        results['overall_accuracy'].append(result['overall'])
        results['square_accuracy'].append(result['per_shape']['square'])
        results['circle_accuracy'].append(result['per_shape']['circle'])
        results['ellipse_accuracy'].append(result['per_shape']['ellipse'])
        results['rectangle_accuracy'].append(result['per_shape']['rectangle'])
        print(f"{result['overall']*100:.1f}%")

    return results


def plot_results(results):
    """Create accuracy vs. noise visualizations."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    noise_levels = results['noise_levels']

    ax1.plot(noise_levels, [a*100 for a in results['overall_accuracy']],
            'ko-', linewidth=2, markersize=8, label='Overall Accuracy')
    ax1.set_xlabel('Noise Level', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Family-Matching Accuracy vs Noise Level', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_ylim([0, 105])

    for name, marker, color in [('square', 's-', 'green'), ('circle', 'o-', 'blue'),
                                 ('ellipse', '^-', 'purple'), ('rectangle', 'D-', 'orange')]:
        ax2.plot(noise_levels, [a*100 for a in results[f'{name}_accuracy']],
                marker, linewidth=2, markersize=6, label=name.capitalize(), color=color)
    ax2.set_xlabel('Noise Level', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Per-Shape Accuracy vs Noise Level', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    ax2.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig('elbow_curve.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# RANDOM FOREST BASELINE
# ============================================================================

def extract_features(points):
    """
    Extract 13 geometric features from a point cloud for Random Forest classification.

    Features: bounding-box aspect ratio, CV of centroid distances, distance range ratio,
    mean/std/max direction changes, corner ratio, circularity, compactness, hull fill ratio,
    section straightness, radius section CV, corner count.

    Args:
        points: List of (x, y) coordinates or Nx2 array.

    Returns:
        List of 13 feature values.
    """
    points = np.array(points)
    x, y = points[:, 0], points[:, 1]
    n = len(points)

    cx, cy = np.mean(x), np.mean(y)
    x_c, y_c = x - cx, y - cy

    width = np.max(x) - np.min(x)
    height = np.max(y) - np.min(y)
    aspect_ratio = max(width, height) / (min(width, height) + 1e-6)

    distances = np.sqrt(x_c**2 + y_c**2)
    mean_dist = np.mean(distances)
    cv_dist = np.std(distances) / (mean_dist + 1e-6)
    dist_range_ratio = (np.max(distances) - np.min(distances)) / (mean_dist + 1e-6)

    angles = np.arctan2(y_c, x_c)
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]

    angle_changes = []
    for i in range(len(sorted_points)):
        p1, p2, p3 = sorted_points[i], sorted_points[(i+1) % n], sorted_points[(i+2) % n]
        v1, v2 = p2 - p1, p3 - p2
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        det = v1[0]*v2[1] - v1[1]*v2[0]
        angle_changes.append(abs(math.atan2(det, dot)))

    angle_changes = np.array(angle_changes)
    mean_ac = np.mean(angle_changes)
    std_ac = np.std(angle_changes)
    max_ac = np.max(angle_changes)
    num_corners = np.sum(angle_changes > math.pi / 4)
    corner_ratio = num_corners / n

    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        shape_area = 0.5 * abs(sum(x_c[i] * y_c[(i+1)%n] - x_c[(i+1)%n] * y_c[i] for i in range(n)))
        hull_fill_ratio = shape_area / (hull_area + 1e-6)
    except:
        hull_area = width * height
        hull_fill_ratio = 1.0

    perimeter = sum(np.sqrt((sorted_points[(i+1)%n][0] - sorted_points[i][0])**2 +
                            (sorted_points[(i+1)%n][1] - sorted_points[i][1])**2) for i in range(n))
    circularity = 4 * math.pi * hull_area / (perimeter**2 + 1e-6)
    compactness = hull_area / (perimeter**2 + 1e-6)

    # Section straightness
    section_size = n // 4
    straightness_scores = []
    for sec in range(4):
        start = sec * section_size
        end = start + section_size if sec < 3 else n
        section_pts = sorted_points[start:end]
        if len(section_pts) < 3:
            continue
        slopes = []
        for i in range(len(section_pts) - 1):
            dx = section_pts[i+1][0] - section_pts[i][0]
            dy = section_pts[i+1][1] - section_pts[i][1]
            if abs(dx) > 1e-6:
                slopes.append(dy / dx)
        if len(slopes) >= 2:
            cv_slope = np.std(slopes) / (abs(np.mean(slopes)) + 0.01)
            straightness_scores.append(1.0 / (1.0 + cv_slope))
    avg_straightness = np.mean(straightness_scores) if straightness_scores else 0.5

    # Radius variance across angular sectors
    section_radii = [[] for _ in range(8)]
    for i in range(n):
        sec = int((angles[i] + math.pi) / (2 * math.pi) * 8) % 8
        section_radii[sec].append(distances[i])
    section_means = [np.mean(r) if r else mean_dist for r in section_radii]
    radius_section_cv = np.std(section_means) / (mean_dist + 1e-6)

    return [aspect_ratio, cv_dist, dist_range_ratio, mean_ac, std_ac, max_ac,
            corner_ratio, circularity, compactness, hull_fill_ratio,
            avg_straightness, radius_section_cv, num_corners]


def train_random_forest(noise_levels, samples_per_noise=60, seed=123):
    """
    Train a Random Forest classifier on generated shape data.

    Args:
        noise_levels: List of noise levels to generate training data for.
        samples_per_noise: Training samples per noise level.
        seed: Random seed for training data.

    Returns:
        dict with 'model', 'scaler', and 'feature_names'.
    """
    print("Generating training data...")
    random.seed(seed)
    np.random.seed(seed)

    features, labels = [], []
    for noise_level in noise_levels:
        for _ in range(samples_per_noise):
            true_shape, points = generate_test_shape(noise_level)
            try:
                features.append(extract_features(points))
                labels.append(true_shape)
            except:
                continue

    print(f"Generated {len(features)} training samples")
    print("Class distribution:", dict(Counter(labels)))

    X = np.array(features)
    y = np.array(labels)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
    model.fit(X_scaled, y)

    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    print(f"Cross-validation accuracy: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)")

    return {'model': model, 'scaler': scaler, 'feature_names': [
        "aspect_ratio", "cv_distance", "dist_range_ratio", "mean_angle_change",
        "std_angle_change", "max_angle_change", "corner_ratio", "circularity",
        "compactness", "hull_fill_ratio", "avg_straightness", "radius_section_cv",
        "num_corners"
    ]}


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

if __name__ == "__main__":

    # Configuration
    NOISE_LEVELS = [round(x * 0.02, 2) for x in range(0, 36)]
    SAMPLES_PER_LEVEL = 100
    TEST_SEED = 42
    TRAIN_SEED = 123

    # --- Run deterministic method ---
    print("\n" + "=" * 70)
    print("RUNNING DETERMINISTIC METHOD")
    print("=" * 70)

    random.seed(TEST_SEED)
    np.random.seed(TEST_SEED)

    your_results = []
    total = len(NOISE_LEVELS) * SAMPLES_PER_LEVEL
    count = 0

    for noise_level in NOISE_LEVELS:
        for sample_idx in range(SAMPLES_PER_LEVEL):
            count += 1
            if count % 500 == 0:
                print(f"Processing {count}/{total}...")

            true_shape, points = generate_test_shape(noise_level)
            try:
                pv = find_polygon_values(points)
                predicted, score = classify_shape(pv)
            except:
                predicted, score = "unknown", float('inf')

            your_results.append({
                "noise_level": noise_level,
                "sample_idx": sample_idx,
                "true_shape": true_shape,
                "predicted_shape": predicted,
                "score": float(score) if score != float('inf') else 999999
            })

    # Save results
    with open('your_method_results.json', 'w') as f:
        json.dump(your_results, f)

    correct_family = sum(1 for r in your_results if is_family_match(r["true_shape"], r["predicted_shape"]))
    print(f"\nDeterministic method family accuracy: {correct_family}/{len(your_results)} "
          f"({100*correct_family/len(your_results):.1f}%)")

    # --- Train and run Random Forest ---
    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST BASELINE")
    print("=" * 70)

    model_data = train_random_forest(NOISE_LEVELS, samples_per_noise=60, seed=TRAIN_SEED)

    # Save model
    with open('ml_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    # Test Random Forest on same test data
    print("\nRunning Random Forest on test data...")
    with open('your_method_results.json', 'r') as f:
        your_results = json.load(f)

    # Re-generate test data with same seed
    random.seed(TEST_SEED)
    np.random.seed(TEST_SEED)

    ml_results = []
    for noise_level in NOISE_LEVELS:
        for sample_idx in range(SAMPLES_PER_LEVEL):
            true_shape, points = generate_test_shape(noise_level)
            try:
                feats = extract_features(points)
                feats_scaled = model_data['scaler'].transform([feats])
                predicted = model_data['model'].predict(feats_scaled)[0]
            except:
                predicted = "circle"
            ml_results.append({
                "noise_level": noise_level,
                "sample_idx": sample_idx,
                "true_shape": true_shape,
                "predicted_shape": predicted
            })

    with open('ml_results.json', 'w') as f:
        json.dump(ml_results, f)

    # --- Generate comparison chart ---
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON CHART")
    print("=" * 70)

    your_by_noise, ml_by_noise = {}, {}
    for r in your_results:
        n = r["noise_level"]
        your_by_noise.setdefault(n, {"family": 0, "total": 0})
        your_by_noise[n]["total"] += 1
        if is_family_match(r["true_shape"], r["predicted_shape"]):
            your_by_noise[n]["family"] += 1

    for r in ml_results:
        n = r["noise_level"]
        ml_by_noise.setdefault(n, {"family": 0, "total": 0})
        ml_by_noise[n]["total"] += 1
        if is_family_match(r["true_shape"], r["predicted_shape"]):
            ml_by_noise[n]["family"] += 1

    noise_sorted = sorted(your_by_noise.keys())
    your_fam = [your_by_noise[n]["family"] / your_by_noise[n]["total"] * 100 for n in noise_sorted]
    ml_fam = [ml_by_noise[n]["family"] / ml_by_noise[n]["total"] * 100 for n in noise_sorted]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(noise_sorted, your_fam, 'b-o', linewidth=2, markersize=4, label='Your Method')
    ax.plot(noise_sorted, ml_fam, 'r--s', linewidth=2, markersize=4, label='Random Forest')
    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Family-Matching Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    your_overall = sum(your_by_noise[n]["family"] for n in noise_sorted) / sum(your_by_noise[n]["total"] for n in noise_sorted) * 100
    ml_overall = sum(ml_by_noise[n]["family"] for n in noise_sorted) / sum(ml_by_noise[n]["total"] for n in noise_sorted) * 100
    ax.text(0.98, 0.98, f'Overall:\nYours: {your_overall:.1f}%\nRF: {ml_overall:.1f}%',
            transform=ax.transAxes, fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('Forest-Model_Comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 70)
    print(f"RESULTS SUMMARY")
    print(f"  Deterministic: {your_overall:.1f}% family accuracy")
    print(f"  Random Forest: {ml_overall:.1f}% family accuracy")
    print("=" * 70)
