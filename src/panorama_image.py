import random
import numpy as np
import math

from src import process_image, matrix, harris_image, image_util, resize_image
from typing import Callable


def match_compare(a: dict, b: dict) -> int:
    # Comparator for matches based on distance
    if a['distance'] < b['distance']:
        return -1
    elif a['distance'] > b['distance']:
        return 1
    else:
        return 0


def make_point(x, y) -> dict:
    return {'x': x, 'y': y}


def both_images(a: dict, b: dict) -> dict:
    # Place two images side by side on canvas for drawing matching pixels
    h: int = max(a['h'], b['h'])
    w: int = a['w'] + b['w']
    c: int = max(a['c'], b['c'])
    both: dict = process_image.make_image(w, h, c)
    both['data'][:a['h'], :a['w'], :a['c']] = a['data']
    both['data'][:b['h'], a['w']:a['w'] + b['w'], :b['c']] = b['data']
    return both


def draw_matches(a: dict, b: dict, matches: list, n, inliers) -> dict:
    both: dict = both_images(a, b)
    counter: int = 0  # To keep track of the current match index

    for match in matches:
        bx, by = match['p']  # Coordinates from image a
        ex, ey = match['q']  # Coordinates from image b, adjusted for offset
        ex += a['w']

        # Decide color based on the current match index relative to inliers
        # color = (0, 255, 0) if counter < inliers else (255, 0, 0)

        dx, dy = ex - bx, ey - by
        steps: float | int = max(abs(dx), abs(dy))

        for step in range(steps + 1):
            t: float | int = step / steps
            x: int = int(bx + dx * t)
            y: int = int(by + dy * t)
            if 0 <= x < both['w'] and 0 <= y < both['h']:
                process_image.set_pixel(both, x, y, 0, 255)
                process_image.set_pixel(both, x, y, 1, 0)
                process_image.set_pixel(both, x, y, 2, 0)  # Assume third color channel, set as needed

        counter += 1  # Increment counter for each match processed

    return both


def draw_inliers(a: dict, b: dict, H: dict, matches: list, n: int, thresh: int) -> dict:
    count, inliers = model_inliers(H, matches, n, thresh)
    return draw_matches(a, b, matches, n, inliers)


def find_and_draw_matches(a: dict, b: dict, sigma: float | int, thresh: int, nms: int) -> dict:
    ad, an = harris_image.harris_corner_detector(a, sigma, thresh, nms)
    bd, bn = harris_image.harris_corner_detector(b, sigma, thresh, nms)
    matches, mn = match_descriptors(ad, an, bd, bn)
    lines: dict = draw_matches(a, b, matches, mn, 0)
    return lines


# Calculates L1 distance between to floating point arrays
def l1_distance(a: dict, b: dict, n: int) -> float | int:
    # Implement the L1 distance calculation
    return sum(abs(a[i] - b[i]) for i in range(n))


def match_descriptors(a: dict, an: int, b, bn):
    matches = []
    seen = np.zeros(bn, dtype=int)  # Tracks which descriptors in b have been matched

    for j in range(an):
        descriptor_a = a[j]['data']
        bind = 0
        q = float('inf')
        for k in range(bn):
            descriptor_b = b[k]['data']
            distance = l1_distance(descriptor_a, descriptor_b, len(descriptor_a))
            if distance < q:
                q = distance
                bind = k
        match = {
            'ai': j,
            'bi': bind,
            'p': (a[j]['x'], a[j]['y']),
            'q': (b[bind]['x'], b[bind]['y']),
            'distance': q
        }
        matches.append(match)

    sorted_matches = sorted(matches, key=lambda x: list(x.values())[-1])
    best_match = []

    for match in sorted_matches:
        if not seen[match['bi']]:
            seen[match['bi']] = 1
            best_match.append(match)
        else:
            continue

    mn = len(best_match)  # Update this based on actual good matches after filtering

    return best_match, mn


# Apply a projective transformation to a point
def project_point(H, p):
    if isinstance(p, dict):
        x, y = p['x'], p['y']
    elif isinstance(p, tuple):
        x, y = p

    c = np.array([[x], [y], [1]])
    multiplied = matrix.matrix_mult_matrix(H, c)
    x_prime = multiplied[0, 0] / multiplied[2, 0]
    y_prime = multiplied[1, 0] / multiplied[2, 0]
    return make_point(x_prime, y_prime)


# Calculate L2 distance between two points
def point_distance(p, q):
    return math.sqrt((p['x'] - q[0]) ** 2 + (p['y'] - q[1]) ** 2)


# Count number of inliers in a set of matches
def model_inliers(H, m, n, thresh):
    count = 0
    match_list = []
    for match in m:
        point_a = match['p']
        point_b = match['q']
        p_hat = project_point(H, point_a)
        distance = point_distance(p_hat, point_b)
        if distance <= thresh:
            match_list.append(match)
            count += 1
    # Bring outlier in the match_list
    for rest in m:
        if rest not in match_list:
            match_list.append(rest)
        else:
            continue
    return count, match_list


# Randomly shuffle matches for RASAC
def randomize_matches(m, n):
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        m[i], m[j] = m[j], m[i]


# Computes homography between two images given matching pixels
def compute_homography(matches):
    num_matches = len(matches)
    M = np.zeros((num_matches * 2, 8))
    b = np.zeros((num_matches * 2, 1))

    for i, match in enumerate(matches):
        mx, my = match['p']
        nx, ny = match['q']
        M[2 * i] = [mx, my, 1, 0, 0, 0, -mx * nx, -my * nx]
        M[2 * i + 1] = [0, 0, 0, mx, my, 1, -mx * ny, -my * ny]
        b[2 * i, 0] = nx
        b[2 * i + 1, 0] = ny

    a, res, rank, s = np.linalg.lstsq(M, b, rcond=None)

    if a is None or a.shape[0] != 8:
        return None

    # Construct the homography matrix H from the solution vector 'a'
    H = np.zeros((3, 3))
    H[0, 0:3] = a[0:3].T
    H[1, 0:3] = a[3:6].T
    H[2, 0:2] = a[6:8].T
    H[2, 2] = 1

    return H


# Perform RANdom SAmple Consensus to calculate homography for noisy matches
def RANSAC(m, n, thresh, k, cutoff):
    best_model = None
    best_fit = float('-inf')
    for epoch in range(k):
        randomize_matches(m, n)
        random_subset = random.sample(m, 4)
        H = compute_homography(random_subset)
        inliers, match = model_inliers(H, m, n, thresh)
        if inliers > best_fit:
            H_update = compute_homography(match)
            update_inliers, _ = model_inliers(H_update, m, n, thresh)
            if update_inliers > best_fit:
                best_fit = update_inliers
                best_model = H_update
                if update_inliers > cutoff:
                    return H_update
    return best_model


# Stitches two images together using a projective transformation
def combine_images(a_img, b_img, H):
    # Calculate inverse homography matrix
    H_inv = np.linalg.inv(H)

    # Project the corners of image b into image a coordinates
    c1 = project_point(H_inv, make_point(0, 0))
    c2 = project_point(H_inv, make_point(b_img['w'] - 1, 0))
    c3 = project_point(H_inv, make_point(0, b_img['h'] - 1))
    c4 = project_point(H_inv, make_point(b_img['w'] - 1, b_img['h'] - 1))

    # Find top left and bottom right corners of image b warped into image a
    top_left = {'x': min(c1['x'], c2['x'], c3['x'], c4['x']), 'y': min(c1['y'], c2['y'], c3['y'], c4['y'])}
    btm_right = {'x': max(c1['x'], c2['x'], c3['x'], c4['x']), 'y': max(c1['y'], c2['y'], c3['y'], c4['y'])}

    # Determine new image dimensions and offsets
    dx = min(0, top_left['x'])
    dy = min(0, top_left['y'])
    w = max(a_img['w'], btm_right['x']) - dx
    h = max(a_img['h'], btm_right['y']) - dy

    # Create new image with determined dimensions
    c = process_image.make_image(int(w), int(h), a_img['c'])

    # Paste image a into the new image offset by dx and dy
    for channel in range(a_img['c']):
        for y in range(a_img['h']):
            for x in range(a_img['w']):
                value = process_image.get_pixel(a_img, x, y, channel)
                process_image.set_pixel(c, x - dx, y - dy, channel, value)

    # Paste image b into the new image based on the projection
    for y in range(int(top_left['y']), int(btm_right['y'])):
        for x in range(int(top_left['x']), int(btm_right['x'])):
            point = project_point(H, {'x': x, 'y': y})
            if 0 <= point['x'] < b_img['w'] and 0 <= point['y'] < b_img['h']:
                # Use bilinear interpolation to get the pixel value from image b
                r, g, b = resize_image.bilinear_interpolate(b_img, point['x'], point['y'], c['c'])
                # Set the pixel value in the new image
                process_image.set_pixel(c, x - dx, y - dy, 0, r)
                process_image.set_pixel(c, x - dx, y - dy, 1, g)
                process_image.set_pixel(c, x - dx, y - dy, 2, b)
    return c


# Create a panorama between two images
def panorama_image(a, b, thresh, sigma=2, nms=3, inlier_thresh=3, iters=25000, cutoff=60):
    ad, an = harris_image.harris_corner_detector(a, sigma, thresh, nms)
    bd, bn = harris_image.harris_corner_detector(b, sigma, thresh, nms)

    matches, mn = match_descriptors(ad, an, bd, bn)
    H = RANSAC(matches, mn, inlier_thresh, iters, cutoff)

    if False:  # Turn it off if you don't wanna make inliners image.
        harris_image.mark_corners(a, ad)
        harris_image.mark_corners(b, bd)
        inlier_match = draw_inliers(a, b, H, matches, mn, inlier_thresh)
        image_util.save_image(inlier_match, "inliners")

    combined_image = combine_images(a, b, H)

    return combined_image


# Project an image onto a cylinder
def cylindrical_project(im, f):
    # Implement cylindrical projection here
    return process_image.copy_image(im)
