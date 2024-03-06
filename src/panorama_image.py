import numpy as np
import math
from src import process_image
from src import matrix
from src import harris_image


def match_compare(a, b):
    # Comparator for matches based on distance
    if a['distance'] < b['distance']:
        return -1
    elif a['distance'] > b['distance']:
        return 1
    else:
        return 0


def make_point(x, y):
    # Helper function to create 2D points
    return {'x': x, 'y': y}


def both_images(a, b):
    # Place two images side by side on canvas for drawing matching pixels
    h = max(a['h'], b['h'])
    w = a['w'] + b['w']
    c = max(a['c'], b['c'])
    both = process_image.make_image(w, h, c)
    both['data'][:a['h'], :a['w'], :a['c']] = a['data']
    both['data'][:b['h'], a['w']:a['w'] + b['w'], :b['c']] = b['data']
    return both


def draw_matches(a, b, matches, n, inliers):
    both = both_images(a, b)

    for i, match in enumerate(matches):
        # Extract 'p' and 'q' directly from the match dictionary
        bx, by = match['p']  # Coordinates from image a
        ex, ey = match['q']  # Coordinates from image b
        ex += a['w']  # Adjust 'ex' for the combined width of images a and b

        # Decide color based on whether the match is an inlier or not
        color = (0, 255, 0) if i < inliers else (255, 0, 0)  # Green for inliers, red for others

        # Drawing the line from (bx, by) to (ex, ey)
        dx, dy = ex - bx, ey - by
        steps = max(abs(dx), abs(dy))

        for step in range(steps + 1):
            t = step / steps
            x = int(bx + dx * t)
            y = int(by + dy * t)
            if 0 <= x < both['w'] and 0 <= y < both['h']:
                process_image.set_pixel(both, x, y, 0, color[0])
                process_image.set_pixel(both, x, y, 1, color[1])
                process_image.set_pixel(both, x, y, 2, color[2])

    return both


def draw_inliers(a, b, H, matches, n, thresh):
    inliers = model_inliers(H, matches, n, thresh)
    return draw_matches(a, b, matches, n, inliers)


def find_and_draw_matches(a, b, sigma, thresh, nms):
    ad, an = harris_image.harris_corner_detector(a, sigma, thresh, nms)
    bd, bn = harris_image.harris_corner_detector(b, sigma, thresh, nms)
    matches = match_descriptors(ad, an, bd, bn)
    print(matches)
    lines = draw_matches(a, b, matches, len(matches), 0)
    return lines


# Calculates L1 distance between to floating point arrays
def l1_distance(a, b, n):
    # Implement the L1 distance calculation
    return sum(abs(a[i] - b[i]) for i in range(n))


def match_descriptors(a, an, b, bn):
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
    # Implement point projection using homography H
    return make_point(0, 0)


# Calculate L2 distance between two points
def point_distance(p, q):
    return math.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2)


# Count number of inliers in a set of matches
def model_inliers(H, m, n, thresh):
    count = 0
    # Implement inlier counting and sorting logic here
    return count


# Randomly shuffle matches for RANSAC
def randomize_matches(m, n):
    np.random.shuffle(m)


# Computes homography between two images given matching pixels
def compute_homography(matches, n):
    # Implement homography computation here
    return matrix.make_matrix(3, 3)  # Placeholder


# Perform RANdom SAmple Consensus to calculate homography for noisy matches
def RANSAC(m, n, thresh, k, cutoff):
    # Implement RANSAC algorithm here
    return matrix.make_matrix(3, 3)  # Placeholder


# Stitches two images together using a projective transformation
def combine_images(a, b, H):
    # Implement image combination logic here
    return process_image.make_image(0, 0, 0)  # Placeholder


# Create a panorama between two images
def panorama_image(a, b, sigma, thresh, nms, inlier_thresh, iters, cutoff):
    # Implement panorama creation logic here
    return process_image.make_image(0, 0, 0)  # Placeholder


# Project an image onto a cylinder
def cylindrical_project(im, f):
    # Implement cylindrical projection here
    return process_image.copy_image(im)
