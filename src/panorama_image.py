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
    # Draws lines between matching pixels in two images
    both = both_images(a, b)
    for i in range(n):
        bx, by = matches[i]['p']['x'], matches[i]['p']['y']
        ex, ey = matches[i]['q']['x'] + a['w'], matches[i]['q']['y']
        color = (0, 1, 0) if i < inliers else (1, 0, 0)  # Green for inliers, Red for outliers
        for j in range(int(bx), int(ex + 1)):
            r = int((float(j - bx) / (ex - bx)) * (ey - by) + by)
            if 0 <= r < both['h']:
                process_image.set_pixel(both, j, r, 0, color[0])
                process_image.set_pixel(both, j, r, 1, color[1])
                process_image.set_pixel(both, j, r, 2, color[2])
    return both


def draw_inliers(a, b, H, matches, n, thresh):
    inliers = model_inliers(H, matches, n, thresh)
    return draw_matches(a, b, matches, n, inliers)


def find_and_draw_matches(a, b, sigma, thresh, nms):
    ad = harris_image.harris_corner_detector(a, sigma, thresh, nms)
    bd = harris_image.harris_corner_detector(b, sigma, thresh, nms)
    matches = match_descriptors(ad, bd)
    lines = draw_matches(a, b, matches, len(matches), 0)
    return lines


# Calculates L1 distance between to floating point arrays
def l1_distance(a, b, n):
    # Implement the L1 distance calculation
    return sum(abs(a[i] - b[i]) for i in range(n))


def match_descriptors(a, an, b, bn):
    """
    Finds best matches between descriptors of two images.

    Parameters:
    a, b: Lists of descriptors for pixels in two images.
    an, bn: Number of descriptors in arrays a and b.

    Returns:
    matches: List of best matches found. Each descriptor in a should match with at most
             one other descriptor in b.
    """
    matches = []
    seen = np.zeros(bn, dtype=int)  # Tracks which descriptors in b have been matched

    for j in range(an):
        bind = 0  # Placeholder for finding the best match
        # Placeholder logic to find best match:
        # You will compare descriptor a[j] with each descriptor in b to find the best match
        # and calculate the L1 distance for the actual matching logic.
        match = {
            'ai': j,
            'bi': bind,
            'p': a[j]['p'],  # Assuming each descriptor has a 'p' key for its point
            'q': b[bind]['p'],  # Same assumption for b
            'distance': 0  # Placeholder for the distance calculation
        }
        matches.append(match)

    # Placeholder for sorting matches based on distance and ensuring injectivity (one-to-one matches)
    # You might use the sorted() function with a custom key, or implement any sorting algorithm
    # Then iterate through sorted matches to eliminate duplicates as described in the C function

    # Update the matches list based on injectivity and sorting criteria
    # Assume we've done that and filtered out the matches, update mn accordingly
    mn = len(matches)  # Update this based on actual good matches after filtering

    return matches, mn


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
