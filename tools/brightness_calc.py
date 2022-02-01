import numpy as np
from scipy.stats import beta
import math
import cv2

def distribution_pmf(dist, start: float, stop: float, nr_of_steps: int):
    xs = np.linspace(start, stop, nr_of_steps)
    ys = dist.pdf(xs)
    return ys / np.sum(ys)

RED_PMF = distribution_pmf(beta(2, 2), 0, 1, 256)    

def get_resolution(image: np.ndarray):
    height, width = image.shape[:2]
    return height * width

def brightness_histogram(image: np.ndarray) -> np.ndarray:
    nr_of_pixels = get_resolution(image)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_brightness = image_hsv[:, :, 2]
    hist, _ = np.histogram(hsv_brightness, bins=256, range=(0, 255))
    del image_hsv
    return hist / nr_of_pixels
    
def correlation_distance(
    distribution_a: np.ndarray, distribution_b: np.ndarray
) -> float:
    dot_product = np.dot(distribution_a, distribution_b)
    squared_dist_a = np.sum(distribution_a ** 2)
    squared_dist_b = np.sum(distribution_b ** 2)
    return dot_product / math.sqrt(squared_dist_a * squared_dist_b)

def compute_hdr(img: np.ndarray):
    img_brightness_pmf = brightness_histogram(img)
    return correlation_distance(RED_PMF, img_brightness_pmf)

def compute_brightness_base(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_brightness = np.mean(image_hsv[:, :, 2])
    
    return hsv_brightness


