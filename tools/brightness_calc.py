import numpy as np
from scipy.stats import beta
import math

RED_SENSITIVITY = 0.299
GREEN_SENSITIVITY = 0.587
BLUE_SENSITIVITY = 0.114


def convert_to_brightness_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        raise ValueError("uint8 is not a good dtype for the image")

    return np.sqrt(
        image[..., 0] ** 2 * RED_SENSITIVITY
        + image[..., 1] ** 2 * GREEN_SENSITIVITY
        + image[..., 2] ** 2 * BLUE_SENSITIVITY
    )

def get_resolution(image: np.ndarray):
    height, width = image.shape[:2]
    return height * width

def brightness_histogram(image: np.ndarray) -> np.ndarray:
    nr_of_pixels = get_resolution(image)
    brightness_image = convert_to_brightness_image(image)
    hist, _ = np.histogram(brightness_image, bins=256, range=(0, 255))
    return hist / nr_of_pixels
    
def distribution_pmf(dist, start: float, stop: float, nr_of_steps: int):
    xs = np.linspace(start, stop, nr_of_steps)
    ys = dist.pdf(xs)
    # divide by the sum to make a probability mass function
    return ys / np.sum(ys)

def correlation_distance(
    distribution_a: np.ndarray, distribution_b: np.ndarray
) -> float:
    dot_product = np.dot(distribution_a, distribution_b)
    squared_dist_a = np.sum(distribution_a ** 2)
    squared_dist_b = np.sum(distribution_b ** 2)
    return dot_product / math.sqrt(squared_dist_a * squared_dist_b)

def compute_hdr(img: np.ndarray):
    img_brightness_pmf = brightness_histogram(np.float32(img))
    ref_pmf = distribution_pmf(beta(2, 2), 0, 1, 256)
    return correlation_distance(ref_pmf, img_brightness_pmf)

def pixel_brightness(pixel):
    r, g, b = pixel
    return math.sqrt(0.299 * r ** 2 + 0.587 * g ** 2 + 0.114 * b ** 2)

def image_brightness(img):
    nr_of_pixels = img.shape[0] * img.shape[1]
    img_sq = img.reshape(img.shape[0]*img.shape[1],3)
    return sum(pixel_brightness(pixel) for pixel in img_sq) / nr_of_pixels


def get_stats(image:np.ndarray):
    brightness = image_brightness(image)
    hdr_brightness = compute_hdr(image)

    return hdr_brightness, brightness
