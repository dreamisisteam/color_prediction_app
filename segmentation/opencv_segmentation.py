from pathlib import Path, PosixPath
from typing import Union, Tuple
import numpy as np
from matplotlib import pyplot as plt

import cv2

def extract_object_from_file_path(image_path: Union[PosixPath, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the largest object from an image based on contours and returns the original image, 
    binary mask, object mask, and the object over a white background.

    Parameters:
        image_path (Union[PosixPath, str]): Path to the input image file.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - Original image (BGR format).
            - Binary image highlighting the object (binary mask).
            - Object mask (grayscale).
            - Extracted object with a white background.

    Raises:
        FileNotFoundError: If the image file cannot be loaded from the specified path.
        ValueError: If no contours are found in the image.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converting to grayscale

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the image.")

    largest_contour = max(contours, key=cv2.contourArea) # Identify the largest contour

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    x, y, w, h = cv2.boundingRect(largest_contour)

    cropped_object = image[y:y+h, x:x+w]
    mask_cropped = mask[y:y+h, x:x+w]

    white_background = np.full_like(cropped_object, 255)
    result = np.where(mask_cropped[:, :, None] == 255, cropped_object, white_background)

    return image, binary, mask, result


def extract_object_from_file(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the largest object from an image based on contours and returns the original image, 
    binary mask, object mask, and the object over a white background.

    Parameters:
        image (np.ndarray): image in np.array

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - Original image (BGR format).
            - Binary image highlighting the object (binary mask).
            - Object mask (grayscale).
            - Extracted object with a white background.

    Raises:
        FileNotFoundError: If the image file cannot be loaded from the specified path.
        ValueError: If no contours are found in the image.
    """
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converting to grayscale

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the image.")

    largest_contour = max(contours, key=cv2.contourArea) # Identify the largest contour

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    x, y, w, h = cv2.boundingRect(largest_contour)

    cropped_object = image[y:y+h, x:x+w]
    mask_cropped = mask[y:y+h, x:x+w]

    white_background = np.full_like(cropped_object, 255)
    result = np.where(mask_cropped[:, :, None] == 255, cropped_object, white_background)

    return image, binary, mask, result


if __name__ == "__main__":

    root_dir = Path(__file__).parents[2]
    image_path = root_dir / "data" / "test" / "258-5.JPG"

    original, binary, mask, cropped = extract_object_from_file_path(image_path)

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 4, 2)
    plt.title("Binary Image")
    plt.imshow(binary, cmap='gray')

    plt.subplot(1, 4, 3)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')

    plt.subplot(1, 4, 4)
    plt.title("Cropped Object")
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()
