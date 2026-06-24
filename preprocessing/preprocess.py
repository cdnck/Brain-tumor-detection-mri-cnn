import os
import cv2
import numpy as np


IMG_SIZE = 224


def load_image(path: str) -> np.ndarray:
    """Load and preprocess a single MRI image."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img


def load_dataset(dataset_path: str):
    """
    Load all images from a directory structured as:
        dataset_path/
            class1/img1.jpg ...
            class2/img1.jpg ...
    Returns (images, labels, class_names)
    """
    class_names = sorted(os.listdir(dataset_path))
    images, labels = [], []

    for label, cls in enumerate(class_names):
        cls_path = os.path.join(dataset_path, cls)
        for fname in os.listdir(cls_path):
            fpath = os.path.join(cls_path, fname)
            try:
                img = load_image(fpath)
                images.append(img)
                labels.append(label)
            except Exception:
                continue

    return np.array(images), np.array(labels), class_names
