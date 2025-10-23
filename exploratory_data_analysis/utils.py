from collections import defaultdict
import os
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from typing import Dict, List

def compute_visibility(objects, img_root, default_size=(1280, 720)):
    """
    Computes the visibility of objects in images as a percentage of the image area.

    Args:
        objects (dict): A dictionary where keys are image filenames and values are lists of 
                        dictionaries representing object annotations. Each object annotation 
                        should contain the keys "x_min", "x_max", "y_min", "y_max", and "class".
        img_root (str): The root directory containing the images.
        default_size (tuple, optional): The default image size (width, height) to use if the 
                                         image cannot be loaded. Defaults to (1280, 720).

    Returns:
        pd.DataFrame: A DataFrame containing the following columns:
            - "class": The object class.
            - "mean_visibility(%)": The mean visibility percentage of the objects in the class.
            - "std_visibility(%)": The standard deviation of the visibility percentages.
            - "sample_count": The number of objects in the class.
    """
    vis = defaultdict(list)
    for img, labels in objects.items():
        img_path = os.path.join(img_root, img)
        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                w, h = default_size
        else:
            w, h = default_size
        img_area = w * h
        for o in labels:
            bw, bh = o["x_max"] - o["x_min"], o["y_max"] - o["y_min"]
            if bw > 0 and bh > 0:
                vis[o["class"]].append((bw * bh / img_area) * 100)
    rows = []
    for cls, vals in vis.items():
        arr = np.array(vals)
        rows.append({
            "class": cls,
            "mean_visibility(%)": np.mean(arr),
            "std_visibility(%)": np.std(arr),
            "sample_count": len(arr)
        })
    return pd.DataFrame(rows)


def cooccurrence_matrix(objects: Dict[str, List[dict]], classes: List[str]) -> pd.DataFrame:
    """
    Computes a co-occurrence matrix for a given set of objects and their associated classes.

    The co-occurrence matrix represents how often pairs of classes appear together 
    within the same object. The diagonal of the matrix indicates the frequency of 
    individual classes, while the off-diagonal elements represent the co-occurrence 
    counts between pairs of classes.

    Args:
        objects (Dict[str, List[dict]]): A dictionary where the keys are object identifiers 
            (e.g., strings) and the values are lists of dictionaries. Each dictionary in the 
            list must contain a "class" key, representing the class label of the object.
        classes (List[str]): A list of all possible class labels. These will be used as 
            the row and column indices of the resulting co-occurrence matrix.

    Returns:
        pd.DataFrame: A square DataFrame where both the rows and columns are indexed by 
        the class labels from the `classes` list. Each cell (i, j) contains the count 
        of how many times class `i` and class `j` co-occurred in the input data.

    Example:
        objects = {
            "obj1": [{"class": "A"}, {"class": "B"}],
            "obj2": [{"class": "A"}, {"class": "C"}],
            "obj3": [{"class": "B"}, {"class": "C"}]
        }
        classes = ["A", "B", "C"]
        result = cooccurrence_matrix(objects, classes)
        # The resulting DataFrame will look like:
        #      A  B  C
        # A    2  1  1
        # B    1  2  1
        # C    1  1  2
    """
    mat = pd.DataFrame(0, index=classes, columns=classes)
    for labels in objects.values():
        present = list({o["class"] for o in labels})
        for i in range(len(present)):
            for j in range(i, len(present)):
                mat.loc[present[i], present[j]] += 1
                if i != j:
                    mat.loc[present[j], present[i]] += 1
    return mat


def kmeans_anchors(objects: Dict[str, List[dict]], k=9):
    """
    Perform k-means clustering to determine anchor box dimensions for object detection.

    Args:
        objects (Dict[str, List[dict]]): A dictionary where keys are object categories 
            and values are lists of dictionaries. Each dictionary represents an object 
            with bounding box coordinates, containing the keys "x_min", "x_max", "y_min", 
            and "y_max".
        k (int, optional): The number of clusters (anchor boxes) to generate. Defaults to 9.

    Returns:
        np.ndarray: A 2D array of shape (k, 2) containing the cluster centers, where each 
            row represents the width and height of an anchor box. If no valid boxes are 
            provided, returns an empty array of shape (0, 2).
    """
    boxes = []
    for labels in objects.values():
        for o in labels:
            w, h = o["x_max"] - o["x_min"], o["y_max"] - o["y_min"]
            if w > 0 and h > 0:
                boxes.append([w, h])
    if not boxes:
        return np.zeros((0, 2))
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(boxes)
    return km.cluster_centers_


def draw_boxes(image_path, objects, show_classes=None):
    """
    Draws bounding boxes on an image based on the provided object annotations.

    Args:
        image_path (str): Path to the image file.
        objects (list of dict): A list of dictionaries, where each dictionary represents
            an object with the following keys:
                - "class" (str): The class name of the object.
                - "x_min" (int): The x-coordinate of the top-left corner of the bounding box.
                - "y_min" (int): The y-coordinate of the top-left corner of the bounding box.
                - "x_max" (int): The x-coordinate of the bottom-right corner of the bounding box.
                - "y_max" (int): The y-coordinate of the bottom-right corner of the bounding box.
        show_classes (set, optional): A set of class names to filter which objects to draw.
            If None, all objects are drawn. Defaults to None.

    Returns:
        PIL.Image.Image: The image with bounding boxes drawn on it.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for o in objects:
        cls = o["class"]
        if show_classes and cls not in show_classes:
            continue
        x1, y1, x2, y2 = o["x_min"], o["y_min"], o["x_max"], o["y_max"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(0, y1 - 10)), cls, fill="yellow", font=font)
    return img
