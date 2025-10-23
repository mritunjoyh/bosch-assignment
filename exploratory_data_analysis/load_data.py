from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import json
import os
import pandas as pd
import streamlit as st

@st.cache_data
def load_annotations(json_path: str) -> Tuple[Dict[str, List[dict]], List[str]]:
    """
    Load and parse annotations from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing annotation data.

    Returns:
        Tuple[Dict[str, List[dict]], List[str]]:
            - A dictionary where the keys are image names (str) and the values are lists of dictionaries,
              each representing an object with the following keys:
                - "class" (str): The category of the object.
                - "x_min" (float): The minimum x-coordinate of the bounding box.
                - "y_min" (float): The minimum y-coordinate of the bounding box.
                - "x_max" (float): The maximum x-coordinate of the bounding box.
                - "y_max" (float): The maximum y-coordinate of the bounding box.
            - A sorted list of unique class names (str) found in the annotations.

    Notes:
        - The function skips annotations without a valid "box2d" field or with invalid bounding box coordinates.
        - Bounding boxes with non-positive width or height are ignored.
        - Images without a "name" field are skipped.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    objects = defaultdict(list)
    classes = set()
    for item in data:
        image_name = item.get("name")
        if not image_name:
            continue
        for label in item.get("labels", []):
            if "box2d" not in label:
                continue
            box = label["box2d"]
            cls = label["category"]
            try:
                x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
            except Exception:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            classes.add(cls)
            objects[image_name].append(
                {"class": cls, "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2}
            )
    return objects, sorted(list(classes))


def objects_to_df(objects: Dict[str, List[dict]]) -> pd.DataFrame:
    """
    Converts a dictionary of labeled objects into a pandas DataFrame.

    Args:
        objects (Dict[str, List[dict]]): A dictionary where the keys are image file names (str)
            and the values are lists of dictionaries. Each dictionary in the list represents
            an object with the following keys:
                - "class" (str): The class label of the object.
                - "x_min" (float): The x-coordinate of the top-left corner of the bounding box.
                - "y_min" (float): The y-coordinate of the top-left corner of the bounding box.
                - "x_max" (float): The x-coordinate of the bottom-right corner of the bounding box.
                - "y_max" (float): The y-coordinate of the bottom-right corner of the bounding box.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to an object. The DataFrame contains
        the following columns:
            - "image" (str): The image file name.
            - "class" (str): The class label of the object.
            - "x_min" (float): The x-coordinate of the top-left corner of the bounding box.
            - "y_min" (float): The y-coordinate of the top-left corner of the bounding box.
            - "x_max" (float): The x-coordinate of the bottom-right corner of the bounding box.
            - "y_max" (float): The y-coordinate of the bottom-right corner of the bounding box.
    """
    rows = []
    for img, labels in objects.items():
        for o in labels:
            rows.append({
                "image": img, "class": o["class"],
                "x_min": o["x_min"], "y_min": o["y_min"],
                "x_max": o["x_max"], "y_max": o["y_max"]
            })
    return pd.DataFrame(rows)


def compute_class_stats(objects):
    """
    Computes statistics for object classes in a dataset.

    This function calculates the number of bounding boxes and the number of 
    images associated with each class in the dataset.

    Args:
        objects (dict): A dictionary where the keys are image identifiers 
                        (e.g., file names or IDs) and the values are lists 
                        of dictionaries. Each dictionary in the list represents 
                        an object and must contain a "class" key indicating the 
                        class of the object.

    Returns:
        tuple: A tuple containing two Counter objects:
            - box_counts (Counter): A Counter where the keys are class names 
              and the values are the total number of bounding boxes for each class.
            - img_counts (Counter): A Counter where the keys are class names 
              and the values are the number of images containing at least one 
              object of each class.
    """
    box_counts, img_counts = Counter(), Counter()
    for img, labels in objects.items():
        seen = set()
        for o in labels:
            box_counts[o["class"]] += 1
            seen.add(o["class"])
        for c in seen:
            img_counts[c] += 1
    return box_counts, img_counts