import os
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
from typing import List, Tuple, Optional, Dict, Set, Any, Union
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# --- DataFrame Creation ---

def path_to_dataframe(
    folder_path: str,
    extensions: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'},
    include_metadata: bool = False
) -> pd.DataFrame:
    """
    Scans the folder for image files and returns a DataFrame.

    Args:
        folder_path (str): The path to the main folder to scan.
        extensions (Set[str]): A set of allowed image file extensions (case-insensitive).
        include_metadata (bool): If True, includes image width, height, and mode in the DataFrame.
                                 This will slow down scanning as images need to be opened.

    Returns:
        pd.DataFrame: DataFrame with columns: 'path', 'name'.
                      If include_metadata is True, also includes 'width', 'height', 'img_mode'.
    """
    if not os.path.isdir(folder_path):
        logger.error(f"Folder not found: {folder_path}")
        return pd.DataFrame(columns=['path', 'name'] + (['width', 'height', 'img_mode'] if include_metadata else []))

    data = []
    logger.info(f"Scanning folder: {folder_path} for images...")
    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                full_path = os.path.join(root, file)
                entry = {'path': full_path, 'name': file}
                if include_metadata:
                    try:
                        with Image.open(full_path) as img:
                            entry['width'] = img.width
                            entry['height'] = img.height
                            entry['img_mode'] = img.mode
                    except UnidentifiedImageError:
                        logger.warning(f"Could not identify image file (metadata): {full_path}. Skipping metadata.")
                        entry['width'], entry['height'], entry['img_mode'] = None, None, "Unknown"
                    except Exception as e:
                        logger.warning(f"Error reading metadata for {full_path}: {e}. Skipping metadata.")
                        entry['width'], entry['height'], entry['img_mode'] = None, None, "Error"
                data.append(entry)
    
    df = pd.DataFrame(data)
    if df.empty:
        logger.warning(f"No images found in {folder_path} with the specified extensions.")
    else:
        logger.info(f"Found {len(df)} image files.")
    return df

def path_to_dataframe_with_labels(
    folder_path: str,
    extensions: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'},
    include_metadata: bool = False
) -> pd.DataFrame:
    """
    Scans the folder for image files and returns a DataFrame with inferred labels.
    The label is inferred from the immediate parent directory name.
    (e.g., for datasets structured as root/class_name/image.jpg).

    Args:
        folder_path (str): The path to the main folder containing class subdirectories.
        extensions (Set[str]): A set of allowed image file extensions (case-insensitive).
        include_metadata (bool): If True, includes image width, height, and mode.

    Returns:
        pd.DataFrame: DataFrame with columns: 'path', 'name', 'label'.
                      If include_metadata is True, also includes 'width', 'height', 'img_mode'.
    """
    if not os.path.isdir(folder_path):
        logger.error(f"Folder not found: {folder_path}")
        return pd.DataFrame(columns=['path', 'name', 'label'] + (['width', 'height', 'img_mode'] if include_metadata else []))

    data = []
    logger.info(f"Scanning folder: {folder_path} for images with labels...")
    for root, _, files in os.walk(folder_path):
        if os.path.samefile(root, folder_path): 
            current_label = None
        else:
            current_label = os.path.basename(root)

        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                if current_label is None:
                    logger.debug(f"Skipping image in root scan directory: {file}")
                    continue

                full_path = os.path.join(root, file)
                entry = {'path': full_path, 'name': file, 'label': current_label}
                if include_metadata:
                    try:
                        with Image.open(full_path) as img:
                            entry['width'] = img.width
                            entry['height'] = img.height
                            entry['img_mode'] = img.mode
                    except UnidentifiedImageError:
                        logger.warning(f"Could not identify image file (metadata): {full_path}. Skipping metadata.")
                        entry['width'], entry['height'], entry['img_mode'] = None, None, "Unknown"
                    except Exception as e:
                        logger.warning(f"Error reading metadata for {full_path}: {e}. Skipping metadata.")
                        entry['width'], entry['height'], entry['img_mode'] = None, None, "Error"
                data.append(entry)
    
    df = pd.DataFrame(data)
    if df.empty:
        logger.warning(f"No images found in subdirectories of {folder_path} with the specified extensions.")
    else:
        logger.info(f"Found {len(df)} image files with labels. Unique labels: {df['label'].unique().tolist()}")
    return df

# --- Image Loading ---

def load_image_as_array(
    image_path: str,
    size: Optional[Tuple[int, int]] = None,
    mode: str = 'RGB',
    resample_algo: Optional[Image.Resampling] = Image.Resampling.LANCZOS,
    normalize: bool = False,
    dtype: Union[np.dtype, str] = np.uint8
) -> Optional[np.ndarray]:
    """
    Loads an image from disk, optionally resizes and normalizes, and returns it as a NumPy array.

    Args:
        image_path (str): Path to the image file.
        size (Optional[Tuple[int, int]]): Target (width, height) to resize. If None, original size is kept.
        mode (str): Target color mode (e.g., 'RGB', 'L' for grayscale, 'RGBA').
        resample_algo (Optional[Image.Resampling]): Resampling algorithm for resizing.
                                   Defaults to Image.Resampling.LANCZOS for quality.
                                   Set to None to use Pillow's default (often BILINEAR).
        normalize (bool): If True, converts image to float32 and normalizes pixel values to [0, 1].
                          If False, uses the specified dtype (defaults to uint8).
        dtype (Union[np.dtype, str]): NumPy data type for the output array if not normalizing.

    Returns:
        Optional[np.ndarray]: Image as a NumPy array, or None if loading fails.
    """
    try:
        img = Image.open(image_path)
        if img.mode != mode:
            img = img.convert(mode)

        if size:
            actual_resample_algo = resample_algo if resample_algo is not None else Image.Resampling.BILINEAR
            img = img.resize(size, resample=actual_resample_algo)

        img_array = np.array(img)

        if normalize:
            if img_array.dtype != np.float32:
                img_array = img_array.astype(np.float32)
            img_array /= 255.0
        elif img_array.dtype != dtype:
            img_array = img_array.astype(dtype)
            
        return img_array

    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        return None
    except UnidentifiedImageError:
        logger.error(f"Could not identify or open image (possibly corrupted): {image_path}")
        return None
    except Exception as e:
        logger.error(f"An error occurred while loading image {image_path}: {e}")
        return None

def load_images_from_dataframe(
    df: pd.DataFrame,
    image_col: str = 'path',
    label_col: Optional[str] = None,
    size: Optional[Tuple[int, int]] = None,
    mode: str = 'RGB',
    resample_algo: Optional[Image.Resampling] = Image.Resampling.LANCZOS,
    normalize: bool = False,
    dtype: Union[np.dtype, str] = np.uint8,
    skip_errors: bool = True,
    show_progress: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Loads all images listed in the DataFrame. Optionally returns labels if present.

    Args:
        df (pd.DataFrame): DataFrame containing image paths and optionally labels.
        image_col (str): Name of the column containing image paths.
        label_col (Optional[str]): Name of the column containing labels. If None or not present, only images are returned.
        size, mode, resample_algo, normalize, dtype: Passed to load_image_as_array.
        skip_errors (bool): If True, skips images that fail to load and logs a warning.
                            If False, an error during loading will stop the process (not yet fully implemented, currently always skips).
        show_progress (bool): If True, displays a progress bar using tqdm.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            - If no valid label_col: A NumPy array of loaded images.
            - If valid label_col: A tuple (NumPy array of images, NumPy array of labels).
            Returns empty arrays if no images are successfully loaded.
    """
    images = []
    labels = []
    has_labels = label_col is not None and label_col in df.columns
    
    iterable = df.iterrows()
    if show_progress:
        iterable = tqdm(df.iterrows(), total=len(df), desc="Loading images")

    for _, row in iterable:
        path = row[image_col]
        img_array = load_image_as_array(
            path, size=size, mode=mode, resample_algo=resample_algo,
            normalize=normalize, dtype=dtype
        )

        if img_array is not None:
            images.append(img_array)
            if has_labels:
                labels.append(row[label_col])
        elif not skip_errors:
            logger.error(f"Failed to load {path} and skip_errors is False. Halting (not truly implemented, just logged).")

    if not images:
        logger.warning("No images were successfully loaded from the DataFrame.")
        empty_images_array = np.array([])
        return (empty_images_array, np.array([])) if has_labels else empty_images_array

    stacked_images = np.stack(images)
    
    if has_labels:
        return stacked_images, np.array(labels)
    return stacked_images

# --- Image Utilities ---

def display_images(
    images: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    labels: Optional[Union[List[Any], np.ndarray]] = None,
    predictions: Optional[Union[List[Any], np.ndarray]] = None,
    n_cols: int = 5,
    figsize_per_image: Tuple[int, int] = (3, 3),
    class_names: Optional[Dict[Any, str]] = None
) -> None:
    """
    Displays a list/array of images, optionally with true labels and predicted labels.

    Args:
        images (Union[np.ndarray, List[np.ndarray]]):
            A single image (H, W, C) or (H, W),
            or a list of such images,
            or a batch of images (N, H, W, C) or (N, H, W).
        labels (Optional[Union[List[Any], np.ndarray]]): True labels for the images.
        predictions (Optional[Union[List[Any], np.ndarray]]): Predicted labels for the images.
        n_cols (int): Number of columns in the display grid.
        figsize_per_image (Tuple[int, int]): Approximate (width, height) for each subplot in inches.
        class_names (Optional[Dict[Any, str]]): Dictionary mapping label values to human-readable names.
    """
    if isinstance(images, tuple) and len(images) == 2:
        if labels is None:
            labels = images[1]
        images = images[0]
    
    if not isinstance(images, list):
        if images.ndim == 2 or (images.ndim == 3 and images.shape[-1] in [1, 3, 4]):
            images = [images]
   
    num_images = len(images)
    if num_images == 0:
        print("No images to display.")
        return

    n_rows = (num_images - 1) // n_cols + 1
    
    fig_width = n_cols * figsize_per_image[0]
    fig_height = n_rows * figsize_per_image[1]
    
    plt.figure(figsize=(fig_width, fig_height))

    for i in range(num_images):
        plt.subplot(n_rows, n_cols, i + 1)
        current_image = images[i]
        
        if current_image.ndim == 3 and current_image.shape[-1] == 1:
            plt.imshow(current_image.squeeze(), cmap='gray')
        else:
            plt.imshow(current_image)

        title_parts = []
        if labels is not None and i < len(labels):
            true_label = labels[i]
            if class_names and true_label in class_names:
                true_label = class_names[true_label]
            title_parts.append(f"True: {true_label}")

        if predictions is not None and i < len(predictions):
            pred_label = predictions[i]
            if class_names and pred_label in class_names:
                pred_label = class_names[pred_label]
            
            color = "green"
            if labels is not None and i < len(labels) and labels[i] != predictions[i]:
                color = "red"
            title_parts.append(f"Pred: {pred_label}")
        
        plt.title("\n".join(title_parts), color=color if 'color' in locals() else 'black')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def sample_images(
    images: Union[np.ndarray, List[np.ndarray]],
    n_samples: int = 5,
    random_seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    Randomly samples a specified number of images from a list or array of images.

    Args:
        images (Union[np.ndarray, List[np.ndarray]]): Array or list of images.
        n_samples (int): Number of images to sample.
        random_seed (Optional[int]): Seed for reproducibility.

    Returns:
        List[np.ndarray]: List of sampled images.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if isinstance(images, np.ndarray):
        if images.ndim == 4:
            indices = np.random.choice(images.shape[0], size=min(n_samples, images.shape[0]), replace=False)
            return [images[i] for i in indices]
        elif images.ndim == 3:
            return [images] * min(n_samples, 1)
    elif isinstance(images, list):
        return np.random.choice(images, size=min(n_samples, len(images)), replace=False).tolist()
    
    return []

