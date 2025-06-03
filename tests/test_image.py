import pytest
import numpy as np
import pandas as pd
import os
from PIL import Image

from semiq_ml.image import (
    path_to_dataframe, 
    path_to_dataframe_with_labels,
    load_image_as_array,
    load_images_from_dataframe,
    display_images
)

class TestImageUtils:
    def test_path_to_dataframe(self, sample_image_directory):
        """Test path_to_dataframe function."""
        # Test basic functionality
        df = path_to_dataframe(sample_image_directory)
        
        # Check that DataFrame contains expected columns
        assert isinstance(df, pd.DataFrame)
        assert 'path' in df.columns
        assert 'name' in df.columns
        
        # Should find all images in directory and subdirectories
        total_images = 3 * 2 + 3  # 3 images x 2 classes + 3 grayscale
        assert len(df) == total_images
        
        # Test with metadata
        df_with_metadata = path_to_dataframe(sample_image_directory, include_metadata=True)
        assert 'width' in df_with_metadata.columns
        assert 'height' in df_with_metadata.columns
        assert 'img_mode' in df_with_metadata.columns
        
        # All images should have width and height of 50x50
        assert (df_with_metadata['width'] == 50).all()
        assert (df_with_metadata['height'] == 50).all()
        
        # Test with non-existent directory
        empty_df = path_to_dataframe('/path/that/does/not/exist')
        assert len(empty_df) == 0
        
    def test_path_to_dataframe_with_labels(self, sample_image_directory):
        """Test path_to_dataframe_with_labels function."""
        # Test basic functionality
        df = path_to_dataframe_with_labels(sample_image_directory)
        
        # Check that DataFrame contains expected columns
        assert isinstance(df, pd.DataFrame)
        assert 'path' in df.columns
        assert 'name' in df.columns
        assert 'label' in df.columns
        
        # Should derive labels from subdirectory names
        assert 'class1' in df['label'].values
        assert 'class2' in df['label'].values
        
        # Test label counts
        label_counts = df['label'].value_counts()
        assert label_counts['class1'] == 6  # 3 regular + 3 grayscale
        assert label_counts['class2'] == 3
        
    def test_load_image_as_array(self, sample_image_directory):
        """Test load_image_as_array function."""
        # Get path to a test image
        df = path_to_dataframe(sample_image_directory)
        image_path = df['path'].iloc[0]
        
        # Test basic loading
        img_array = load_image_as_array(image_path)
        assert isinstance(img_array, np.ndarray)
        assert img_array.shape == (50, 50, 3)  # RGB image of size 50x50
        
        # Test resize
        resized_array = load_image_as_array(image_path, size=(30, 30))
        assert resized_array.shape == (30, 30, 3)
        
        # Test grayscale conversion
        gray_array = load_image_as_array(image_path, mode='L')
        assert gray_array.shape == (50, 50)  # Grayscale has no channel dimension
        
        # Test normalization
        normalized_array = load_image_as_array(image_path, normalize=True)
        assert normalized_array.dtype == np.float32
        assert 0 <= normalized_array.min() <= 1
        assert 0 <= normalized_array.max() <= 1
        
        # Test handling non-existent image
        non_existent = load_image_as_array('/path/to/nonexistent.jpg')
        assert non_existent is None
        
    def test_load_images_from_dataframe(self, sample_image_directory):
        """Test load_images_from_dataframe function."""
        # Get dataframe with images
        df = path_to_dataframe_with_labels(sample_image_directory)
        
        # Test basic loading (first 3 images)
        small_df = df.head(3)
        images = load_images_from_dataframe(small_df)
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == 3
        assert images.shape[1:] == (50, 50, 3)
        
        # Test with labels
        images, labels = load_images_from_dataframe(small_df, label_col='label')
        assert isinstance(images, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == 3
        
        # Test resizing
        small_images = load_images_from_dataframe(small_df, size=(25, 25))
        assert small_images.shape[1:] == (25, 25, 3)
        
        # Test normalization
        norm_images = load_images_from_dataframe(small_df, normalize=True)
        assert norm_images.dtype == np.float32
        assert 0 <= norm_images.min() <= 1
        assert 0 <= norm_images.max() <= 1
        
    def test_display_images(self, sample_image_directory):
        """Test display_images function (just checking if it runs)."""
        # Get some test images
        df = path_to_dataframe_with_labels(sample_image_directory)
        df_small = df.head(3)
        images = load_images_from_dataframe(df_small)
        labels = df_small['label'].values
        
        try:
            # This should run without error, but we can't test visual output
            display_images(images[:2])
            display_images(images[:2], labels=labels[:2])
            display_images(images[:2], labels=labels[:2], predictions=labels[:2])
        except Exception as e:
            # Matplotlib might not be available in test environment
            # So we'll just verify it's not a code logic error
            assert "matplotlib" in str(e).lower() or "display" in str(e).lower()
