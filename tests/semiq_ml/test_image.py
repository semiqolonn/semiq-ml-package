"""Tests for the image module."""
import os
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
import pytest
from unittest.mock import patch, MagicMock, mock_open

from semiq_ml.image import (
    path_to_dataframe,
    path_to_dataframe_with_labels,
    load_image_as_array,
    load_images_from_dataframe,
    display_images,
    sample_images
)

# --- Fixtures ---

@pytest.fixture
def sample_image_path(tmp_path):
    """Create a sample image for testing."""
    img_path = tmp_path / "test_img.jpg"
    # Create a small RGB test image
    img = Image.new('RGB', (10, 10), color=(73, 109, 137))
    img.save(img_path)
    return str(img_path)

@pytest.fixture
def sample_image_paths(tmp_path):
    """Create a directory with sample images for testing."""
    # Create a structure like:
    # tmp_path/
    #  ├── class1/
    #  │   ├── img1.jpg
    #  │   └── img2.jpg
    #  └── class2/
    #      ├── img3.jpg
    #      └── img4.jpg
    
    class1_dir = tmp_path / "class1"
    class2_dir = tmp_path / "class2"
    class1_dir.mkdir()
    class2_dir.mkdir()
    
    # Create 2 images per class
    paths = []
    for cls_dir in [class1_dir, class2_dir]:
        cls_name = cls_dir.name
        for i in range(2):
            img_path = cls_dir / f"img{len(paths)+1}.jpg"
            img = Image.new('RGB', (10, 10), color=(73, 109, 137))
            img.save(img_path)
            paths.append(str(img_path))
    
    return tmp_path, paths

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with image paths."""
    return pd.DataFrame({
        'path': [
            '/fake/path/img1.jpg',
            '/fake/path/img2.jpg',
            '/fake/path/img3.jpg'
        ],
        'label': [0, 1, 0]
    })

# --- Tests for path_to_dataframe ---

def test_path_to_dataframe(sample_image_paths):
    """Test path_to_dataframe function."""
    folder_path, _ = sample_image_paths
    
    # Without metadata
    df = path_to_dataframe(folder_path, include_metadata=False)
    assert len(df) == 4  # 4 images total
    assert set(df.columns) == {'path', 'name'}
    
    # With metadata
    df_meta = path_to_dataframe(folder_path, include_metadata=True)
    assert len(df_meta) == 4
    assert set(df_meta.columns) == {'path', 'name', 'width', 'height', 'img_mode'}
    assert df_meta['width'].iloc[0] == 10
    assert df_meta['height'].iloc[0] == 10
    assert df_meta['img_mode'].iloc[0] == 'RGB'

def test_path_to_dataframe_nonexistent_folder():
    """Test path_to_dataframe with non-existent folder."""
    df = path_to_dataframe('/nonexistent/folder')
    assert len(df) == 0
    assert set(df.columns) == {'path', 'name'}

def test_path_to_dataframe_with_filter():
    """Test path_to_dataframe with extension filter."""
    with patch('os.walk') as mock_walk, \
         patch('os.path.isdir', return_value=True):  # Mock isdir to return True
        # Mock os.walk to return a test directory structure
        mock_walk.return_value = [
            ('/fake/path', [], ['img1.jpg', 'img2.png', 'doc.txt'])
        ]
        
        # Only jpg should be included
        df = path_to_dataframe('/fake/path', extensions={'.jpg'})
        assert len(df) == 1
        assert df['name'].iloc[0] == 'img1.jpg'
        
        # Both jpg and png should be included
        df = path_to_dataframe('/fake/path', extensions={'.jpg', '.png'})
        assert len(df) == 2

# --- Tests for path_to_dataframe_with_labels ---

def test_path_to_dataframe_with_labels(sample_image_paths):
    """Test path_to_dataframe_with_labels function."""
    folder_path, _ = sample_image_paths
    
    # Without metadata
    df = path_to_dataframe_with_labels(folder_path, include_metadata=False)
    assert len(df) == 4  # 4 images total
    assert set(df.columns) == {'path', 'name', 'label'}
    
    # Check labels
    class1_paths = df[df['label'] == 'class1']['path'].tolist()
    class2_paths = df[df['label'] == 'class2']['path'].tolist()
    assert len(class1_paths) == 2
    assert len(class2_paths) == 2

def test_path_to_dataframe_with_labels_nonexistent_folder():
    """Test path_to_dataframe_with_labels with non-existent folder."""
    df = path_to_dataframe_with_labels('/nonexistent/folder')
    assert len(df) == 0
    assert set(df.columns) == {'path', 'name', 'label'}

# --- Tests for load_image_as_array ---

def test_load_image_as_array(sample_image_path):
    """Test load_image_as_array function."""
    # Test loading with default settings
    img_array = load_image_as_array(sample_image_path)
    assert img_array is not None
    assert img_array.shape == (10, 10, 3)
    assert img_array.dtype == np.uint8
    
    # Test resizing
    img_array = load_image_as_array(sample_image_path, size=(20, 20))
    assert img_array.shape == (20, 20, 3)
    
    # Test normalization
    img_array = load_image_as_array(sample_image_path, normalize=True)
    assert img_array.dtype == np.float32
    assert 0.0 <= img_array.min() and img_array.max() <= 1.0
    
    # Test different mode
    img_array = load_image_as_array(sample_image_path, mode='L')
    assert img_array.shape == (10, 10)  # Grayscale image has no channel dimension

def test_load_image_as_array_errors():
    """Test load_image_as_array error handling."""
    # Test file not found
    result = load_image_as_array('/nonexistent/image.jpg')
    assert result is None
    
    # Test unidentified image
    with patch('PIL.Image.open') as mock_open:
        mock_open.side_effect = UnidentifiedImageError("Test error")
        result = load_image_as_array('path/to/corrupted.jpg')
        assert result is None
    
    # Test general exception
    with patch('PIL.Image.open') as mock_open:
        mock_open.side_effect = Exception("Test error")
        result = load_image_as_array('path/to/problematic.jpg')
        assert result is None

# --- Tests for load_images_from_dataframe ---

def test_load_images_from_dataframe():
    """Test load_images_from_dataframe function."""
    # Create a mock DataFrame and mock the load_image_as_array function
    df = pd.DataFrame({
        'path': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
        'label': [0, 1, 2]
    })
    
    # Create test image arrays
    mock_arrays = {
        'img1.jpg': np.zeros((10, 10, 3), dtype=np.uint8),
        'img2.jpg': np.ones((10, 10, 3), dtype=np.uint8),
        'img3.jpg': np.ones((10, 10, 3), dtype=np.uint8) * 2
    }
    
    def mock_load_image(path, **kwargs):
        return mock_arrays.get(path)
    
    with patch('semiq_ml.image.load_image_as_array', side_effect=mock_load_image):
        # Test without labels
        images = load_images_from_dataframe(df, image_col='path', label_col=None, show_progress=False)
        assert isinstance(images, np.ndarray)
        assert images.shape == (3, 10, 10, 3)
        
        # Test with labels
        images, labels = load_images_from_dataframe(df, image_col='path', label_col='label', show_progress=False)
        assert isinstance(images, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert images.shape == (3, 10, 10, 3)
        assert labels.tolist() == [0, 1, 2]

def test_load_images_from_dataframe_with_errors():
    """Test load_images_from_dataframe with errors."""
    df = pd.DataFrame({
        'path': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
        'label': [0, 1, 2]
    })
    
    # Mock load_image_as_array to simulate some failures
    def mock_load_image(path, **kwargs):
        if path == 'img2.jpg':
            return None  # Simulate failure
        shape = (10, 10, 3)
        if path == 'img1.jpg':
            return np.zeros(shape, dtype=np.uint8)
        return np.ones(shape, dtype=np.uint8)
    
    with patch('semiq_ml.image.load_image_as_array', side_effect=mock_load_image):
        # Test with skip_errors=True
        images, labels = load_images_from_dataframe(
            df, image_col='path', label_col='label', 
            skip_errors=True, show_progress=False
        )
        assert images.shape == (2, 10, 10, 3)
        assert len(labels) == 2
        assert 1 not in labels  # Label 1 corresponds to img2.jpg which failed

def test_load_images_from_dataframe_empty_result():
    """Test load_images_from_dataframe with all images failing to load."""
    df = pd.DataFrame({
        'path': ['img1.jpg', 'img2.jpg'],
        'label': [0, 1]
    })
    
    # Mock load_image_as_array to always return None
    with patch('semiq_ml.image.load_image_as_array', return_value=None):
        # Test without labels
        images = load_images_from_dataframe(df, image_col='path', label_col=None, show_progress=False)
        assert images.size == 0
        
        # Test with labels
        images, labels = load_images_from_dataframe(df, image_col='path', label_col='label', show_progress=False)
        assert images.size == 0
        assert labels.size == 0

# --- Tests for display_images ---

def test_display_images(monkeypatch):
    """Test display_images function."""
    # Mock matplotlib functions to avoid actual plotting
    mock_show = MagicMock()
    monkeypatch.setattr('matplotlib.pyplot.figure', MagicMock())
    monkeypatch.setattr('matplotlib.pyplot.subplot', MagicMock())
    monkeypatch.setattr('matplotlib.pyplot.imshow', MagicMock())
    monkeypatch.setattr('matplotlib.pyplot.title', MagicMock())
    monkeypatch.setattr('matplotlib.pyplot.axis', MagicMock())
    monkeypatch.setattr('matplotlib.pyplot.tight_layout', MagicMock())
    monkeypatch.setattr('matplotlib.pyplot.show', mock_show)
    
    # Create a test image
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    
    # Test with a single image
    display_images(img)
    assert mock_show.call_count == 1
    
    # Reset mock
    mock_show.reset_mock()
    
    # Test with a batch of images
    batch = np.stack([img, img, img])
    display_images(batch)
    assert mock_show.call_count == 1
    
    # Reset mock
    mock_show.reset_mock()
    
    # Test with images and labels
    display_images(batch, labels=[0, 1, 0])
    assert mock_show.call_count == 1
    
    # Reset mock
    mock_show.reset_mock()
    
    # Test with class names
    class_names = {0: 'cat', 1: 'dog'}
    display_images(batch, labels=[0, 1, 0], class_names=class_names)
    assert mock_show.call_count == 1
    
    # Reset mock
    mock_show.reset_mock()
    
    # Test with images and predictions
    display_images(batch, labels=[0, 1, 0], predictions=[0, 0, 1], class_names=class_names)
    assert mock_show.call_count == 1
    
    # Reset mock
    mock_show.reset_mock()
    
    # Test no images
    display_images([])
    # No images should not call show
    assert mock_show.call_count == 0

# --- Tests for sample_images ---

def test_sample_images():
    """Test sample_images function."""
    # Create a batch of images
    imgs = np.stack([
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.ones((10, 10, 3), dtype=np.uint8),
        np.ones((10, 10, 3), dtype=np.uint8) * 2
    ])
    
    # Mock np.random.choice to return predictable selections
    with patch('numpy.random.choice') as mock_choice:
        mock_choice.return_value = imgs[:2]  # Return first two images
        
        # Test sampling with fixed seed for reproducibility
        samples = sample_images(imgs, n_samples=2, random_seed=42)
        assert len(samples) == 2
        
        # Verify random seed was set
        assert np.random.get_state()[1][0] == 42
        
        # Test sampling more than available (should return all)
        # Reset mock to return all images
        mock_choice.return_value = imgs
        samples = sample_images(imgs, n_samples=5)
        assert len(samples) == 3  # Because mock returns all 3 images
    
    # Test with single image - doesn't use np.random.choice
    single_img = np.zeros((10, 10, 3), dtype=np.uint8)
    samples = sample_images(single_img, n_samples=3)
    assert len(samples) == 1
    assert np.array_equal(samples[0], single_img)
    
    # Test with invalid input
    samples = sample_images("not_an_image", n_samples=2)
    assert len(samples) == 0
