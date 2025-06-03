import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from PIL import Image

@pytest.fixture
def sample_classification_data():
    """Fixture providing a small classification dataset."""
    # Generate a simple dataset with two features and binary target
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    
    df = pd.DataFrame(X, columns=['feature1', 'feature2'])
    return df, pd.Series(y, name='target')

@pytest.fixture
def sample_regression_data():
    """Fixture providing a small regression dataset."""
    # Generate a simple dataset with two features and continuous target
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 0.1, 100)
    
    df = pd.DataFrame(X, columns=['feature1', 'feature2'])
    return df, pd.Series(y, name='target')

@pytest.fixture
def sample_mixed_data():
    """Fixture providing a dataset with mixed feature types."""
    np.random.seed(42)
    # Create numeric features
    num_samples = 100
    numeric1 = np.random.rand(num_samples)
    numeric2 = np.random.rand(num_samples)
    
    # Create categorical features
    categories = ['A', 'B', 'C']
    cat1 = np.random.choice(categories, num_samples)
    cat2 = np.random.choice(['X', 'Y'], num_samples)
    
    # Create target
    y = (numeric1 + numeric2 > 1).astype(int)
    
    df = pd.DataFrame({
        'numeric1': numeric1, 
        'numeric2': numeric2,
        'categorical1': cat1,
        'categorical2': cat2
    })
    
    return df, pd.Series(y, name='target')

@pytest.fixture
def sample_image_directory():
    """Create a temporary directory with test images and return its path."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create subdirectories with class names
        os.makedirs(os.path.join(tmpdirname, 'class1'), exist_ok=True)
        os.makedirs(os.path.join(tmpdirname, 'class2'), exist_ok=True)
        
        # Create test images in each class
        for i in range(3):
            # Create RGB image for class1
            img1 = Image.new('RGB', (50, 50), color=(255, 0, 0))
            img1_path = os.path.join(tmpdirname, 'class1', f'img{i}.jpg')
            img1.save(img1_path)
            
            # Create RGB image for class2
            img2 = Image.new('RGB', (50, 50), color=(0, 0, 255))
            img2_path = os.path.join(tmpdirname, 'class2', f'img{i}.jpg')
            img2.save(img2_path)
            
            # Create a grayscale image too
            img3 = Image.new('L', (50, 50), color=128)
            img3_path = os.path.join(tmpdirname, 'class1', f'img_gray{i}.jpg')
            img3.save(img3_path)
            
        yield tmpdirname
