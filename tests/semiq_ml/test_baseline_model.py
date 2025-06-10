"""Tests for the baseline_model module."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from typing import Tuple, Dict, Any, List, Optional, Union, cast

from semiq_ml.baseline_model import BaselineModel

# --- Fixtures ---

@pytest.fixture
def classification_data() -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate a simple classification dataset."""
    # Features: 2 numerical, 2 categorical
    X = pd.DataFrame({
        'num_feat1': np.random.normal(0, 1, 100),
        'num_feat2': np.random.normal(0, 1, 100),
        'cat_feat1': np.random.choice(['A', 'B', 'C'], 100),
        'cat_feat2': np.random.choice(['X', 'Y'], 100)
    })
    # Binary target
    y = np.random.randint(0, 2, 100)
    return X, y

@pytest.fixture
def regression_data() -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate a simple regression dataset."""
    # Features: 2 numerical, 2 categorical
    X = pd.DataFrame({
        'num_feat1': np.random.normal(0, 1, 100),
        'num_feat2': np.random.normal(0, 1, 100),
        'cat_feat1': np.random.choice(['A', 'B', 'C'], 100),
        'cat_feat2': np.random.choice(['X', 'Y'], 100)
    })
    # Continuous target
    y = X['num_feat1'] * 2 + X['num_feat2'] * 0.5 + np.random.normal(0, 0.1, 100)
    return X, y

@pytest.fixture
def classification_model() -> BaselineModel:
    """Create a BaselineModel instance for classification."""
    return BaselineModel(task_type='classification', models='gbm')

@pytest.fixture
def regression_model() -> BaselineModel:
    """Create a BaselineModel instance for regression."""
    return BaselineModel(task_type='regression', models='gbm')

# --- Tests for BaselineModel initialization ---

def test_baseline_model_init_classification() -> None:
    """Test initialization of BaselineModel for classification."""
    model = BaselineModel(task_type='classification', metric='accuracy')
    assert model.task_type == 'classification'
    assert model.metric == 'accuracy'

def test_baseline_model_init_regression() -> None:
    """Test initialization of BaselineModel for regression."""
    model = BaselineModel(task_type='regression', metric='r2')
    assert model.task_type == 'regression'
    assert model.metric == 'r2'

def test_baseline_model_init_invalid_task() -> None:
    """Test initialization with invalid task type."""
    with pytest.raises(ValueError):
        BaselineModel(task_type='invalid_task')

def test_baseline_model_init_invalid_models() -> None:
    """Test initialization with invalid models parameter."""
    with pytest.raises(ValueError):
        BaselineModel(models='invalid_models')

def test_baseline_model_init_invalid_metric() -> None:
    """Test initialization with invalid metric."""
    with pytest.raises(ValueError):
        BaselineModel(task_type='classification', metric='invalid_metric')

def test_baseline_model_models_gbm() -> None:
    """Test initialization with 'gbm' models set."""
    model = BaselineModel(models='gbm')
    model._initialize_models()  # Make sure models are initialized
    model_names = list(model.models_to_run.keys())
    assert len(model_names) <= 5  # Fewer models than 'all'
    assert any(name for name in model_names if 'XGBoost' in name or 'LGBM' in name or 'CatBoost' in name)

def test_baseline_model_models_trees() -> None:
    """Test initialization with 'trees' models set."""
    model = BaselineModel(models='trees')
    model._initialize_models()  # Make sure models are initialized
    model_names = list(model.models_to_run.keys())
    assert any(name for name in model_names if 'Random Forest' in name)
    assert any(name for name in model_names if 'Decision Tree' in name)

def test_baseline_model_models_all() -> None:
    """Test initialization with 'all' models set."""
    model = BaselineModel(models='all')
    model._initialize_models()  # Make sure models are initialized
    model_names = list(model.models_to_run.keys())
    assert len(model_names) > 5  # More models than just trees or boosting

# --- Tests for model fitting and evaluation ---

def test_fit_classification(classification_data: Tuple[pd.DataFrame, np.ndarray]) -> None:
    """Test fit method with classification data."""
    X, y = classification_data
    model = BaselineModel(task_type='classification', models='gbm')
    results = model.fit(X, y, validation_size=0.2)
    
    # Check results
    assert isinstance(results, dict)
    assert len(results) > 0
    
    # Check model presence
    for model_name, model_info in results.items():
        assert 'train_score' in model_info
        assert 'val_score' in model_info
        assert 'fit_time' in model_info
        assert 'model' in model_info
        assert 'preprocessor' in model_info

def test_fit_regression(regression_data: Tuple[pd.DataFrame, np.ndarray]) -> None:
    """Test fit method with regression data."""
    X, y = regression_data
    model = BaselineModel(task_type='regression', models='gbm')
    results = model.fit(X, y, validation_size=0.2)
    
    # Check results
    assert isinstance(results, dict)
    assert len(results) > 0

def test_get_model(classification_data: Tuple[pd.DataFrame, np.ndarray]) -> None:
    """Test get_model method."""
    X, y = classification_data
    model = BaselineModel(task_type='classification', models='gbm')
    model.fit(X, y)
    
    # Try getting a fitted model
    first_model_name = list(model.results.keys())[0]
    fitted_model = model.get_model(first_model_name)
    assert fitted_model is not None
    
    # Try getting a non-existent model
    with pytest.raises(ValueError):
        model.get_model('non_existent_model')

def test_evaluate_all(classification_data: Tuple[pd.DataFrame, np.ndarray]) -> None:
    """Test evaluate_all method."""
    X, y = classification_data
    model = BaselineModel(task_type='classification', models='gbm')
    model.fit(X, y)
    
    # Evaluate on new data
    eval_results = model.evaluate_all(X, y)
    assert isinstance(eval_results, dict)
    assert len(eval_results) > 0
    
    # Check that scores were calculated
    for model_name, score in eval_results.items():
        assert isinstance(score, (float, int))

def test_get_results(classification_data: Tuple[pd.DataFrame, np.ndarray]) -> None:
    """Test get_results method."""
    X, y = classification_data
    model = BaselineModel(task_type='classification', models='gbm')
    model.fit(X, y)
    
    # Get results as DataFrame
    results_df = model.get_results()
    assert isinstance(results_df, pd.DataFrame)
    assert 'model' in results_df.columns
    assert 'train_score' in results_df.columns
    assert 'val_score' in results_df.columns

# --- Tests for visualization methods ---

def test_roc_curves(classification_data: Tuple[pd.DataFrame, np.ndarray], monkeypatch: pytest.MonkeyPatch) -> None:
    """Test roc_curves method."""
    # Mock plt.figure and plt.show to avoid actual plotting
    mock_plot = MagicMock()
    monkeypatch.setattr('matplotlib.pyplot.figure', MagicMock())
    monkeypatch.setattr('matplotlib.pyplot.plot', mock_plot)
    monkeypatch.setattr('matplotlib.pyplot.legend', MagicMock())
    monkeypatch.setattr('matplotlib.pyplot.xlabel', MagicMock())
    monkeypatch.setattr('matplotlib.pyplot.ylabel', MagicMock())
    monkeypatch.setattr('matplotlib.pyplot.title', MagicMock())
    monkeypatch.setattr('matplotlib.pyplot.grid', MagicMock())
    monkeypatch.setattr('matplotlib.pyplot.show', MagicMock())
    
    # Create a model and fit it
    X, y = classification_data
    model = BaselineModel(task_type='classification', models='gbm')
    model.fit(X, y)
    
    # Generate ROC curves
    model.roc_curves(X, y)
    # Assert plot was called at least once
    assert mock_plot.called

def test_precision_recall_curves(classification_data: Tuple[pd.DataFrame, np.ndarray], monkeypatch: pytest.MonkeyPatch) -> None:
    """Test precision_recall_curves method."""
    # Mock plt functions
    mock_plot = MagicMock()
    monkeypatch.setattr('matplotlib.pyplot.figure', MagicMock())
    monkeypatch.setattr('matplotlib.pyplot.plot', mock_plot)
    monkeypatch.setattr('matplotlib.pyplot.legend', MagicMock())
    monkeypatch.setattr('matplotlib.pyplot.xlabel', MagicMock())
    monkeypatch.setattr('matplotlib.pyplot.ylabel', MagicMock())
    monkeypatch.setattr('matplotlib.pyplot.title', MagicMock())
    monkeypatch.setattr('matplotlib.pyplot.grid', MagicMock())
    monkeypatch.setattr('matplotlib.pyplot.show', MagicMock())
    
    # Create a model and fit it
    X, y = classification_data
    model = BaselineModel(task_type='classification', models='gbm')
    model.fit(X, y)
    
    # Generate PR curves
    model.precision_recall_curves(X, y)
    # Assert plot was called at least once
    assert mock_plot.called
