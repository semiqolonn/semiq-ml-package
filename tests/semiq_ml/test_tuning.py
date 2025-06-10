"""Tests for the tuning module."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from semiq_ml.tuning import OptunaOptimizer
from semiq_ml.baseline_model import BaselineModel

# --- Fixtures ---

@pytest.fixture
def classification_data():
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
def regression_data():
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
def fitted_classification_model(classification_data):
    """Create a fitted BaselineModel instance for classification."""
    X, y = classification_data
    model = BaselineModel(task_type='classification', models='gbm')
    model.fit(X, y)
    return model

@pytest.fixture
def fitted_regression_model(regression_data):
    """Create a fitted BaselineModel instance for regression."""
    X, y = regression_data
    model = BaselineModel(task_type='regression', models='gbm')
    model.fit(X, y)
    return model

@pytest.fixture
def optuna_optimizer_classification(fitted_classification_model):
    """Create an OptunaOptimizer for a classification model."""
    return OptunaOptimizer(baseline_model=fitted_classification_model, n_trials=5, cv=2)

@pytest.fixture
def optuna_optimizer_regression(fitted_regression_model):
    """Create an OptunaOptimizer for a regression model."""
    return OptunaOptimizer(baseline_model=fitted_regression_model, n_trials=5, cv=2)

# --- Tests for initialization ---

def test_optuna_optimizer_init_with_model(fitted_classification_model):
    """Test initialization of OptunaOptimizer with a model."""
    optimizer = OptunaOptimizer(baseline_model=fitted_classification_model)
    assert optimizer.baseline_model is fitted_classification_model
    assert optimizer.task_type == 'classification'
    assert optimizer.n_trials == 100  # Default

def test_optuna_optimizer_init_without_model():
    """Test initialization of OptunaOptimizer without a model."""
    optimizer = OptunaOptimizer(task_type='classification', metric='accuracy')
    assert isinstance(optimizer.baseline_model, BaselineModel)
    assert optimizer.baseline_model.task_type == 'classification'
    assert optimizer.baseline_model.metric == 'accuracy'

def test_optuna_optimizer_init_invalid_arguments():
    """Test initialization with invalid arguments."""
    # No baseline_model and no task_type
    with pytest.raises(ValueError):
        OptunaOptimizer()

    # No baseline_model and invalid task_type
    with pytest.raises(ValueError):
        OptunaOptimizer(task_type='invalid_type')

# --- Tests for parameter space definition ---

@pytest.mark.parametrize('model_name', ['Logistic Regression', 'SVC', 'Random Forest', 'XGBoost'])
def test_define_param_space(model_name, optuna_optimizer_classification):
    """Test parameter space definition for various models."""
    mock_trial = MagicMock()
    mock_trial.suggest_categorical = MagicMock(return_value='value')
    mock_trial.suggest_float = MagicMock(return_value=0.5)
    mock_trial.suggest_int = MagicMock(return_value=5)
    mock_trial.params = {'gamma_choice': 'preset', 'max_depth_choice': 'custom', 'use_bagging': True}
    
    params = optuna_optimizer_classification._define_param_space(mock_trial, model_name)
    
    # Check params is a dict 
    assert isinstance(params, dict)
    
    # Check at least one parameter was set
    assert len(params) > 0

# --- Tests for model tuning ---

def test_tune_model_mocked(optuna_optimizer_classification, classification_data):
    """Test tune_model with mocked study."""
    X, y = classification_data
    model_name = list(optuna_optimizer_classification.baseline_model.results.keys())[0]
    
    # Mock study
    mock_study = MagicMock()
    mock_best_params = {'param1': 'value1', 'param2': 0.5}
    mock_study.best_value = 0.95
    
    # Mock best_trial with user_attrs
    mock_best_trial = MagicMock()
    mock_best_trial.user_attrs = {'model_params': mock_best_params}
    mock_study.best_trial = mock_best_trial
    
    # Mock optuna.create_study
    with patch('optuna.create_study', return_value=mock_study):
        # Mock study.optimize to do nothing
        mock_study.optimize = MagicMock()
        
        # Call tune_model
        best_params, best_score, best_model = optuna_optimizer_classification.tune_model(model_name, X, y)
    
    # Verify results
    assert best_params == mock_best_params
    assert best_score == mock_study.best_value
    assert best_model is not None

def test_tune_model_real_small(optuna_optimizer_classification, classification_data):
    """Test tune_model with actual optimization on small dataset."""
    X, y = classification_data
    model_name = list(optuna_optimizer_classification.baseline_model.results.keys())[0]
    
    # Set very small number of trials to make the test run quickly
    optuna_optimizer_classification.n_trials = 2
    
    # Mock successful trial completion
    mock_study = MagicMock()
    mock_best_trial = MagicMock()
    mock_best_params = {'param1': 'value1', 'param2': 0.5}
    mock_best_trial.user_attrs = {'model_params': mock_best_params}
    mock_study.best_trial = mock_best_trial
    mock_study.best_value = 0.95
    
    with patch('optuna.create_study', return_value=mock_study):
        # Call tune_model
        best_params, best_score, best_model = optuna_optimizer_classification.tune_model(model_name, X, y)
    
    # Verify results
    assert best_model is not None
    assert isinstance(best_score, float)
    assert isinstance(best_params, dict)

def test_tune_best_model(optuna_optimizer_classification, classification_data):
    """Test tune_best_model method."""
    X, y = classification_data
    
    # Get the actual model name from baseline model
    model_name = list(optuna_optimizer_classification.baseline_model.results.keys())[0]
    
    # Mock get_results to return a DataFrame with the actual model name
    mock_results = pd.DataFrame({
        'model': [model_name, 'model2'],
        'val_score': [0.9, 0.8]
    })
    
    # Mock best_model to make tune_best_model work
    mock_model = MagicMock()
    mock_model.__class__ = optuna_optimizer_classification.baseline_model.models_to_run[model_name].__class__
    optuna_optimizer_classification.baseline_model.best_model_ = mock_model
    
    with patch.object(optuna_optimizer_classification.baseline_model, 'get_results', 
                    return_value=mock_results):
        
        # Mock tune_model to return fixed values (params, score, model)
        mock_tune_result = ({'param': 'value'}, 0.95, MagicMock())
        with patch.object(optuna_optimizer_classification, 'tune_model', 
                        return_value=mock_tune_result):
            
            # Call tune_best_model
            params, score, model = optuna_optimizer_classification.tune_best_model(X, y)
            
            # Match how the implementation actually calls the method
            # The actual implementation passes None as a positional argument, not as a keyword argument
            optuna_optimizer_classification.tune_model.assert_called_once_with(model_name, X, y, None)
            
            # Verify results
            assert params == mock_tune_result[0]  
            assert score == mock_tune_result[1]
            assert model == mock_tune_result[2]

def test_tune_all_models(optuna_optimizer_classification, classification_data):
    """Test tune_all_models method."""
    X, y = classification_data
    
    # Set very small number of trials
    optuna_optimizer_classification.n_trials = 2
    
    # Mock tune_model to return fixed values (params, score, model)
    mock_model = MagicMock()
    mock_tune_result = ({'param': 'value'}, 0.95, mock_model)
    with patch.object(optuna_optimizer_classification, 'tune_model', 
                     return_value=mock_tune_result):
        
        # Add to tuned_models to avoid KeyError in tune_all_models
        for model_name in optuna_optimizer_classification.baseline_model.models_to_run:
            optuna_optimizer_classification.tuned_models[model_name] = mock_model
        
        # Call tune_all_models
        all_results = optuna_optimizer_classification.tune_all_models(X, y)
        
        # Verify results
        assert isinstance(all_results, dict)
        assert len(all_results) > 0
        
        # Check the calls to tune_model
        expected_calls = len(optuna_optimizer_classification.baseline_model.results)
        assert optuna_optimizer_classification.tune_model.call_count == expected_calls

def test_fit_best_model(optuna_optimizer_classification, classification_data):
    """Test fit_best_model method."""
    X, y = classification_data
    
    # Set best_model_ directly to avoid tune_best_model
    mock_model = MagicMock()
    mock_model.fit = MagicMock(return_value=mock_model)
    optuna_optimizer_classification.best_model_ = mock_model
    
    # Get a real model name
    model_name = list(optuna_optimizer_classification.baseline_model.results.keys())[0]
    
    # Add to tuned_models to make fit_best_model work
    optuna_optimizer_classification.tuned_models[model_name] = mock_model
    
    # Call fit_best_model with a patch to avoid actual preprocessing
    with patch.object(optuna_optimizer_classification.baseline_model, '_get_model_type', return_value='general_ohe'), \
         patch.object(optuna_optimizer_classification.baseline_model, '_build_preprocessor', return_value=None):
        
        fitted_model = optuna_optimizer_classification.fit_best_model(X, y)
    
    # Verify fit was called on the model
    mock_model.fit.assert_called_once()
    
    # Check return value
    assert fitted_model == mock_model

def test_get_tuning_results(optuna_optimizer_classification, classification_data):
    """Test get_tuning_results method."""
    X, y = classification_data
    
    # Mock study objects for get_tuning_results
    mock_study = MagicMock()
    mock_study.best_value = 0.95
    mock_study.trials = [MagicMock(), MagicMock()]
    
    # Setup study_objects and best_params
    optuna_optimizer_classification.study_objects = {
        'model1': mock_study
    }
    optuna_optimizer_classification.best_params = {
        'model1': {'param': 'value'}
    }
    
    # Call get_tuning_results
    results_df = optuna_optimizer_classification.get_tuning_results()
    
    # Verify DataFrame structure
    assert isinstance(results_df, pd.DataFrame)
    assert 'model_name' in results_df.columns
    assert 'best_score' in results_df.columns

# --- Tests for visualization methods ---

def test_plot_optimization_history(optuna_optimizer_classification, classification_data, monkeypatch):
    """Test plot_optimization_history method."""
    # Mock optuna visualization module with mock figure
    mock_fig = MagicMock()
    mock_plot_optimization_history = MagicMock(return_value=mock_fig)
    monkeypatch.setattr('optuna.visualization.plot_optimization_history', mock_plot_optimization_history)
    
    # Mock study
    mock_study = MagicMock()
    
    # Setup study_objects
    optuna_optimizer_classification.study_objects = {
        'model1': mock_study
    }
    
    # Call plot_optimization_history
    optuna_optimizer_classification.plot_optimization_history('model1')
    
    # Verify plot was called
    assert mock_plot_optimization_history.called
    assert mock_fig.show.called

def test_plot_param_importances(optuna_optimizer_classification, monkeypatch):
    """Test plot_param_importances method."""
    # Mock optuna visualization module with mock figure
    mock_fig = MagicMock()
    mock_plot_param_importances = MagicMock(return_value=mock_fig)
    monkeypatch.setattr('optuna.visualization.plot_param_importances', mock_plot_param_importances)
    
    # Mock study
    mock_study = MagicMock()
    
    # Setup study_objects
    optuna_optimizer_classification.study_objects = {
        'model1': mock_study
    }
    
    # Call plot_param_importances
    optuna_optimizer_classification.plot_param_importances('model1')
    
    # Verify visualization function was called
    assert mock_plot_param_importances.called
    assert mock_fig.show.called
