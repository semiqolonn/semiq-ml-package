import pytest
import numpy as np
import pandas as pd

from semiq_ml.tuning import BaseOptimizer, RandomSearchOptimizer, GridSearchOptimizer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

class TestBaseOptimizer:
    def test_initialization(self):
        """Test BaseOptimizer initialization."""
        # Create a concrete test subclass that implements the abstract method
        class TestOptimizer(BaseOptimizer):
            def _get_default_param_config(self):
                return {"Mock": {"param": [1, 2]}}
                
        # Use the test subclass
        optimizer = TestOptimizer(
            search_strategy_cv=RandomizedSearchCV, 
            task_type="classification",
            n_iter_or_None=10
        )
        assert optimizer.task_type == "classification"
        assert optimizer.metric == "accuracy"
        assert optimizer.cv == 5
        assert optimizer.n_iter_or_None == 10
        assert optimizer.search_strategy_cv == RandomizedSearchCV
    
    def test_abstract_method(self):
        """Test that abstract methods raise NotImplementedError."""
        class IncompleteOptimizer(BaseOptimizer):
            pass  # Doesn't implement the abstract method
            
        # Should raise NotImplementedError when instantiated
        with pytest.raises(NotImplementedError):
            IncompleteOptimizer(
                search_strategy_cv=RandomizedSearchCV, 
                task_type="classification"
            )
            
class TestRandomSearchOptimizer:
    def test_initialization(self):
        """Test RandomSearchOptimizer initialization."""
        optimizer = RandomSearchOptimizer()
        assert optimizer.task_type == "classification"
        assert optimizer.metric == "accuracy"
        assert optimizer.n_iter_or_None == 10
        assert optimizer.search_strategy_cv == RandomizedSearchCV
        
        # Test with custom parameters
        optimizer = RandomSearchOptimizer(
            task_type="regression", 
            metric="r2",
            n_iter=5,
            cv=3
        )
        assert optimizer.task_type == "regression"
        assert optimizer.metric == "r2"
        assert optimizer.n_iter_or_None == 5
        assert optimizer.cv == 3
        
    def test_param_config(self):
        """Test parameter configuration generation."""
        # Classification
        cls_optimizer = RandomSearchOptimizer()
        cls_params = cls_optimizer._get_default_param_config()
        
        # Check if all expected models have parameter configurations
        assert "Logistic Regression" in cls_params
        assert "SVC" in cls_params
        assert "KNN" in cls_params
        assert "Decision Tree" in cls_params
        assert "Random Forest" in cls_params
        assert "LGBM" in cls_params
        assert "XGBoost" in cls_params
        assert "CatBoost" in cls_params
        
        # Regression
        reg_optimizer = RandomSearchOptimizer(task_type="regression")
        reg_params = reg_optimizer._get_default_param_config()
        
        # Check regression-specific models
        assert "Linear Regression" in reg_params
        assert "SVR" in reg_params
        
    def test_tune_model_small_data(self, sample_classification_data):
        """Test model tuning with a small dataset."""
        X, y = sample_classification_data
        
        # Use a very small configuration for testing
        optimizer = RandomSearchOptimizer(n_iter=2, cv=2, random_state=42)
        
        # Limit parameters to speed up test (just using Logistic Regression)
        optimizer.param_config = {
            "Logistic Regression": {'C': [0.1, 1.0], 'penalty': ['l2']}
        }
        
        # Tune a specific model
        tuned_model = optimizer.tune_model("Logistic Regression", X, y, validation_size=0.3)
        
        # Check that tuning worked
        assert tuned_model is not None
        assert "Logistic Regression" in optimizer.tuned_models
        results = optimizer.tuned_models["Logistic Regression"]
        assert "best_params" in results
        assert "score_on_holdout" in results
        
    def test_update_param_config(self):
        """Test updating parameter configuration."""
        optimizer = RandomSearchOptimizer()
        
        # Update an existing model's parameters
        new_params = {'C': [0.001, 0.01, 0.1], 'penalty': ['l1', 'l2']}
        optimizer.update_param_config("Logistic Regression", new_params)
        
        # Check that the update worked
        assert optimizer.param_config["Logistic Regression"] == new_params
        
        # Invalid model should raise error
        with pytest.raises(ValueError):
            optimizer.update_param_config("InvalidModel", {})

class TestGridSearchOptimizer:
    def test_initialization(self):
        """Test GridSearchOptimizer initialization."""
        optimizer = GridSearchOptimizer()
        assert optimizer.task_type == "classification"
        assert optimizer.metric == "accuracy"
        assert optimizer.search_strategy_cv == GridSearchCV
        assert optimizer.n_iter_or_None is None # Grid search doesn't use n_iter
        
    def test_param_config(self):
        """Test parameter configuration for grid search."""
        # Classification
        cls_optimizer = GridSearchOptimizer()
        cls_params = cls_optimizer._get_default_param_config()
        
        # Check a few models have parameter configurations suitable for grid search
        assert "Logistic Regression" in cls_params
        assert isinstance(cls_params["Logistic Regression"]['C'], list)  # Should be a list not distribution
        
        # Regression
        reg_optimizer = GridSearchOptimizer(task_type="regression")
        reg_params = reg_optimizer._get_default_param_config()
        
        # Check regression params exist and are in grid format (lists)
        assert "SVR" in reg_params
        assert isinstance(reg_params["SVR"]['C'], list)
        
    def test_tune_all_models_small_subset(self, sample_regression_data):
        """Test tuning multiple models with a small configuration."""
        X, y = sample_regression_data
        
        optimizer = GridSearchOptimizer(task_type="regression", cv=2, random_state=42)
        
        # Use a model that has parameters to tune
        optimizer.param_config = {
            "SVR": {'C': [0.1, 1.0]},  # Simple parameter grid for SVR
        }
        
        # Tune the model
        results = optimizer.tune_all_models(
            X, y, 
            validation_size=0.3,
            models_to_tune=["SVR"]
        )
        
        # Get and check tuning results
        results_df = optimizer.get_tuning_results()
        assert isinstance(results_df, pd.DataFrame)
        assert "model_name" in results_df.columns
