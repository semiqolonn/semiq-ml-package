import pytest
import numpy as np
import pandas as pd

from semiq_ml.baseline_model import BaselineModel

class TestBaselineModel:
    def test_initialization(self):
        """Test BaselineModel initialization with different parameters."""
        # Default initialization for classification
        model = BaselineModel()
        assert model.task_type == "classification"
        assert model.metric == "accuracy"
        assert model.maximize_metric is True
        
        # Initialization for regression
        reg_model = BaselineModel(task_type="regression")
        assert reg_model.task_type == "regression"
        assert reg_model.metric == "neg_root_mean_squared_error"
        assert reg_model.maximize_metric is True
        
        # Custom metric for classification
        cls_model = BaselineModel(metric="f1_weighted")
        assert cls_model.metric == "f1_weighted"
        
        # Invalid task type should raise error
        with pytest.raises(ValueError):
            BaselineModel(task_type="invalid_type")
            
        # Invalid metric should raise error
        with pytest.raises(ValueError):
            BaselineModel(metric="invalid_metric")

    def test_model_initialization(self):
        """Test if all models are properly initialized."""
        model = BaselineModel()
        models = model.models_to_run
        
        # Check if all expected models are initialized
        assert "Logistic Regression" in models
        assert "SVC" in models
        assert "KNN" in models
        assert "Decision Tree" in models
        assert "Random Forest" in models
        assert "LGBM" in models
        assert "XGBoost" in models
        assert "CatBoost" in models
        
        # Check regression models
        reg_model = BaselineModel(task_type="regression")
        reg_models = reg_model.models_to_run
        
        assert "Linear Regression" in reg_models
        assert "SVR" in reg_models
        assert "KNN" in reg_models
        assert "Decision Tree" in reg_models
        assert "Random Forest" in reg_models
        assert "LGBM" in reg_models
        assert "XGBoost" in reg_models
        assert "CatBoost" in reg_models

    def test_preprocessor_selection(self):
        """Test model type detection for preprocessor selection."""
        model = BaselineModel()
        
        assert model._get_model_type("Logistic Regression") == "general_ohe"
        assert model._get_model_type("SVC") == "distance_kernel"
        assert model._get_model_type("KNN") == "distance_kernel"
        assert model._get_model_type("Random Forest") == "general_ohe"
        assert model._get_model_type("CatBoost") == "catboost_internal"

    def test_classification_fit(self, sample_classification_data):
        """Test fitting a classification model."""
        X, y = sample_classification_data
        model = BaselineModel(random_state=42)
        
        # Fit the model
        fitted_model = model.fit(X, y, validation_size=0.2)
        
        # Check if fit returned a model
        assert fitted_model is not None
        
        # Check that results were stored
        assert len(model.results) > 0
        
        # Check that best model and score were stored
        assert model.best_model_ is not None
        assert model.best_score_ is not None
        
        # Check if get_results returns a DataFrame
        results_df = model.get_results()
        assert isinstance(results_df, pd.DataFrame)
        assert "model" in results_df.columns
        assert "score" in results_df.columns

    def test_regression_fit(self, sample_regression_data):
        """Test fitting a regression model."""
        X, y = sample_regression_data
        model = BaselineModel(task_type="regression", random_state=42)
        
        # Fit the model
        model.fit(X, y, validation_size=0.2)
        
        # Check if results were stored
        assert len(model.results) > 0
        
        # Check if get_results returns a DataFrame
        results_df = model.get_results()
        assert isinstance(results_df, pd.DataFrame)
        
    def test_evaluate_all(self, sample_classification_data):
        """Test evaluating all models on test data."""
        X, y = sample_classification_data
        model = BaselineModel(random_state=42)
        
        # First fit the model
        model.fit(X, y, validation_size=0.2)
        
        # Now evaluate on the same data (for testing purposes)
        eval_results = model.evaluate_all(X, y)
        
        # Check that evaluation results are returned
        assert isinstance(eval_results, pd.DataFrame)
        assert len(eval_results) > 0
        
        # Classification metrics should be in results
        if "accuracy" in eval_results.columns:
            assert not eval_results["accuracy"].isnull().all()

    def test_mixed_data_preprocessing(self, sample_mixed_data):
        """Test handling of mixed data types (numeric and categorical)."""
        X, y = sample_mixed_data
        model = BaselineModel(random_state=42)
        
        # Fit should process categorical features correctly
        model.fit(X, y, validation_size=0.2)
        
        # Check that results were stored
        assert len(model.results) > 0

    def test_get_model(self, sample_classification_data):
        """Test retrieving a specific trained model."""
        X, y = sample_classification_data
        model = BaselineModel(random_state=42)
        
        # First fit the model
        model.fit(X, y, validation_size=0.2)
        
        # Try to get a specific model
        try:
            lgbm = model.get_model("LGBM")
            assert lgbm is not None
        except ValueError:
            # If LGBM failed to train, try another model
            for name in model.results:
                if model.results[name].get("model") is not None:
                    retrieved_model = model.get_model(name)
                    assert retrieved_model is not None
                    break
        
        # Invalid model name should raise error
        with pytest.raises(ValueError):
            model.get_model("NonExistentModel")

    def test_roc_curves(self, sample_classification_data):
        """Test ROC curve generation function."""
        X, y = sample_classification_data
        model = BaselineModel(random_state=42)
        
        # First fit the model
        model.fit(X, y, validation_size=0.2)
        
        # ROC curves should not raise error (will not test actual plotting)
        # Just verify the function runs without errors
        try:
            model.roc_curves(X, y)
        except Exception as e:
            # Matplotlib might not be available in test environment
            # So we'll just check it's not a code logic error
            assert "matplotlib" in str(e).lower() or "display" in str(e).lower()
