import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from ml_helper.baseline_model import BaselineModel

class TestBaselineModel(unittest.TestCase):
    
    def setUp(self):
        """Set up test data for classification and regression tasks"""
        # Create synthetic classification data
        X_cls, y_cls = make_classification(
            n_samples=100, 
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        self.X_cls = pd.DataFrame(X_cls, columns=[f'feature_{i}' for i in range(X_cls.shape[1])])
        self.y_cls = pd.Series(y_cls, name='target')
        
        # Create synthetic regression data
        X_reg, y_reg = make_regression(
            n_samples=100,
            n_features=5, 
            n_informative=3,
            noise=0.1,
            random_state=42
        )
        self.X_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(X_reg.shape[1])])
        self.y_reg = pd.Series(y_reg, name='target')

    def test_initialization(self):
        """Test that the BaselineModel initializes correctly."""
        # Test classification model initialization
        cls_model = BaselineModel(task_type='classification', metric='accuracy', random_state=42)
        self.assertEqual(cls_model.task_type, 'classification')
        self.assertEqual(cls_model.metric, 'accuracy')
        self.assertTrue(cls_model.maximize_metric)
        
        # Test regression model initialization
        reg_model = BaselineModel(task_type='regression', metric='r2', random_state=42)
        self.assertEqual(reg_model.task_type, 'regression')
        self.assertEqual(reg_model.metric, 'r2')
        self.assertTrue(reg_model.maximize_metric)

    def test_invalid_task_type(self):
        """Test that invalid task types raise a ValueError."""
        with self.assertRaises(ValueError):
            BaselineModel(task_type='invalid_type')

    def test_invalid_metric(self):
        """Test that invalid metrics raise a ValueError."""
        with self.assertRaises(ValueError):
            BaselineModel(task_type='classification', metric='invalid_metric')
        
        with self.assertRaises(ValueError):
            BaselineModel(task_type='regression', metric='invalid_metric')

    def test_fit_classification(self):
        """Test model fitting for classification."""
        cls_model = BaselineModel(task_type='classification', metric='accuracy', random_state=42)
        best_model = cls_model.fit(self.X_cls, self.y_cls, validation_size=0.2)
        
        self.assertIsNotNone(best_model, "Best model should not be None")
        self.assertIsNotNone(cls_model.best_score_, "Best score should be set")
        self.assertGreater(len(cls_model.results), 0, "Results should not be empty")

    def test_fit_regression(self):
        """Test model fitting for regression."""
        reg_model = BaselineModel(task_type='regression', metric='r2', random_state=42)
        best_model = reg_model.fit(self.X_reg, self.y_reg, validation_size=0.2)
        
        self.assertIsNotNone(best_model, "Best model should not be None")
        self.assertIsNotNone(reg_model.best_score_, "Best score should be set")
        self.assertGreater(len(reg_model.results), 0, "Results should not be empty")

    def test_get_results(self):
        """Test getting results DataFrame."""
        cls_model = BaselineModel(task_type='classification', random_state=42)
        
        # Before fitting, should return empty DataFrame
        empty_df = cls_model.get_results()
        self.assertTrue(empty_df.empty)
        
        # After fitting, should return non-empty DataFrame
        cls_model.fit(self.X_cls, self.y_cls, validation_size=0.2)
        results_df = cls_model.get_results()
        self.assertFalse(results_df.empty)
        self.assertIn('model', results_df.columns)
        self.assertIn('score', results_df.columns)
        self.assertIn('time', results_df.columns)

    def test_get_model(self):
        """Test retrieving a specific model by name."""
        cls_model = BaselineModel(task_type='classification', random_state=42)
        cls_model.fit(self.X_cls, self.y_cls, validation_size=0.2)
        
        # Try to get a valid model
        model_name = list(cls_model.results.keys())[0]  # Get the first model name
        model = cls_model.get_model(model_name)
        self.assertIsNotNone(model)
        
        # Try to get an invalid model
        with self.assertRaises(ValueError):
            cls_model.get_model('NonExistentModel')

    def test_evaluate_all(self):
        """Test evaluating all models."""
        cls_model = BaselineModel(task_type='classification', random_state=42)
        cls_model.fit(self.X_cls, self.y_cls, validation_size=0.2)
        
        # Evaluate on the same data
        metrics = cls_model.evaluate_all(self.X_cls, self.y_cls)
        self.assertGreater(len(metrics), 0, "Metrics should not be empty")
        
        # Check that each successful model has metrics
        for name, model_info in cls_model.results.items():
            if model_info['model'] is not None:
                self.assertIn(name, metrics)
                self.assertIn('accuracy', metrics[name])

if __name__ == '__main__':
    unittest.main()
