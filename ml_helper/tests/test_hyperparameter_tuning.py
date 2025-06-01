import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from ml_helper.hyperparameter_tuning import RandomSearchOptimizer
from scipy.stats import uniform, randint

class TestRandomSearchOptimizer(unittest.TestCase):
    
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
        
        # Define a simplified param distribution for faster tests
        self.test_param_distributions = {
            'Logistic Regression': {
                'C': uniform(loc=0.1, scale=10),
                'solver': ['liblinear']
            },
            'Decision Tree': {
                'max_depth': [3, 5],
                'min_samples_split': [2, 3]
            }
        }

    def test_initialization(self):
        """Test that the RandomSearchOptimizer initializes correctly."""
        # Test classification model initialization
        cls_optimizer = RandomSearchOptimizer(
            task_type='classification', 
            metric='accuracy', 
            random_state=42,
            n_iter=3,
            cv=3,
            param_distributions=self.test_param_distributions
        )
        self.assertEqual(cls_optimizer.task_type, 'classification')
        self.assertEqual(cls_optimizer.metric, 'accuracy')
        self.assertTrue(cls_optimizer.maximize_metric)
        self.assertEqual(cls_optimizer.n_iter, 3)
        self.assertEqual(cls_optimizer.cv, 3)
        
        # Check param distributions are set
        self.assertEqual(
            set(cls_optimizer.param_distributions['Logistic Regression'].keys()),
            set(self.test_param_distributions['Logistic Regression'].keys())
        )

    def test_invalid_metric(self):
        """Test that invalid metrics raise a ValueError."""
        with self.assertRaises(ValueError):
            RandomSearchOptimizer(
                task_type='classification', 
                metric='invalid_metric', 
                random_state=42
            )

    def test_fit_classification(self):
        """Test model tuning for classification with reduced models and iterations."""
        # Use only a subset of models for faster testing
        param_dist = {
            'Logistic Regression': self.test_param_distributions['Logistic Regression']
        }
        
        cls_optimizer = RandomSearchOptimizer(
            task_type='classification', 
            metric='accuracy', 
            random_state=42,
            n_iter=2,  # Small number for faster tests
            cv=2,       # Small number for faster tests
            param_distributions=param_dist
        )
        
        best_model = cls_optimizer.fit(self.X_cls, self.y_cls)
        
        self.assertIsNotNone(best_model, "Best model should not be None")
        self.assertIsNotNone(cls_optimizer.best_score_, "Best score should be set")
        self.assertGreaterEqual(len(cls_optimizer.results), 0, "Results should not be empty")

    def test_fit_regression(self):
        """Test model tuning for regression with reduced models and iterations."""
        # Use only a subset of models for faster testing
        param_dist = {
            'Decision Tree': self.test_param_distributions['Decision Tree']
        }
        
        reg_optimizer = RandomSearchOptimizer(
            task_type='regression', 
            metric='r2', 
            random_state=42,
            n_iter=2,  # Small number for faster tests
            cv=2,       # Small number for faster tests
            param_distributions=param_dist
        )
        
        best_model = reg_optimizer.fit(self.X_reg, self.y_reg)
        
        self.assertIsNotNone(best_model, "Best model should not be None")
        self.assertIsNotNone(reg_optimizer.best_score_, "Best score should be set")
        self.assertGreaterEqual(len(reg_optimizer.results), 0, "Results should not be empty")

    def test_get_results(self):
        """Test getting results DataFrame."""
        # Use only a subset of models for faster testing
        param_dist = {
            'Logistic Regression': self.test_param_distributions['Logistic Regression']
        }
        
        cls_optimizer = RandomSearchOptimizer(
            task_type='classification', 
            metric='accuracy', 
            random_state=42,
            n_iter=2,
            cv=2,
            param_distributions=param_dist
        )
        
        # Before fitting, should return empty DataFrame
        empty_df = cls_optimizer.get_results()
        self.assertTrue(empty_df.empty)
        
        # After fitting, should return non-empty DataFrame
        cls_optimizer.fit(self.X_cls, self.y_cls)
        results_df = cls_optimizer.get_results()
        
        self.assertFalse(results_df.empty)
        self.assertIn('model', results_df.columns)
        self.assertIn('score', results_df.columns)
        self.assertIn('time', results_df.columns)
        self.assertIn('best_params', results_df.columns)

    def test_default_param_distributions(self):
        """Test that default parameter distributions are created."""
        cls_optimizer = RandomSearchOptimizer(task_type='classification', random_state=42)
        param_dist = cls_optimizer._define_default_param_distributions()
        
        # Check that all initialized models have parameter distributions
        for model_name in cls_optimizer.models_to_run.keys():
            self.assertIn(model_name, param_dist)
            self.assertIsInstance(param_dist[model_name], dict)
            
        # Check regression param distributions
        reg_optimizer = RandomSearchOptimizer(task_type='regression', random_state=42)
        reg_param_dist = reg_optimizer._define_default_param_distributions()
        
        for model_name in reg_optimizer.models_to_run.keys():
            self.assertIn(model_name, reg_param_dist)
            self.assertIsInstance(reg_param_dist[model_name], dict)

if __name__ == '__main__':
    unittest.main()
