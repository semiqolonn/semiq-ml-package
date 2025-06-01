import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from ml_helper.baseline_model import BaselineModel
from ml_helper.hyperparameter_tuning import RandomSearchOptimizer

class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test data for classification task"""
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
        
        # Define a simplified param distribution for faster tests
        self.test_param_distributions = {
            'Logistic Regression': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear']
            }
        }

    def test_baseline_to_optimizer_workflow(self):
        """Test a workflow from baseline models to hyperparameter tuning."""
        
        # 1. Run baseline models first
        baseline = BaselineModel(task_type='classification', metric='accuracy', random_state=42)
        baseline.fit(self.X_cls, self.y_cls, validation_size=0.2)
        
        # Check baseline results
        baseline_results = baseline.get_results()
        self.assertFalse(baseline_results.empty)
        
        # 2. Run hyperparameter tuning on the best model type from baseline
        best_model_type = baseline_results.iloc[0]['model']
        
        # Create a param distribution with only the best model type
        limited_param_dist = {
            best_model_type: self.test_param_distributions.get(
                best_model_type, 
                {'max_depth': [3, 5]} if 'Tree' in best_model_type else {'C': [0.1, 1.0]}
            )
        }
        
        # Run the optimizer with limited models and iterations for testing
        optimizer = RandomSearchOptimizer(
            task_type='classification',
            metric='accuracy',
            random_state=42,
            n_iter=2,
            cv=2,
            param_distributions=limited_param_dist
        )
        
        # Fit the optimizer
        best_tuned_model = optimizer.fit(self.X_cls, self.y_cls)
        
        # Verify we got results
        self.assertIsNotNone(best_tuned_model)
        optimizer_results = optimizer.get_results()
        self.assertFalse(optimizer_results.empty)
        
        # 3. Compare baseline vs tuned performance
        baseline_best_score = baseline_results.iloc[0]['score']
        tuned_best_score = optimizer_results.iloc[0]['score']
        
        # Note: This is not a strict test as tuning might not always improve performance
        # especially with such limited iterations, but useful as a sanity check
        print(f"Baseline best score: {baseline_best_score}")
        print(f"Tuned best score: {tuned_best_score}")

if __name__ == '__main__':
    unittest.main()
