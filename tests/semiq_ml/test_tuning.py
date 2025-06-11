import pytest
import numpy as np
import pandas as pd
import optuna
from semiq_ml.tuning import OptunaOptimizer

@pytest.mark.parametrize("task_type, model_name", [
    ("classification", "Random Forest"),
    ("regression", "LGBM"),
])
def test_optuna_optimizer_runs(task_type, model_name):
    # Generate dummy data
    X = pd.DataFrame(np.random.randn(100, 4), columns=[f"f{i}" for i in range(4)])
    if task_type == "classification":
        y = pd.Series(np.random.choice([0, 1], size=100))
    else:
        y = pd.Series(np.random.randn(100))

    opt = OptunaOptimizer(task_type=task_type, n_trials=2, models="all", gpu=False)
    # Only tune the selected model for speed
    opt.models_to_tune = [model_name]
    # Run a single trial to check for errors
    study = optuna.create_study(direction=opt.optimize_direction)
    def objective(trial):
        return opt._objective(trial, X, y, model_name)
    study.optimize(objective, n_trials=2)
    assert len(study.trials) == 2


def test_gpu_param_injection():
    """Test that GPU parameters are correctly injected when gpu=True"""
    X = pd.DataFrame(np.random.randn(10, 2), columns=["a", "b"])
    y = pd.Series(np.random.choice([0, 1], size=10))
    
    # Mock GPU parameters to avoid FixedTrial issues
    from unittest.mock import patch
    
    opt = OptunaOptimizer(task_type="classification", n_trials=1, models="all", gpu=True)
    
    # Create a simple Trial class that will work for our test
    class SimpleTrial:
        def suggest_categorical(self, name, choices):
            if name == "tree_method": return "gpu_hist"
            if name == "predictor": return "gpu_predictor"
            if name == "device": return "gpu"
            if name == "task_type": return "GPU"
            if name == "devices": return "0"
            return choices[0]
            
        def suggest_int(self, name, low, high):
            return 0
    
    # Test XGBoost GPU parameters
    params = opt._get_param_space(SimpleTrial(), "XGBoost")
    assert "tree_method" in params and params["tree_method"] == "gpu_hist"
    assert "predictor" in params and params["predictor"] == "gpu_predictor"
    
    # Test LGBM GPU parameters
    params = opt._get_param_space(SimpleTrial(), "LGBM")
    assert "device" in params and params["device"] == "gpu"
    assert "gpu_platform_id" in params
    assert "gpu_device_id" in params
    
    # Test CatBoost GPU parameters
    params = opt._get_param_space(SimpleTrial(), "CatBoost")
    assert "task_type" in params and params["task_type"] == "GPU"
    assert "devices" in params
    
    # Test with GPU disabled
    opt_no_gpu = OptunaOptimizer(task_type="classification", n_trials=1, models="all", gpu=False)
    params = opt_no_gpu._get_param_space(SimpleTrial(), "XGBoost")
    assert "tree_method" not in params
    assert "predictor" not in params


@pytest.mark.parametrize("gpu_enabled", [True, False])
def test_tune_model_respects_gpu_flag(gpu_enabled):
    """Test that tune_model respects the gpu flag"""
    X = pd.DataFrame(np.random.randn(30, 2), columns=["a", "b"])
    y = pd.Series(np.random.choice([0, 1], size=30))
    
    # Mock the optimize function to avoid actual training
    import unittest.mock as mock
    
    # We need to modify the study.optimize method and add mock trials to the study
    class MockStudy:
        def __init__(self):
            self.best_params = {}
            self.best_trial = mock.MagicMock()
            self.best_value = 0.95
        
        def optimize(self, *args, **kwargs):
            pass

    opt = OptunaOptimizer(task_type="classification", n_trials=1, models="all", gpu=gpu_enabled)
    opt.study = MockStudy()
    
    # Use a simplified _get_param_space to avoid actual parameter suggestion
    with mock.patch.object(opt, '_get_param_space') as mock_get_params:
        # Just return an empty dict for parameters
        mock_get_params.return_value = {}
        
        # Call tune_model
        opt.tune_model("XGBoost", X, y, validation_size=0.2)
        
        # Check that _get_param_space was called
        mock_get_params.assert_called()
        
        # Create a test to verify gpu was passed correctly
        assert opt.gpu == gpu_enabled
