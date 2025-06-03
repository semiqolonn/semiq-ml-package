import importlib
import semiq_ml

def test_version():
    """Test that the package has a version."""
    assert hasattr(semiq_ml, '__version__')
    assert isinstance(semiq_ml.__version__, str)

def test_imports():
    """Test that core classes are importable from package."""
    # Check direct imports from the package
    assert hasattr(semiq_ml, 'BaselineModel')
    assert hasattr(semiq_ml, 'RandomSearchOptimizer')
    assert hasattr(semiq_ml, 'GridSearchOptimizer')
    
    # Test the actual imports
    from semiq_ml import BaselineModel, RandomSearchOptimizer, GridSearchOptimizer
    assert BaselineModel.__name__ == 'BaselineModel'
    assert RandomSearchOptimizer.__name__ == 'RandomSearchOptimizer'
    assert GridSearchOptimizer.__name__ == 'GridSearchOptimizer'
