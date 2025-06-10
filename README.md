# semiq-ml - Machine Learning Workflow Simplifier

Welcome to the semiq-ml documentation. This package provides helper functions and classes to simplify common machine learning workflows, including baseline model training, evaluation, and hyperparameter tuning.

## Overview

semiq-ml is designed to:

- Quickly compare multiple machine learning models on your dataset
- Automate hyperparameter tuning with Optuna
- Provide consistent preprocessing and evaluation
- Support both classification and regression tasks
- Handle categorical features correctly, especially for tree-based models
- Offer flexible model selection with 'all', 'trees', or 'gbm' options

## Key Components

### BaselineModel

The `BaselineModel` class automates the training and evaluation of multiple ML models, providing:

- Automatic handling of preprocessing (scaling, encoding, imputation)
- Performance comparison across standard algorithms
- Support for common evaluation metrics
- Special handling for boosting libraries (LightGBM, XGBoost, CatBoost)
- Visualization of ROC curves and precision-recall curves
- Flexible model selection with 'all', 'trees', or 'gbm' options

### OptunaOptimizer

The `OptunaOptimizer` class enhances the BaselineModel by adding:

- Efficient hyperparameter tuning with Optuna
- Smart parameter space sampling for all supported models
- Detailed tuning results and best parameter reporting
- Visualization of optimization history and parameter importance
- Flexible control over trials and cross-validation

## Getting Started

Please refer to these guides to get started with semiq-ml:

- [Installation Guide](https://github.com/semiqolonn/semiq-ml/wiki/Installation) - Setup instructions and requirements
- [Basic Usage Examples](https://github.com/semiqolonn/semiq-ml/wiki/BasicUsage) - Simple examples to get you started
- [API Reference](https://github.com/semiqolonn/semiq-ml/wiki/APIReference) - Complete documentation of all classes and methods

## Example Usage

The following example demonstrates a typical semiq-ml workflow:

```python
# Import required libraries
from semiq_ml import BaselineModel
from semiq_ml.tuning import OptunaOptimizer
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load your dataset
data = pd.read_csv('your_data.csv')
X = data.drop('target', axis=1)  # Features
y = data['target']               # Target variable

# 2. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train and evaluate baseline models
baseline = BaselineModel(
    task_type="classification",  # Use "regression" for regression tasks
    metric="f1_weighted",        # Choose an appropriate evaluation metric
    models="trees"               # Only use tree-based models (options: 'all', 'trees', 'gbm')
)
baseline.fit(X_train, y_train)
results = baseline.get_results()
print(results)

# 4. Tune the best performing model with OptunaOptimizer
best_model_name = results.iloc[0]['model']
tuner = OptunaOptimizer(
    task_type="classification", 
    metric="f1_weighted",
    n_trials=20                  # Number of parameter combinations to try
)
tuned_model = tuner.tune_model(best_model_name, X_train, y_train)
tuning_results = tuner.get_tuning_results()
print(tuning_results)
```

For more examples and advanced usage, see the [Basic Usage Examples](https://github.com/semiqolonn/semiq-ml/wiki/BasicUsage) guide.

## Support

If you encounter issues or have questions about semiq-ml:

- **Bug Reports**: Please [open an issue](https://github.com/semiqolonn/semiq-ml/issues) with a detailed description of the problem, steps to reproduce it, and your environment details.
- **Feature Requests**: Submit your ideas through the [issue tracker](https://github.com/semiqolonn/semiq-ml/issues) using the "Feature Request" template.
- **Questions**: For usage questions, reach out via [GitHub Discussions](https://github.com/semiqolonn/semiq-ml/discussions)

## Contributing

We welcome contributions to semiq-ml! Here's how you can help:

1. **Code Contributions**: Fork the repository, create a feature branch, and submit a pull request.
2. **Documentation**: Help improve or translate documentation.
3. **Bug Reports**: Report bugs or suggest features via the issue tracker.

Please review our [Contributing Guidelines](https://github.com/semiqolonn/ml-helper/wiki/Contributing) for more details on code style, testing requirements, and the pull request process.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.