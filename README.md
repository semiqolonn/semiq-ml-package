# ML Helper

A collection of reusable machine learning pipeline helpers designed to streamline ML workflows.

## Description

ML Helper is a Python package that provides helper functions and classes to simplify common machine learning tasks, including baseline model training and hyperparameter tuning. It supports popular ML frameworks like LightGBM, XGBoost, and CatBoost.

## Installation

Install the package directly from GitHub:

```bash
pip install git+https://github.com/yourusername/ml-helper.git
```

Or install from source:

```bash
git clone https://github.com/yourusername/ml-helper.git
cd ml-helper
pip install -e .
```

## Features

- **Baseline Models**: Quickly train baseline models with sensible defaults
- **Hyperparameter Tuning**: Optimize model performance using random search
- **Integration**: Seamless integration with scikit-learn, LightGBM, XGBoost, and CatBoost

## Usage

### Basic Example

```python
from ml_helper import BaselineModel
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('your_data.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a baseline model
model = BaselineModel(algorithm='lightgbm')
model.fit(X_train, y_train)

# Evaluate model
score = model.score(X_test, y_test)
print(f"Model accuracy: {score}")

# Make predictions
predictions = model.predict(X_test)
```

### Hyperparameter Tuning

```python
from ml_helper import RandomSearchOptimizer
import numpy as np

# Define parameter grid
param_grid = {
    'n_estimators': np.arange(50, 200, 25),
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2]
}

# Initialize optimizer
optimizer = RandomSearchOptimizer(
    algorithm='xgboost',
    param_grid=param_grid,
    n_iter=20,
    cv=5
)

# Run optimization
best_model = optimizer.optimize(X_train, y_train)
print(f"Best parameters: {optimizer.best_params_}")
print(f"Best score: {optimizer.best_score_}")
```

## Requirements

- Python >=3.8
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- lightgbm
- xgboost
- catboost

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
