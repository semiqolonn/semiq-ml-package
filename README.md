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

model = BaselineModel()
model.fit(X, y)

model.get_results() # to get the results of the baseline model
lgbm = model.get_model('LGBM')
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
