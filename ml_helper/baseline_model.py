import pandas as pd
import numpy as np
import time
import logging


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Import specific boosting libraries
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# Import all necessary metrics
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    mean_squared_error, r2_score, mean_absolute_error, make_scorer
)


# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaselineModel:
    """
    A class to automate the training and basic evaluation of multiple baseline
    machine learning models for either classification or regression tasks.
    """
    def __init__(self, task_type='classification', metric=None, random_state=42):
        """
        Initializes the BaselineModel instance.

        Args:
            task_type (str): 'classification' or 'regression'.
            metric (str, optional): The evaluation metric to optimize for.
                                    For classification: 'accuracy', 'f1_weighted', 'roc_auc', 'precision_weighted', 'recall_weighted'.
                                    For regression: 'neg_root_mean_squared_error', 'r2', 'neg_mean_absolute_error'.
                                    If None, defaults to 'accuracy' for classification and 'neg_root_mean_squared_error' for regression.
            random_state (int): Seed for reproducibility.
        """
        if task_type not in ['classification', 'regression']:
            raise ValueError("task_type must be 'classification' or 'regression'.")
        self.task_type = task_type
        self.random_state = random_state
        self.results = {}
        self.best_model_ = None
        self.best_score_ = -np.inf # Initialize for maximizing score

        self._set_metric_and_direction(metric)
        self.models_to_run = self._initialize_models() # Now holds a dict of {name: model_instance}

        # Mapping of metric names to sklearn metric functions
        self._metric_functions = {
            'accuracy': accuracy_score,
            'f1_weighted': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score,
            'auc': roc_auc_score,  # Alias for compatibility
            'precision_weighted': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'neg_root_mean_squared_error': lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score,
            'neg_mean_absolute_error': lambda y_true, y_pred: -mean_absolute_error(y_true, y_pred),
            'log_loss': lambda y_true, y_pred: -np.mean(np.log(y_pred + 1e-15)) if len(np.unique(y_true)) == 2 else None
        }


    def _set_metric_and_direction(self, metric):
        """Sets the optimization metric and its direction (maximize/minimize)."""
        self.maximize_metric = True # Default to maximizing (e.g., accuracy, R2, ROC AUC)

        if self.task_type == 'classification':
            self.metric = metric if metric else 'accuracy'
            if self.metric not in ['accuracy', 'f1_weighted', 'roc_auc', 'precision_weighted', 'recall_weighted', 'auc']:
                raise ValueError(f"Invalid metric '{self.metric}' for classification. Choose from 'accuracy', 'f1_weighted', 'roc_auc', 'precision_weighted', 'recall_weighted'.")
        else: # regression
            self.metric = metric if metric else 'neg_root_mean_squared_error'
            if self.metric not in ['neg_root_mean_squared_error', 'r2', 'neg_mean_absolute_error']:
                raise ValueError(f"Invalid metric '{self.metric}' for regression. Choose from 'neg_root_mean_squared_error', 'r2', 'neg_mean_absolute_error'.")
            
            # Metrics like RMSE, MAE are minimized, so 'neg_' prefix makes them maximize
            if 'neg_' in self.metric:
                self.maximize_metric = True
            else: # r2 is maximized
                self.maximize_metric = True


    def _preprocess_features(self, X):
        """
        Preprocesses the features:
        - Encodes categorical variables (One-Hot)
        - Scales numerical features (optional)
        
        Args:
            X (pd.DataFrame): Raw input features.

        Returns:
            np.array: Processed features suitable for ML models.
        """
        if not isinstance(X, pd.DataFrame):
            return X  # Assume already preprocessed

        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        transformers = []

        if categorical_cols:
            transformers.append(
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            )

        if numeric_cols:
            transformers.append(
                ('num', StandardScaler(), numeric_cols)
            )

        self._preprocessor = ColumnTransformer(transformers)
        return self._preprocessor.fit_transform(X)


    def _initialize_models(self):
        """
        Initializes a dictionary of model instances with basic defaults.
        These models are unfitted and can be used directly or as base estimators
        for hyperparameter tuning.
        """
        if self.task_type == 'classification':
            return {
                'Logistic Regression': LogisticRegression(random_state=self.random_state, solver='liblinear', max_iter=1000),
                # SVC probability is expensive; enable only if ROC AUC is desired metric
                'SVC': SVC(random_state=self.random_state, probability=(self.metric in ('roc_auc', 'auc'))),
                'KNN': KNeighborsClassifier(),
                'Decision Tree': DecisionTreeClassifier(random_state=self.random_state),
                'Random Forest': RandomForestClassifier(random_state=self.random_state), # n_estimators handled by tuning
                'LGBM': LGBMClassifier(random_state=self.random_state, verbosity=-1), # n_estimators, lr handled by tuning
                'XGBoost': XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='logloss'),
                'CatBoost': CatBoostClassifier(random_state=self.random_state, silent=True) # iterations, lr handled by tuning
            }
        else: # Regression
            return {
                'Linear Regression': LinearRegression(),
                'SVR': SVR(),
                'KNN': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(random_state=self.random_state),
                'Random Forest': RandomForestRegressor(random_state=self.random_state), # n_estimators handled by tuning
                'LGBM': LGBMRegressor(random_state=self.random_state), # n_estimators, lr handled by tuning
                'XGBoost': XGBRegressor(random_state=self.random_state, use_label_encoder=False, eval_metric='rmse'),
                'CatBoost': CatBoostRegressor(random_state=self.random_state, silent=True) # iterations, lr handled by tuning
            }
    
    def _evaluate_model_score(self, model, X_val, y_val):
        """Calculates the score for a given model and metric."""
        if self.task_type == 'classification':
            if self.metric in ('roc_auc', 'auc'):
                has_proba = (
                    hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba'))
                ) or (
                    hasattr(model, 'probability') and getattr(model, 'probability') is True
                )

                if has_proba:
                    # Use np.unique to support Series or ndarray
                    import numpy as np
                    is_multiclass = len(np.unique(y_val)) > 2

                    if is_multiclass:
                        logger.warning("ROC AUC for multi-class is complex. Using f1_weighted instead.")
                        y_pred = model.predict(X_val)
                        return self._metric_functions['f1_weighted'](y_val, y_pred)
                    else:
                        y_pred_proba = model.predict_proba(X_val)[:, 1]
                        return self._metric_functions[self.metric](y_val, y_pred_proba)
                else:
                    logger.warning(f"Model {model.__class__.__name__} does not support predict_proba or `probability=True` not set. Falling back to accuracy for ROC AUC metric.")
                    y_pred = model.predict(X_val)
                    return accuracy_score(y_val, y_pred)
            else:
                y_pred = model.predict(X_val)
                return self._metric_functions[self.metric](y_val, y_pred)
        else:  # Regression
            y_pred = model.predict(X_val)
            return self._metric_functions[self.metric](y_val, y_pred)



    def fit(self, X, y, validation_size=0.2, **fit_params):
        """
        Trains all initialized models and evaluates them on a validation set.

        Args:
            X (pd.DataFrame or np.array): Features for training and validation.
            y (pd.Series or np.array): Target variable for training and validation.
            validation_size (float): Proportion of the dataset to include in the validation split.
            **fit_params: Additional keyword arguments to pass to the model's .fit() method.
                          This is especially useful for boosting models (e.g., `early_stopping_rounds`).
                          `eval_set` will be automatically passed for boosting models.

        Returns:
            object: The best trained model instance based on the chosen metric.
        """
        X = self._preprocess_features(X)  # Preprocess features
        
        if self.task_type == 'classification':
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_size, random_state=self.random_state, stratify=y)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_size, random_state=self.random_state)
        
        self.results = {}
        self.best_model_ = None
        # Reset best score for fresh run, considering maximize_metric
        self.best_score_ = -np.inf if self.maximize_metric else np.inf

        logger.info(f"Starting BaselineModel training for {self.task_type} with metric: {self.metric}")
        logger.info(f"Validation set size: {validation_size * 100:.0f}%")

        for name, model in self.models_to_run.items():
            start_time = time.time()
            try:
                current_fit_params = fit_params.copy()

                # Special handling for boosting models' eval_set and verbosity
                if isinstance(model, (LGBMClassifier, LGBMRegressor, XGBClassifier, XGBRegressor, CatBoostClassifier, CatBoostRegressor)):
                    if 'eval_set' not in current_fit_params:
                        current_fit_params['eval_set'] = [(X_val, y_val)]
                    
                    # For CatBoost, if categorical features are in X (DataFrame), pass them
                    if isinstance(model, (CatBoostClassifier, CatBoostRegressor)) and isinstance(X_train, pd.DataFrame):
                        # Attempt to auto-detect categorical features if not explicitly provided
                        categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
                        if categorical_cols:
                            current_fit_params['cat_features'] = categorical_cols
                            logger.info(f"CatBoost: Auto-detected categorical features: {categorical_cols}")

                    # XGBoost specific: silent mode via verbosity
                    if isinstance(model, (XGBClassifier, XGBRegressor)) and 'verbose' not in current_fit_params:
                        current_fit_params['verbose'] = False # Suppress boosting rounds output

                model.fit(X_train, y_train, **current_fit_params)
                
                score = self._evaluate_model_score(model, X_val, y_val)
                elapsed_time = time.time() - start_time
                
                logger.info(f"  {name} {self.metric}: {score:.4f} (Training Time: {elapsed_time:.2f}s)")
                
                self.results[name] = {
                    'model': model,
                    'score': score,
                    'time': elapsed_time
                }

                # Update best model based on metric direction
                if (self.maximize_metric and score > self.best_score_) or \
                   (not self.maximize_metric and score < self.best_score_):
                    self.best_score_ = score
                    self.best_model_ = model
                    logger.info(f"  --> NEW BEST model: {name} with {self.metric}: {score:.4f}")

            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                self.results[name] = {
                    'model': None,
                    'score': None,
                    'time': None,
                    'error': str(e)
                }
        
        if not self.best_model_:
            logger.error("No models were successfully trained or evaluated to determine a best model.")
            return None

        logger.info(f"BaselineModel run complete. Best model: {self.best_model_.__class__.__name__} with {self.best_score_:.4f}")
        return self.best_model_
    
    def get_model(self, model_name):
        """
        Retrieves a trained model by its name.

        Args:
            model_name (str): The name of the model (e.g., 'Logistic Regression', 'LGBM').

        Returns:
            object: The trained model instance.
        """
        if model_name in self.results and self.results[model_name]['model'] is not None:
            return self.results[model_name]['model']
        else:
            raise ValueError(f"Model '{model_name}' not found or failed to train. Available models: {list(self.results.keys())}")
    
    def evaluate_all(self, X, y):
        """
        Evaluates all trained models on a given dataset (X, y) and returns
        a dictionary of detailed metric scores for each model.

        Args:
            X (pd.DataFrame or np.array): Features for evaluation.
            y (pd.Series or np.array): Target variable for evaluation.

        Returns:
            dict: A dictionary where keys are model names and values are
                  dictionaries of detailed metrics (e.g., {'accuracy': 0.85, 'f1': 0.82}).
        """
        predictions_metrics = {}
        if not self.results:
            logger.warning("No models have been trained yet. Run .fit() first.")
            return predictions_metrics

        for name, model_info in self.results.items():
            model = model_info['model']
            if model is None:
                logger.warning(f"Skipping evaluation for {name} as it failed to train.")
                continue

            try:
                if self.task_type == 'classification':
                    if hasattr(self, '_preprocessor'):
                        X = self._preprocessor.transform(X)
                    preds = model.predict(X)

                    metrics = {'accuracy': accuracy_score(y, preds)}

                    # Add other classification metrics if applicable
                    if hasattr(model, 'predict_proba') and (hasattr(model, 'probability') and model.probability): # Check for SVC probability
                        y_pred_proba = model.predict_proba(X)
                        metrics['f1_weighted'] = f1_score(y, preds, average='weighted', zero_division=0)
                        # ROC AUC for binary classification only
                        if y.nunique() <= 2:
                            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba[:, 1])
                        else:
                            # For multiclass, roc_auc_score needs specific average
                            metrics['roc_auc_ovr_weighted'] = roc_auc_score(y, y_pred_proba, multi_class='ovr', average='weighted')
                        metrics['precision_weighted'] = precision_score(y, preds, average='weighted', zero_division=0)
                        metrics['recall_weighted'] = recall_score(y, preds, average='weighted', zero_division=0)
                    else: # Fallback metrics if proba not available
                         metrics['f1_weighted'] = f1_score(y, preds, average='weighted', zero_division=0)
                         metrics['precision_weighted'] = precision_score(y, preds, average='weighted', zero_division=0)
                         metrics['recall_weighted'] = recall_score(y, preds, average='weighted', zero_division=0)
                else: # Regression
                    if hasattr(self, '_preprocessor'):
                        X = self._preprocessor.transform(X)
                    preds = model.predict(X)

                    metrics = {
                        'mse': mean_squared_error(y, preds),
                        'mae': mean_absolute_error(y, preds),
                        'r2': r2_score(y, preds)
                    }
                predictions_metrics[name] = metrics
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                predictions_metrics[name] = {'error': str(e)}
        
        return predictions_metrics
    
    def get_results(self):
        """
        Returns a Pandas DataFrame summarizing the results of the BaselineModel run.

        Returns:
            pd.DataFrame: A DataFrame with model names, primary score, and training time.
        """
        if not self.results:
            logger.info("No results available. Run .fit() first.")
            return pd.DataFrame()

        results_data = []
        for name, info in self.results.items():
            results_data.append({
                'model': name,
                'score': info['score'],
                'time': info['time'],
                'status': 'Success' if info['model'] is not None else 'Failed',
                'error': info.get('error', '')
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Sort based on the primary metric and its direction
        if self.maximize_metric:
            results_df = results_df.sort_values(by='score', ascending=False)
        else:
            results_df = results_df.sort_values(by='score', ascending=True)

        return results_df.reset_index(drop=True)