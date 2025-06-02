# Hyperparameter tuning utilities for machine learning models.
from .baseline_model import BaselineModel
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import pandas as pd
import time
import logging

logger = logging.getLogger(__name__)

class RandomSearchOptimizer(BaselineModel):
    """
    A class for hyperparameter tuning of machine learning models using RandomizedSearchCV.
    Inherits from BaselineModel to leverage its model initialization and evaluation capabilities.
    """
    
    def __init__(self, task_type="classification", metric=None, random_state=42, n_iter=10, cv=5):
        """
        Initialize the RandomSearchOptimizer.
        
        Args:
            task_type (str): 'classification' or 'regression'.
            metric (str, optional): The evaluation metric to optimize.
            random_state (int): Random seed for reproducibility.
            n_iter (int): Number of parameter settings sampled in random search.
            cv (int): Number of cross-validation folds.
        """
        super().__init__(task_type=task_type, metric=metric, random_state=random_state)
        self.n_iter = n_iter
        self.cv = cv
        self.param_distributions = self._get_default_param_distributions()
        self.tuned_models = {}
        
    def _get_default_param_distributions(self):
        """
        Define default parameter distributions for each model type.
        These are reasonable ranges for hyperparameters to search through.
        """
        if self.task_type == "classification":
            return {
                "Logistic Regression": {
                    'C': np.logspace(-3, 3, 7),
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                },
                "SVC": {
                    'C': np.logspace(-3, 3, 7),
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 0, 4))
                },
                "KNN": {
                    'n_neighbors': np.arange(3, 31, 2),
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                },
                "Decision Tree": {
                    'max_depth': [None] + list(np.arange(5, 31, 5)),
                    'min_samples_split': np.arange(2, 11),
                    'min_samples_leaf': np.arange(1, 11),
                    'criterion': ['gini', 'entropy']
                },
                "Random Forest": {
                    'n_estimators': np.arange(50, 501, 50),
                    'max_depth': [None] + list(np.arange(5, 31, 5)),
                    'min_samples_split': np.arange(2, 11),
                    'min_samples_leaf': np.arange(1, 11),
                    'max_features': ['sqrt', 'log2', None]
                },
                "LGBM": {
                    'learning_rate': np.logspace(-3, -1, 5),
                    'n_estimators': np.arange(50, 501, 50),
                    'num_leaves': np.arange(20, 151, 10),
                    'max_depth': [-1] + list(np.arange(5, 21, 5)),
                    'min_child_samples': np.arange(10, 101, 10),
                    'subsample': np.arange(0.5, 1.01, 0.1)
                },
                "XGBoost": {
                    'learning_rate': np.logspace(-3, -1, 5),
                    'n_estimators': np.arange(50, 501, 50),
                    'max_depth': np.arange(3, 11),
                    'subsample': np.arange(0.5, 1.01, 0.1),
                    'colsample_bytree': np.arange(0.5, 1.01, 0.1)
                },
                "CatBoost": {
                    'iterations': np.arange(50, 501, 50),
                    'learning_rate': np.logspace(-3, -1, 5),
                    'depth': np.arange(4, 11),
                    'l2_leaf_reg': np.logspace(-3, 3, 7),
                    'border_count': [32, 64, 128]
                }
            }
        else:  # Regression
            return {
                "Linear Regression": {},  # Linear Regression doesn't have hyperparameters to tune
                "SVR": {
                    'C': np.logspace(-3, 3, 7),
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto'] + list(np.logspace(-3, 0, 4)),
                    'epsilon': np.logspace(-3, 0, 4)
                },
                "KNN": {
                    'n_neighbors': np.arange(3, 31, 2),
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                },
                "Decision Tree": {
                    'max_depth': [None] + list(np.arange(5, 31, 5)),
                    'min_samples_split': np.arange(2, 11),
                    'min_samples_leaf': np.arange(1, 11),
                    'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']
                },
                "Random Forest": {
                    'n_estimators': np.arange(50, 501, 50),
                    'max_depth': [None] + list(np.arange(5, 31, 5)),
                    'min_samples_split': np.arange(2, 11),
                    'min_samples_leaf': np.arange(1, 11),
                    'max_features': ['sqrt', 'log2', None]
                },
                "LGBM": {
                    'learning_rate': np.logspace(-3, -1, 5),
                    'n_estimators': np.arange(50, 501, 50),
                    'num_leaves': np.arange(20, 151, 10),
                    'max_depth': [-1] + list(np.arange(5, 21, 5)),
                    'min_child_samples': np.arange(10, 101, 10),
                    'subsample': np.arange(0.5, 1.01, 0.1)
                },
                "XGBoost": {
                    'learning_rate': np.logspace(-3, -1, 5),
                    'n_estimators': np.arange(50, 501, 50),
                    'max_depth': np.arange(3, 11),
                    'subsample': np.arange(0.5, 1.01, 0.1),
                    'colsample_bytree': np.arange(0.5, 1.01, 0.1)
                },
                "CatBoost": {
                    'iterations': np.arange(50, 501, 50),
                    'learning_rate': np.logspace(-3, -1, 5),
                    'depth': np.arange(4, 11),
                    'l2_leaf_reg': np.logspace(-3, 3, 7),
                    'border_count': [32, 64, 128]
                }
            }
    
    def update_param_distributions(self, model_name, param_dict):
        """
        Update the parameter distributions for a specific model.
        
        Args:
            model_name (str): Name of the model to update parameters for.
            param_dict (dict): Dictionary of parameter distributions to use.
        """
        if model_name not in self.models_to_run:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models_to_run.keys())}")
        
        if model_name in self.param_distributions:
            self.param_distributions[model_name] = param_dict
            logger.info(f"Updated parameter distributions for {model_name}")
        else:
            raise KeyError(f"No parameter distributions found for {model_name}. Check model name.")
    
    def tune_model(self, model_name, X, y, validation_size=0.2, **fit_params):
        """
        Tune a specific model using RandomizedSearchCV.
        
        Args:
            model_name (str): Name of the model to tune.
            X (pd.DataFrame or np.array): Features for training and validation.
            y (pd.Series or np.array): Target variable for training and validation.
            validation_size (float): Proportion of data to use for holdout validation.
            **fit_params: Additional parameters to pass to the model's fit method.
            
        Returns:
            object: The best estimator from RandomizedSearchCV.
        """
        if model_name not in self.models_to_run:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models_to_run.keys())}")
        
        if model_name not in self.param_distributions or not self.param_distributions[model_name]:
            logger.warning(f"No parameter distributions defined for {model_name}. Using default model.")
            return self.models_to_run[model_name]
        
        # Handle CatBoost separately due to categorical features
        is_catboost = model_name == "CatBoost"
        original_X = X.copy() if isinstance(X, pd.DataFrame) else None
        categorical_cols = None
        
        if is_catboost and original_X is not None:
            categorical_cols = original_X.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Preprocess features for non-CatBoost models
        if not is_catboost:
            X = self._preprocess_features(X)
        
        # Split the data
        from sklearn.model_selection import train_test_split
        if self.task_type == "classification":
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_size, random_state=self.random_state, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_size, random_state=self.random_state
            )
        
        # Create scoring function based on the metric
        if self.metric in self._metric_functions:
            from sklearn.metrics import make_scorer
            scorer = make_scorer(
                self._metric_functions[self.metric], 
                greater_is_better=self.maximize_metric
            )
        else:
            # Use sklearn's built-in scoring if our metric isn't defined as a function
            scorer = self.metric
        
        # Get the base model and parameters
        base_model = self.models_to_run[model_name]
        params = self.param_distributions[model_name]
        
        # Special handling for CatBoost
        if is_catboost and categorical_cols:
            # Add cat_features parameter to the fit_params for CatBoost
            fit_params["cat_features"] = categorical_cols
        
        logger.info(f"Starting RandomizedSearchCV for {model_name} with {self.n_iter} iterations and {self.cv}-fold CV")
        start_time = time.time()
        
        try:
            # Create and run RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=params,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=scorer,
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )
            
            random_search.fit(X_train, y_train, **fit_params)
            
            # Evaluate on validation data
            best_model = random_search.best_estimator_
            
            # For CatBoost, we need to evaluate on the original data
            if is_catboost and original_X is not None:
                # Re-split the original data for validation
                _, X_val_original, _, _ = train_test_split(
                    original_X, y, test_size=validation_size, random_state=self.random_state,
                    stratify=y if self.task_type == "classification" else None
                )
                score = self._evaluate_model_score(best_model, X_val_original, y_val)
            else:
                score = self._evaluate_model_score(best_model, X_val, y_val)
            
            elapsed_time = time.time() - start_time
            
            # Store results
            self.tuned_models[model_name] = {
                "model": best_model,
                "score": score,
                "time": elapsed_time,
                "best_params": random_search.best_params_,
                "cv_results": random_search.cv_results_
            }
            
            logger.info(f"Tuning {model_name} completed in {elapsed_time:.2f}s")
            logger.info(f"Best {self.metric}: {score:.4f}")
            logger.info(f"Best parameters: {random_search.best_params_}")
            
            return best_model
            
        except Exception as e:
            logger.error(f"Error tuning {model_name}: {e}")
            self.tuned_models[model_name] = {
                "model": None,
                "score": None,
                "time": None,
                "error": str(e)
            }
            return None
    
    def tune_all_models(self, X, y, validation_size=0.2, models_to_tune=None, **fit_params):
        """
        Tune all specified models or all available models.
        
        Args:
            X (pd.DataFrame or np.array): Features for training and validation.
            y (pd.Series or np.array): Target variable for training and validation.
            validation_size (float): Proportion of data to use for holdout validation.
            models_to_tune (list, optional): List of model names to tune. If None, tune all models.
            **fit_params: Additional parameters to pass to the model's fit method.
            
        Returns:
            dict: Dictionary with tuned models and their performance.
        """
        if models_to_tune is None:
            models_to_tune = list(self.models_to_run.keys())
        else:
            # Validate that all specified models exist
            for model_name in models_to_tune:
                if model_name not in self.models_to_run:
                    raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models_to_run.keys())}")
        
        logger.info(f"Tuning {len(models_to_tune)} models: {', '.join(models_to_tune)}")
        
        best_score = -np.inf if self.maximize_metric else np.inf
        best_model_name = None
        
        for model_name in models_to_tune:
            try:
                self.tune_model(model_name, X, y, validation_size, **fit_params)
                
                # Update best model if this one is better
                current_score = self.tuned_models[model_name]["score"]
                if current_score is not None:
                    if (self.maximize_metric and current_score > best_score) or \
                       (not self.maximize_metric and current_score < best_score):
                        best_score = current_score
                        best_model_name = model_name
                        
            except Exception as e:
                logger.error(f"Error during tuning of {model_name}: {e}")
        
        if best_model_name:
            self.best_model_ = self.tuned_models[best_model_name]["model"]
            self.best_score_ = best_score
            logger.info(f"Best tuned model: {best_model_name} with {self.metric}: {best_score:.4f}")
        else:
            logger.warning("No models were successfully tuned.")
        
        return self.tuned_models
    
    def get_tuning_results(self):
        """
        Returns a Pandas DataFrame summarizing the tuning results.
        
        Returns:
            pd.DataFrame: A DataFrame with model names, scores, and best parameters.
        """
        if not self.tuned_models:
            logger.info("No tuning results available. Run tune_model() or tune_all_models() first.")
            return pd.DataFrame()
            
        results_data = []
        for name, info in self.tuned_models.items():
            if info["model"] is not None:
                results_data.append({
                    "model": name,
                    "score": info["score"],
                    "time": info["time"],
                    "best_params": info.get("best_params", {}),
                    "status": "Success"
                })
            else:
                results_data.append({
                    "model": name,
                    "score": None,
                    "time": None,
                    "best_params": {},
                    "status": "Failed",
                    "error": info.get("error", "")
                })
                
        results_df = pd.DataFrame(results_data)
        
        # Sort based on the primary metric and its direction
        if results_df.empty:
            return results_df
        
        if "score" in results_df.columns:
            if self.maximize_metric:
                results_df = results_df.sort_values(by="score", ascending=False)
            else:
                results_df = results_df.sort_values(by="score", ascending=True)
                
        return results_df.reset_index(drop=True)
    

class GridSearchOptimizer(BaselineModel):
    """
    A class for exhaustive hyperparameter tuning of machine learning models using GridSearchCV.
    Inherits from BaselineModel to leverage its model initialization and evaluation capabilities.
    """
    
    def __init__(self, task_type="classification", metric=None, random_state=42, cv=5):
        """
        Initialize the GridSearchOptimizer.
        
        Args:
            task_type (str): 'classification' or 'regression'.
            metric (str, optional): The evaluation metric to optimize.
            random_state (int): Random seed for reproducibility.
            cv (int): Number of cross-validation folds.
        """
        super().__init__(task_type=task_type, metric=metric, random_state=random_state)
        self.cv = cv
        self.param_grid = self._get_default_param_grid()
        self.tuned_models = {}
        
    def _get_default_param_grid(self):
        """
        Define default parameter grids for each model type.
        These are more focused than RandomSearchOptimizer to make grid search computationally feasible.
        """
        if self.task_type == "classification":
            return {
                "Logistic Regression": {
                    'C': [0.01, 0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                },
                "SVC": {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto', 0.1, 1.0]
                },
                "KNN": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                },
                "Decision Tree": {
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                },
                "LGBM": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200],
                    'num_leaves': [31, 63, 127],
                    'max_depth': [-1, 10, 15],
                    'subsample': [0.8, 1.0]
                },
                "XGBoost": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                },
                "CatBoost": {
                    'iterations': [100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [4, 6, 8],
                    'l2_leaf_reg': [1.0, 3.0, 9.0]
                }
            }
        else:  # Regression
            return {
                "Linear Regression": {},  # Linear Regression doesn't have hyperparameters to tune
                "SVR": {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto', 0.1, 1.0],
                    'epsilon': [0.01, 0.1, 0.2]
                },
                "KNN": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                },
                "Decision Tree": {
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['squared_error', 'absolute_error']
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                },
                "LGBM": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200],
                    'num_leaves': [31, 63, 127],
                    'max_depth': [-1, 10, 15],
                    'subsample': [0.8, 1.0]
                },
                "XGBoost": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                },
                "CatBoost": {
                    'iterations': [100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [4, 6, 8],
                    'l2_leaf_reg': [1.0, 3.0, 9.0]
                }
            }
    
    def update_param_grid(self, model_name, param_dict):
        """
        Update the parameter grid for a specific model.
        
        Args:
            model_name (str): Name of the model to update parameters for.
            param_dict (dict): Dictionary of parameter grid to use.
        """
        if model_name not in self.models_to_run:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models_to_run.keys())}")
        
        if model_name in self.param_grid:
            self.param_grid[model_name] = param_dict
            logger.info(f"Updated parameter grid for {model_name}")
        else:
            raise KeyError(f"No parameter grid found for {model_name}. Check model name.")
    
    def tune_model(self, model_name, X, y, validation_size=0.2, **fit_params):
        """
        Tune a specific model using GridSearchCV.
        
        Args:
            model_name (str): Name of the model to tune.
            X (pd.DataFrame or np.array): Features for training and validation.
            y (pd.Series or np.array): Target variable for training and validation.
            validation_size (float): Proportion of data to use for holdout validation.
            **fit_params: Additional parameters to pass to the model's fit method.
            
        Returns:
            object: The best estimator from GridSearchCV.
        """
        if model_name not in self.models_to_run:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models_to_run.keys())}")
        
        if model_name not in self.param_grid or not self.param_grid[model_name]:
            logger.warning(f"No parameter grid defined for {model_name}. Using default model.")
            return self.models_to_run[model_name]
        
        # Handle CatBoost separately due to categorical features
        is_catboost = model_name == "CatBoost"
        original_X = X.copy() if isinstance(X, pd.DataFrame) else None
        categorical_cols = None
        
        if is_catboost and original_X is not None:
            categorical_cols = original_X.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Preprocess features for non-CatBoost models
        if not is_catboost:
            X = self._preprocess_features(X)
        
        # Split the data
        from sklearn.model_selection import train_test_split
        if self.task_type == "classification":
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_size, random_state=self.random_state, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_size, random_state=self.random_state
            )
        
        # Create scoring function based on the metric
        if self.metric in self._metric_functions:
            from sklearn.metrics import make_scorer
            scorer = make_scorer(
                self._metric_functions[self.metric], 
                greater_is_better=self.maximize_metric
            )
        else:
            # Use sklearn's built-in scoring if our metric isn't defined as a function
            scorer = self.metric
        
        # Get the base model and parameters
        base_model = self.models_to_run[model_name]
        params = self.param_grid[model_name]
        
        # Special handling for CatBoost
        if is_catboost and categorical_cols:
            # Add cat_features parameter to the fit_params for CatBoost
            fit_params["cat_features"] = categorical_cols
        
        logger.info(f"Starting GridSearchCV for {model_name} with {self.cv}-fold CV")
        start_time = time.time()
        
        try:
            # Create and run GridSearchCV
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=params,
                cv=self.cv,
                scoring=scorer,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train, **fit_params)
            
            # Evaluate on validation data
            best_model = grid_search.best_estimator_
            
            # For CatBoost, we need to evaluate on the original data
            if is_catboost and original_X is not None:
                # Re-split the original data for validation
                _, X_val_original, _, _ = train_test_split(
                    original_X, y, test_size=validation_size, random_state=self.random_state,
                    stratify=y if self.task_type == "classification" else None
                )
                score = self._evaluate_model_score(best_model, X_val_original, y_val)
            else:
                score = self._evaluate_model_score(best_model, X_val, y_val)
            
            elapsed_time = time.time() - start_time
            
            # Store results
            self.tuned_models[model_name] = {
                "model": best_model,
                "score": score,
                "time": elapsed_time,
                "best_params": grid_search.best_params_,
                "cv_results": grid_search.cv_results_
            }
            
            logger.info(f"Tuning {model_name} completed in {elapsed_time:.2f}s")
            logger.info(f"Best {self.metric}: {score:.4f}")
            logger.info(f"Best parameters: {grid_search.best_params_}")
            
            return best_model
            
        except Exception as e:
            logger.error(f"Error tuning {model_name}: {e}")
            self.tuned_models[model_name] = {
                "model": None,
                "score": None,
                "time": None,
                "error": str(e)
            }
            return None
    
    def tune_all_models(self, X, y, validation_size=0.2, models_to_tune=None, **fit_params):
        """
        Tune all specified models or all available models.
        
        Args:
            X (pd.DataFrame or np.array): Features for training and validation.
            y (pd.Series or np.array): Target variable for training and validation.
            validation_size (float): Proportion of data to use for holdout validation.
            models_to_tune (list, optional): List of model names to tune. If None, tune all models.
            **fit_params: Additional parameters to pass to the model's fit method.
            
        Returns:
            dict: Dictionary with tuned models and their performance.
        """
        if models_to_tune is None:
            models_to_tune = list(self.models_to_run.keys())
        else:
            # Validate that all specified models exist
            for model_name in models_to_tune:
                if model_name not in self.models_to_run:
                    raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models_to_run.keys())}")
        
        logger.info(f"Tuning {len(models_to_tune)} models: {', '.join(models_to_tune)}")
        
        best_score = -np.inf if self.maximize_metric else np.inf
        best_model_name = None
        
        for model_name in models_to_tune:
            try:
                self.tune_model(model_name, X, y, validation_size, **fit_params)
                
                # Update best model if this one is better
                current_score = self.tuned_models[model_name]["score"]
                if current_score is not None:
                    if (self.maximize_metric and current_score > best_score) or \
                       (not self.maximize_metric and current_score < best_score):
                        best_score = current_score
                        best_model_name = model_name
                        
            except Exception as e:
                logger.error(f"Error during tuning of {model_name}: {e}")
        
        if best_model_name:
            self.best_model_ = self.tuned_models[best_model_name]["model"]
            self.best_score_ = best_score
            logger.info(f"Best tuned model: {best_model_name} with {self.metric}: {best_score:.4f}")
        else:
            logger.warning("No models were successfully tuned.")
        
        return self.tuned_models
    
    def get_tuning_results(self):
        """
        Returns a Pandas DataFrame summarizing the tuning results.
        
        Returns:
            pd.DataFrame: A DataFrame with model names, scores, and best parameters.
        """
        if not self.tuned_models:
            logger.info("No tuning results available. Run tune_model() or tune_all_models() first.")
            return pd.DataFrame()
            
        results_data = []
        for name, info in self.tuned_models.items():
            if info["model"] is not None:
                results_data.append({
                    "model": name,
                    "score": info["score"],
                    "time": info["time"],
                    "best_params": info.get("best_params", {}),
                    "status": "Success"
                })
            else:
                results_data.append({
                    "model": name,
                    "score": None,
                    "time": None,
                    "best_params": {},
                    "status": "Failed",
                    "error": info.get("error", "")
                })
                
        results_df = pd.DataFrame(results_data)
        
        # Sort based on the primary metric and its direction
        if results_df.empty:
            return results_df
        
        if "score" in results_df.columns:
            if self.maximize_metric:
                results_df = results_df.sort_values(by="score", ascending=False)
            else:
                results_df = results_df.sort_values(by="score", ascending=True)
                
        return results_df.reset_index(drop=True)