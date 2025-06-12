import optuna
import numpy as np
import pandas as pd
import logging
import time
import platform
import os
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Literal, cast, NamedTuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .baseline_model import BaselineModel
from .trial_suggest import model_param_suggestions_classification, regression_param_suggestions_regression

# Filter out LightGBM warnings when using GPU
warnings.filterwarnings('ignore', message='.*compiler.*')
warnings.filterwarnings('ignore', message='.*GPU.*')
warnings.filterwarnings('ignore', message='.*cuda.*')
warnings.filterwarnings('ignore', message='.*X does not have valid feature names.*')

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """
    A class for hyperparameter optimization using Optuna.

    This optimizer works with the BaselineModel class to perform hyperparameter tuning
    using the Optuna framework. It supports both classification and regression tasks
    and can optimize hyperparameters for various machine learning models.
    """

    def __init__(
        self,
        task_type: str = "classification",
        metric: Optional[str] = None,
        random_state: int = 42,
        n_trials: Union[int, Dict[str, int]] = 50,
        timeout: Optional[Union[int, Dict[str, int]]] = None,
        models: str = "trees",
        gpu: bool = False,
    ) -> None:
        """
        Initialize the OptunaOptimizer.

        Parameters
        ----------
        task_type : str, default='classification'
            Type of ML task ('classification' or 'regression')
        metric : str, optional
            Evaluation metric to optimize (same as BaselineModel metrics)
        random_state : int, default=42
            Seed for reproducibility
        n_trials : int or dict, default=50
            Number of optimization trials. Can be:
            - An integer (same number of trials for all models)
            - A dictionary mapping model names to trial counts
            - A dictionary mapping model categories to trial counts (e.g. 'boosting': 50)
        timeout : int or dict, optional
            Maximum optimization time in seconds. Can be:
            - None for no limit
            - An integer (same timeout for all models)
            - A dictionary mapping model names or categories to timeouts
        models : str, default='trees'
            Which model set to use ('all', 'trees', or 'gbm')
        gpu : bool, default=False
            Whether to enable GPU acceleration for supported models (XGBoost, LightGBM, CatBoost)
        """
        self.base_model = BaselineModel(
            task_type=task_type, metric=metric, random_state=random_state, models=models
        )
        self.task_type = task_type
        self.metric = metric
        self.random_state = random_state
        self.gpu = gpu

        # Define model categories for grouping in configuration
        self.model_categories = {
            "boosting": ["XGBoost", "LGBM", "CatBoost"],
            "gbm": ["XGBoost", "LGBM", "CatBoost"],
            "trees": ["Random Forest", "Decision Tree"],
            "linear": ["Linear Regression", "Logistic Regression"],
            "svm": ["SVC", "SVR"],
            "other": ["KNN"],
        }

        # Process n_trials configuration
        self.n_trials = self._process_config_param(n_trials, default=50)
        
        # Process timeout configuration
        self.timeout = self._process_config_param(timeout, default=None)

        # List of models to tune
        self.models_to_tune = list(self.base_model.models_to_run.keys())
        
        # Study for optimization
        self.study = None
        
        # Results storage
        self.best_params = {}
        self.best_model = None
        
        # Optimization direction (maximize or minimize)
        self.optimize_direction = (
            "maximize" if self.base_model.maximize_metric else "minimize"
        )
        
        # Parameters for hyperparameter optimization
        self.param_suggestions = (
            model_param_suggestions_classification if task_type == "classification" 
            else regression_param_suggestions_regression
        )

    def _process_config_param(
        self, param: Union[Any, Dict[str, Any]], default: Any
    ) -> Dict[str, Any]:
        """
        Process a configuration parameter that can be a single value or a dictionary.
        
        Parameters
        ----------
        param : any or dict
            The parameter value or dictionary mapping model names/categories to values
        default : any
            Default value to use if the parameter is None
            
        Returns
        -------
        dict
            Dictionary mapping model names to parameter values
        """
        result = {}
        
        # Initialize with default values for all models
        for model_name in self.base_model.models_to_run:
            result[model_name] = default
            
        # If param is not a dictionary, use the provided value for all models
        if not isinstance(param, dict):
            if param is not None:  # Only update if a non-None value was provided
                for model_name in self.base_model.models_to_run:
                    result[model_name] = param
            return result
            
        # Process dictionary param
        for key, value in param.items():
            # Check if key is a model category
            if key in self.model_categories:
                # Apply value to all models in this category
                for model_name in self.model_categories[key]:
                    if model_name in result:
                        result[model_name] = value
            # Check if key is 'all' to apply to all models
            elif key == "all":
                for model_name in result:
                    result[model_name] = value
            # Otherwise, assume key is a specific model name
            elif key in result:
                result[key] = value
                
        return result
    
    def _compute_class_weights(self, y: pd.Series) -> Dict[str, Any]:
        """
        Compute various class weights for imbalanced classification problems.
        
        Parameters
        ----------
        y : pandas.Series
            Target labels
            
        Returns
        -------
        dict
            Dictionary with different weighting strategies
        """
        if self.task_type != "classification":
            return {}
            
        class_counts = y.value_counts()
        n_samples = len(y)
        n_classes = len(class_counts)
        weights = {}
        
        # Standard balanced weighting
        balanced = {i: n_samples / (n_classes * count) for i, count in class_counts.items()}
        weights["balanced"] = balanced
        
        # Sqrt-balanced (less aggressive than balanced)
        sqrt_balanced = {i: np.sqrt(n_samples / (n_classes * count)) for i, count in class_counts.items()}
        weights["sqrt_balanced"] = sqrt_balanced
        
        # Log-balanced (even less aggressive)
        log_balanced = {i: np.log1p(n_samples / (n_classes * count)) for i, count in class_counts.items()}
        weights["log_balanced"] = log_balanced
        
        # For binary classification, add more weighting options
        if n_classes == 2:
            minority_class = class_counts.idxmin()
            majority_class = class_counts.idxmax()
            
            # Various ratios for experimentation
            weights["ratio_1_1"] = {majority_class: 1.0, minority_class: 1.0}
            weights["ratio_1_2"] = {majority_class: 1.0, minority_class: 2.0}
            weights["ratio_1_3"] = {majority_class: 1.0, minority_class: 3.0}
            weights["ratio_1_5"] = {majority_class: 1.0, minority_class: 5.0}
            
            # XGBoost-specific scale_pos_weight
            majority_count = class_counts[majority_class]
            minority_count = class_counts[minority_class]
            
            weights["scale_pos_weight"] = majority_count / minority_count
            weights["scale_pos_weight_sqrt"] = np.sqrt(majority_count / minority_count)
            
        return weights
    
    def _objective(
        self,
        trial: optuna.Trial,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        validation_size: float = 0.2,
    ) -> float:
        """
        Objective function for Optuna optimization.
        
        Parameters
        ----------
        trial : optuna.Trial
            A trial object for hyperparameter suggestion
        X : pandas.DataFrame
            Features
        y : pandas.Series
            Target
        model_name : str
            Name of the model to optimize
        validation_size : float, default=0.2
            Size of validation set
            
        Returns
        -------
        float
            Score to be optimized (maximized/minimized depending on the metric)
        """
        start_time = time.time()
        
        try:
            # Store y for potential class weight calculations
            self._current_y = y
            
            # Get suggested hyperparameters for this trial
            params = self._get_param_space(trial, model_name)
            
            # Create model instance with suggested parameters
            model_instance = self.base_model.models_to_run[model_name].__class__(**params)
            
            # Prepare data for stratified split if needed
            stratify_opt = y if self.task_type == "classification" else None
            
            # Label encode classification targets if they're not numeric
            y_processed = y
            if self.task_type == "classification" and not np.issubdtype(np.array(y).dtype, np.number):
                label_encoder = LabelEncoder()
                y_processed = label_encoder.fit_transform(y)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, 
                y_processed, 
                test_size=validation_size, 
                random_state=self.random_state,
                stratify=stratify_opt
            )
            
            # Get appropriate preprocessor type for this model and apply preprocessing
            preprocessor_key = self.base_model._get_model_type(model_name)
            X_train_proc, X_val_proc, _ = self.base_model._preprocess_for_model(
                model_name, X_train, X_val_raw=X_val
            )
            
            # Fit the model with appropriate silencing parameters for each model type
            if model_name == "XGBoost":
                # XGBoost: verbosity=0 provides complete silence
                model_instance.fit(X_train_proc, y_train)
            elif model_name == "LGBM":
                # LightGBM: verbosity=-1 ensures the quietest operation (fatal errors only)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_instance.fit(X_train_proc, y_train)
            elif model_name == "CatBoost":
                # CatBoost: logging_level='Silent' is the recommended way to silence output
                model_instance.fit(X_train_proc, y_train)
            else:
                model_instance.fit(X_train_proc, y_train)
            
            # Calculate score using BaselineModel's evaluation function
            score = self.base_model._evaluate_model_score(model_instance, X_val_proc, y_val)
            
            elapsed_time = time.time() - start_time
            
            # Log progress occasionally
            if trial.number % 10 == 0:
                logger.info(
                    f"Trial {trial.number} for {model_name} - {self.metric}: {score:.4f} (Time: {elapsed_time:.2f}s)"
                )
                
            return score
            
        except Exception as e:
            logger.error(f"Error in trial for {model_name}: {e}", exc_info=True)
            raise optuna.exceptions.TrialPruned(
                f"Trial for {model_name} failed with error: {str(e)}"
            )
        finally:
            # Clean up
            if hasattr(self, "_current_y"):
                delattr(self, "_current_y")
                
    def _get_param_space(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """
        Define the hyperparameter search space for a specific model.
        
        Parameters
        ----------
        trial : optuna.Trial
            A trial object for hyperparameter suggestion
        model_name : str
            Name of the model to optimize
            
        Returns
        -------
        dict
            Dictionary with suggested hyperparameters
        """
        # Initialize parameters
        params = {}
        
        # Get parameter suggestions for this model
        if model_name in self.param_suggestions:
            param_suggestions = self.param_suggestions[model_name]
            
            # Apply parameter suggestions using lambda functions
            for param_name, suggestion_func in param_suggestions.items():
                # Skip inapplicable parameters
                if self.task_type == "classification" and param_name in ["epsilon"]:
                    continue
                if self.task_type == "regression" and param_name in ["class_weight", "probability"]:
                    continue
                    
                # Skip dart specific parameters if not using dart booster
                if (param_name in ["sample_type", "normalize_type", "rate_drop", "skip_drop"] and
                    "booster" in params and params["booster"] != "dart"):
                    continue
                    
                # Add parameter using the suggestion function
                try:
                    params[param_name] = suggestion_func(trial)
                except Exception as e:
                    logger.warning(f"Error suggesting parameter {param_name}: {e}. Skipping.")

        # --- GPU PARAMS ---
        gpu_only_params = {
            "XGBoost": {
                "tree_method": lambda t: t.suggest_categorical("tree_method", ["gpu_hist"]),
                "predictor": lambda t: t.suggest_categorical("predictor", ["gpu_predictor"]),
            },
            "LGBM": {
                "device": lambda t: t.suggest_categorical("device", ["gpu"]),
                "gpu_platform_id": lambda t: t.suggest_int("gpu_platform_id", 0, 0),
                "gpu_device_id": lambda t: t.suggest_int("gpu_device_id", 0, 0),
            },
            "CatBoost": {
                "task_type": lambda t: t.suggest_categorical("task_type", ["GPU"]),
                "devices": lambda t: t.suggest_categorical("devices", ["0"]),
            },
        }
        if self.gpu and model_name in gpu_only_params:
            for param_name, suggestion_func in gpu_only_params[model_name].items():
                try:
                    params[param_name] = suggestion_func(trial)
                except Exception as e:
                    logger.warning(f"Error suggesting GPU parameter {param_name}: {e}. Skipping.")

        # Special handling for each model type
        if model_name == "XGBoost":
            if self.task_type == "classification":
                boosting_params = self.base_model._get_boosting_params()["xgb"]
                params["objective"] = boosting_params["objective"]
                params["eval_metric"] = boosting_params["eval_metric"]
                if hasattr(self.base_model, "n_classes_") and self.base_model.n_classes_ > 2:
                    params["num_class"] = self.base_model.n_classes_
            
            # Ensure silent operation
            params["verbosity"] = 0  # Silent mode
            params["use_label_encoder"] = False  # Avoid deprecation warnings
            
        elif model_name == "LGBM":
            if self.task_type == "classification":
                boosting_params = self.base_model._get_boosting_params()["lgbm"]
                
                # Make sure n_classes_ is set
                if not hasattr(self.base_model, "n_classes_"):
                    # If somehow we don't have n_classes_ set, default to binary
                    logger.warning("n_classes_ not set for classification task with LGBM. Defaulting to binary classification.")
                    params["objective"] = "binary"
                elif self.base_model.n_classes_ == 2:
                    params["objective"] = "binary"
                else:
                    params["objective"] = "multiclass"
                    params["num_class"] = self.base_model.n_classes_
                
                params["metric"] = boosting_params["metric"]
            else:
                # For regression
                params["objective"] = "regression"
            
            # Ensure silent operation
            params["verbose"] = -1  # Fatal errors only
            params["verbosity"] = -1  # Redundant but for extra assurance
            
        elif model_name == "CatBoost":
            if self.task_type == "classification":
                if hasattr(self.base_model, "n_classes_") and self.base_model.n_classes_ > 2:
                    params["loss_function"] = "MultiClass"
                else:
                    params["loss_function"] = "Logloss"
            else:
                params["loss_function"] = "RMSE"
                
            # Ensure silent operation
            params["verbose"] = False
            params["logging_level"] = "Silent"
        
        # Set random state for models that support it
        if model_name in ["Decision Tree", "Random Forest", "LGBM", "XGBoost"]:
            params["random_state"] = self.random_state
        elif model_name == "CatBoost":
            params["random_seed"] = self.random_state
            if "random_state" in params:
                del params["random_state"]
        
        return params
    
    def tune_model(
        self, 
        model_name: str, 
        X: pd.DataFrame, 
        y: pd.Series, 
        validation_size: float = 0.2,
        n_jobs: int = -1,
    ) -> Any:
        """
        Run hyperparameter tuning for a specific model.
        
        Parameters
        ----------
        model_name : str
            Name of the model to tune (must be one of the models in BaselineModel)
        X : pandas.DataFrame
            Features for training and validation
        y : pandas.Series
            Target variable for training and validation
        validation_size : float, default=0.2
            Proportion of data to use for validation
        n_jobs : int, default=-1
            Number of parallel jobs to run (-1 for using all available cores)
            
        Returns
        -------
        model
            The best model instance found during optimization
        """
        if model_name not in self.base_model.models_to_run:
            raise ValueError(f"Model {model_name} not found in BaselineModel. "
                           f"Available models: {list(self.base_model.models_to_run.keys())}")
        
        # If classification, determine the number of classes and set it on the base_model
        if self.task_type == "classification":
            unique_classes = y.nunique()
            self.base_model.n_classes_ = unique_classes
            logger.info(f"Setting number of classes to {unique_classes}")
            
        n_trials = self.n_trials[model_name]
        timeout = self.timeout[model_name]
        
        logger.info(f"Starting hyperparameter optimization for {model_name}")
        logger.info(f"Number of trials: {n_trials}, timeout: {timeout or 'None'}")
        
        # Create Optuna study with proper direction
        self.study = optuna.create_study(direction=self.optimize_direction)
        
        # Configure sampler and pruner
        if hasattr(optuna.samplers, "TPESampler"):
            self.study.sampler = optuna.samplers.TPESampler(seed=self.random_state)
        
        # Configure pruner for early stopping of unpromising trials
        if hasattr(optuna.pruners, "MedianPruner"):
            self.study.pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5, 
                n_warmup_steps=5, 
                interval_steps=1
            )
        
        # Create a partial function for the objective
        objective_func = lambda trial: self._objective(
            trial, X, y, model_name, validation_size
        )
        
        # Run optimization
        try:
            self.study.optimize(
                objective_func, 
                n_trials=n_trials, 
                timeout=timeout,
                n_jobs=n_jobs,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user.")
        
        # Check if any trials completed successfully
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            logger.warning("No trials completed successfully. Using default parameters.")
            # Use default parameters when no trials complete successfully
            best_params = {}
            self.best_params[model_name] = best_params
            
            # Create a model with default parameters
            best_model_class = self.base_model.models_to_run[model_name].__class__
            best_model = best_model_class()
        else:
            # Get the best parameters and create a model with them
            best_params = self.study.best_params
            self.best_params[model_name] = best_params
            
            # Create a model with the best parameters
            best_model_class = self.base_model.models_to_run[model_name].__class__
            best_model = best_model_class(**best_params)
        
        # Add additional params for certain models
        if model_name == "XGBoost" and self.task_type == "classification":
            if hasattr(self.base_model, "n_classes_") and self.base_model.n_classes_ > 2:
                best_model.set_params(num_class=self.base_model.n_classes_)
                
        # For LGBM classification, ensure the number of classes is set correctly
        if model_name == "LGBM" and self.task_type == "classification" and hasattr(self.base_model, "n_classes_"):
            if self.base_model.n_classes_ > 2:
                best_model.set_params(objective="multiclass", num_class=self.base_model.n_classes_)
        
        # Store the best model
        self.best_model = best_model
        
        # Log appropriate completion message
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            logger.warning(f"Optimization completed but no trials were successful. Using default parameters for {model_name}.")
            logger.info(f"Failed trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])}")
            logger.info(f"Pruned trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        else:
            logger.info(f"Optimization completed. Best score: {self.study.best_value}")
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Successful trials: {len(completed_trials)} of {len(self.study.trials)}")
        
        return best_model
    
    def get_tuning_results(self) -> Dict[str, Any]:
        """
        Get the results of the hyperparameter optimization.
        
        Returns
        -------
        dict
            Dictionary with tuning results
        """
        if self.study is None:
            raise ValueError("No tuning has been performed yet. Call tune_model first.")
        
        # Check if any trials completed successfully
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            results = {
                "best_params": {},
                "best_value": None,
                "best_trial": None,
                "trials": self.study.trials,
                "study": self.study,
                "completed_trials": 0,
                "failed_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]),
                "pruned_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])
            }
        else:
            results = {
                "best_params": self.study.best_params,
                "best_value": self.study.best_value,
                "best_trial": self.study.best_trial,
                "trials": self.study.trials,
                "study": self.study,
                "completed_trials": len(completed_trials),
                "failed_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]),
                "pruned_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])
            }
        
        return results
    
    def plot_optimization_history(self) -> None:
        """
        Plot the optimization history using Optuna's visualization tools.
        
        This method requires plotly to be installed.
        """
        if self.study is None:
            raise ValueError("No tuning has been performed yet. Call tune_model first.")
            
        try:
            from optuna.visualization import plot_optimization_history
            fig = plot_optimization_history(self.study)
            fig.show()
        except ImportError:
            logger.error("plotly is required to use this method. Install it with pip install plotly")
    
    def plot_param_importances(self) -> None:
        """
        Plot the parameter importances using Optuna's visualization tools.
        
        This method requires plotly to be installed.
        """
        if self.study is None:
            raise ValueError("No tuning has been performed yet. Call tune_model first.")
            
        try:
            from optuna.visualization import plot_param_importances
            fig = plot_param_importances(self.study)
            fig.show()
        except ImportError:
            logger.error("plotly is required to use this method. Install it with pip install plotly")
