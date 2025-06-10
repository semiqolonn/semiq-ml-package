import optuna
import numpy as np
import logging
import pandas as pd
import time
from typing import Dict, Any, List, Callable, Optional, Union, Tuple, Type, cast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .baseline_model import BaselineModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """
    A class for hyperparameter optimization using Optuna.
    
    This optimizer extends BaselineModel with hyperparameter tuning capabilities 
    for various models.
    """
    
    def __init__(
        self,
        task_type: str = "classification",
        metric: Optional[str] = None,
        random_state: int = 42,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        models: str = "trees"
    ) -> None:
        """
        Initialize the OptunaOptimizer.
        
        Args:
            task_type: Type of ML task ('classification' or 'regression')
            metric: Evaluation metric to optimize (same as BaselineModel metrics)
            random_state: Seed for reproducibility
            n_trials: Number of optimization trials
            timeout: Maximum optimization time in seconds (None for no limit)
            models: Which model set to use ('all', 'trees', or 'gbm')
        """
        self.base_model: BaselineModel = BaselineModel(
            task_type=task_type,
            metric=metric,
            random_state=random_state,
            models=models
        )
        self.task_type: str = task_type
        self.metric: Optional[str] = metric
        self.random_state: int = random_state
        self.n_trials: int = n_trials
        self.timeout: Optional[int] = timeout
        self.models_to_tune: List[str] = list(self.base_model.models_to_run.keys())
        self.study: Optional[optuna.study.Study] = None
        self.best_params: Dict[str, Dict[str, Any]] = {}
        self.best_model: Optional[str] = None
        self.optimize_direction: str = "maximize" if self.base_model.maximize_metric else "minimize"
        
    def _get_param_space(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """
        Define the hyperparameter search space for a specific model.
        """
        params: Dict[str, Any] = {}
        
        if model_name == "Decision Tree":
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 32),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "random_state": self.random_state
            }
            
        elif model_name == "Random Forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 32),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "random_state": self.random_state
            }
            
        elif model_name == "LGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "num_leaves": trial.suggest_int("num_leaves", 8, 256),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": self.random_state,
                "verbosity": -1
            }
            
        elif model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": self.random_state,
                "verbosity": 0
            }
            
            if self.task_type == "classification":
                boosting_params = self.base_model._get_boosting_params()['xgb']
                params["objective"] = boosting_params['objective']
                params["eval_metric"] = boosting_params['eval_metric']
                if hasattr(self.base_model, 'n_classes_') and self.base_model.n_classes_ > 2:
                    params["num_class"] = self.base_model.n_classes_
            
        elif model_name == "CatBoost":
            params = {
                "iterations": trial.suggest_int("iterations", 50, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "random_strength": trial.suggest_float("random_strength", 1e-8, 1.0, log=True),
                "random_seed": self.random_state,
                "verbose": False
            }
            
            if self.task_type == "classification":
                boosting_params = self.base_model._get_boosting_params()['catboost']
                params["loss_function"] = boosting_params['loss_function']
            
        elif model_name == "Logistic Regression" and self.task_type == "classification":
            params = {
                "C": trial.suggest_float("C", 1e-4, 1e4, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
                "max_iter": 1000,
                "random_state": self.random_state
            }
            
        elif model_name == "SVC" and self.task_type == "classification":
            params = {
                "C": trial.suggest_float("C", 1e-4, 1e4, log=True),
                "gamma": trial.suggest_float("gamma", 1e-6, 1e1, log=True),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
                "probability": True,
                "random_state": self.random_state
            }
            
        elif model_name == "KNN":
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 40),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "p": trial.suggest_int("p", 1, 2)  # p=1: Manhattan, p=2: Euclidean
            }
            
        elif model_name == "Linear Regression" and self.task_type == "regression":
            params = {
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                "normalize": trial.suggest_categorical("normalize", [True, False])
            }
            
        elif model_name == "SVR" and self.task_type == "regression":
            params = {
                "C": trial.suggest_float("C", 1e-4, 1e4, log=True),
                "gamma": trial.suggest_float("gamma", 1e-6, 1e1, log=True),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
                "epsilon": trial.suggest_float("epsilon", 0.01, 1.0)
            }
            
        return params
    
    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, 
                  model_name: str, validation_size: float = 0.2) -> float:
        """
        Objective function for Optuna optimization.
        """
        start_time: float = time.time()
        
        try:
            params: Dict[str, Any] = self._get_param_space(trial, model_name)
            model_instance: Any = self.base_model.models_to_run[model_name].__class__(**params)
            
            stratify_opt: Optional[pd.Series] = y if self.task_type == "classification" else None
            
            y_processed: Union[pd.Series, np.ndarray] = y
            if self.task_type == "classification" and not np.issubdtype(np.array(y).dtype, np.number):
                label_encoder: LabelEncoder = LabelEncoder()
                y_processed = label_encoder.fit_transform(y)
            
            X_train: pd.DataFrame
            X_val: pd.DataFrame  
            y_train: Union[pd.Series, np.ndarray]
            y_val: Union[pd.Series, np.ndarray]
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_processed, test_size=validation_size, 
                random_state=self.random_state, stratify=stratify_opt
            )
            
            preprocessor_key = self.base_model._get_model_type(model_name)
            X_train_proc, X_val_proc, _ = self.base_model._preprocess_for_model(
                model_name, X_train, X_val_raw=X_val
            )
            
            if model_name in ["LGBM", "XGBoost"]:
                if X_val_proc is not None:
                    model_instance.fit(X_train_proc, y_train, eval_set=[(X_val_proc, y_val)], verbose=0)
                else:
                    model_instance.fit(X_train_proc, y_train, verbose=0)
            else:
                model_instance.fit(X_train_proc, y_train)
            
            score: float = self.base_model._evaluate_model_score(model_instance, X_val_proc, y_val)
            
            elapsed_time: float = time.time() - start_time
            logger.info(f"Trial {trial.number} for {model_name} - {self.metric}: {score:.4f} (Time: {elapsed_time:.2f}s)")
            
            return score
            
        except Exception as e:
            logger.error(f"Error in trial for {model_name}: {e}", exc_info=True)
            # Inform Optuna that this trial failed and should be pruned
            raise optuna.exceptions.TrialPruned(f"Trial for {model_name} failed with error: {str(e)}")
    
    def optimize(self, X: pd.DataFrame, y: pd.Series, model_name: Optional[str] = None, 
                validation_size: float = 0.2, **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Optimize hyperparameters for a specific model or all models.
        """
        models_to_optimize: List[str] = [model_name] if model_name else self.models_to_tune
        
        if self.task_type == "classification" and not hasattr(self.base_model, 'n_classes_'):
            self.base_model.n_classes_ = len(np.unique(y))
        
        self.base_model.models_to_run = self.base_model._initialize_models()

        optimization_results: Dict[str, Dict[str, Any]] = {}
        
        for model_name in models_to_optimize:
            if model_name not in self.base_model.models_to_run:
                logger.warning(f"Model {model_name} not found in available models. Skipping.")
                continue
                
            logger.info(f"Optimizing {model_name} hyperparameters...")
            
            study: optuna.study.Study = optuna.create_study(direction=self.optimize_direction, **kwargs)
            
            objective_func = lambda trial: self._objective(
                trial=trial, X=X, y=y, model_name=model_name, validation_size=validation_size
            )
            
            study.optimize(
                objective_func, 
                n_trials=self.n_trials,
                timeout=self.timeout
            )
            
            if len(study.trials) == 0 or all(t.state != optuna.trial.TrialState.COMPLETE for t in study.trials):
                logger.warning(f"No successful trials for {model_name}. Using default parameters.")
                best_params: Dict[str, Any] = {}
                best_score: float = float("-inf") if self.optimize_direction == "maximize" else float("inf")
            else:
                try:
                    best_params = study.best_params
                    best_score = study.best_value
                except (ValueError, RuntimeError) as e:
                    logger.warning(f"Error retrieving best parameters for {model_name}: {e}. Using default parameters.")
                    best_params = {}
                    best_score = float("-inf") if self.optimize_direction == "maximize" else float("inf")
            
            if model_name in ["Decision Tree", "Random Forest", "LGBM", "XGBoost", "CatBoost"]:
                best_params["random_state"] = self.random_state
                
            if model_name in ["XGBoost", "LGBM", "CatBoost"] and self.task_type == "classification":
                boosting_params = self.base_model._get_boosting_params()
                
                if model_name == "XGBoost":
                    best_params["objective"] = boosting_params['xgb']['objective']
                    best_params["eval_metric"] = boosting_params['xgb']['eval_metric']
                    if hasattr(self.base_model, 'n_classes_') and self.base_model.n_classes_ > 2:
                        best_params["num_class"] = self.base_model.n_classes_
                elif model_name == "LGBM":
                    best_params["metric"] = boosting_params['lgbm']['metric']
                    best_params["verbosity"] = -1
                elif model_name == "CatBoost":
                    best_params["loss_function"] = boosting_params['catboost']['loss_function']
                    best_params["verbose"] = False
            
            optimization_results[model_name] = {
                "best_params": best_params,
                "best_score": best_score,
                "study": study
            }
            
            logger.info(f"Best {self.metric} for {model_name}: {best_score:.4f}")
            logger.info(f"Best parameters for {model_name}: {best_params}")
        
        self.best_params = {
            model: results["best_params"] 
            for model, results in optimization_results.items()
        }
        
        if optimization_results:
            if self.optimize_direction == "maximize":
                best_model_name = max(
                    optimization_results.items(), 
                    key=lambda x: x[1]["best_score"]
                )[0]
            else:
                best_model_name = min(
                    optimization_results.items(),
                    key=lambda x: x[1]["best_score"]
                )[0]
            
            self.best_model = best_model_name
            logger.info(f"Best overall model: {best_model_name} with {self.metric}: {optimization_results[best_model_name]['best_score']:.4f}")
        
        return optimization_results
    
    def create_optimized_model(self, model_name: Optional[str] = None) -> BaselineModel:
        """
        Create a BaselineModel with optimized hyperparameters.
        """
        if not self.best_params:
            raise ValueError("No optimization results available. Call optimize() first.")
        
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No best model found. Call optimize() first.")
            model_name = self.best_model
        
        if model_name not in self.best_params:
            raise ValueError(f"Model {model_name} not found in optimization results.")
        
        best_params: Dict[str, Any] = self.best_params[model_name]
        
        optimized_model: BaselineModel = BaselineModel(
            task_type=self.task_type,
            metric=self.metric,
            random_state=self.random_state,
            models="trees"
        )
        
        model_class = self.base_model.models_to_run[model_name].__class__
        optimized_model.models_to_run = {
            model_name: model_class(**best_params)
        }
        
        return optimized_model


class TunedBaselineModel(BaselineModel):
    """
    A class that combines BaselineModel with automatic hyperparameter tuning.
    
    This class first optimizes hyperparameters using OptunaOptimizer and then
    trains models with those optimal hyperparameters.
    """
    
    def __init__(
        self,
        task_type: str = "classification",
        metric: Optional[str] = None,
        random_state: int = 42,
        models: str = "trees",
        n_trials: int = 30,
        timeout: Optional[int] = None
    ) -> None:
        """
        Initialize the TunedBaselineModel.
        """
        super().__init__(task_type, metric, random_state, models)
        self.optimizer: OptunaOptimizer = OptunaOptimizer(
            task_type=task_type,
            metric=metric,
            random_state=random_state,
            n_trials=n_trials, 
            timeout=timeout,
            models=models
        )
        self.best_params: Dict[str, Dict[str, Any]] = {}
        self.optimization_results: Optional[Dict[str, Any]] = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, validation_size: float = 0.2, 
            optimize_first: bool = True, **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Fit models with optional hyperparameter optimization.
        """
        original_models = self.models_to_run.copy()
        
        if optimize_first:
            logger.info("Starting hyperparameter optimization...")
            
            self.optimization_results = self.optimizer.optimize(
                X=X, y=y, validation_size=validation_size
            )
            
            self.best_params = self.optimizer.best_params
            
            for model_name, params in self.best_params.items():
                if model_name in self.models_to_run:
                    model_class = self.models_to_run[model_name].__class__
                    self.models_to_run[model_name] = model_class(**params)
        
        results = super().fit(X, y, validation_size, self.random_state, **kwargs)
        
        return results
    
    def optimize_single_model(self, X: pd.DataFrame, y: pd.Series, model_name: str, 
                              validation_size: float = 0.2, **kwargs) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific model.
        """
        if model_name not in self.models_to_run:
            raise ValueError(f"Model {model_name} not found in available models.")
        
        optimization_result: Dict[str, Any] = self.optimizer.optimize(
            X=X, y=y, model_name=model_name, validation_size=validation_size, **kwargs
        )
        
        self.best_params[model_name] = optimization_result[model_name]["best_params"]
        
        model_class = self.models_to_run[model_name].__class__
        self.models_to_run[model_name] = model_class(**self.best_params[model_name])
        
        return optimization_result[model_name]
