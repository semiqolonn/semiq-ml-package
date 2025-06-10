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
        n_trials: Union[int, Dict[str, int]] = 50,
        timeout: Optional[Union[int, Dict[str, int]]] = None,
        models: str = "trees"
    ) -> None:
        """
        Initialize the OptunaOptimizer.
        
        Args:
            task_type: Type of ML task ('classification' or 'regression')
            metric: Evaluation metric to optimize (same as BaselineModel metrics)
            random_state: Seed for reproducibility
            n_trials: Number of optimization trials. Can be:
                      - An integer (same number of trials for all models)
                      - A dictionary mapping model names to trial counts
                      - A dictionary mapping model categories to trial counts (e.g. 'boosting': 50)
            timeout: Maximum optimization time in seconds. Can be:
                    - None for no limit
                    - An integer (same timeout for all models)
                    - A dictionary mapping model names or categories to timeouts
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
        
        self.model_categories = {
            'boosting': ['XGBoost', 'LGBM', 'CatBoost'],
            'gbm': ['XGBoost', 'LGBM', 'CatBoost'],
            'trees': ['Random Forest', 'Decision Tree'],
            'linear': ['Linear Regression', 'Logistic Regression'],
            'svm': ['SVC', 'SVR'],
            'other': ['KNN']
        }
        
        self.n_trials: Dict[str, int] = self._process_config_param(n_trials, default=50)
        
        self.timeout: Dict[str, Optional[int]] = self._process_config_param(timeout, default=None)
        
        self.models_to_tune: List[str] = list(self.base_model.models_to_run.keys())
        self.study: Optional[optuna.study.Study] = None
        self.best_params: Dict[str, Dict[str, Any]] = {}
        self.best_model: Optional[str] = None
        self.optimize_direction: str = "maximize" if self.base_model.maximize_metric else "minimize"
        
        self.param_config = self._get_default_param_configs()
    
    def _process_config_param(self, param: Union[Any, Dict[str, Any]], default: Any) -> Dict[str, Any]:
        """
        Process a configuration parameter that can be either a single value or a dictionary.
        
        Args:
            param: The parameter (n_trials or timeout)
            default: Default value to use if None
            
        Returns:
            Dictionary mapping each model name to its parameter value
        """
        all_models = set()
        for models in self.model_categories.values():
            all_models.update(models)
        
        result = {model: default for model in all_models}
        
        if param is None:
            return result
            
        if not isinstance(param, dict):
            for model in result:
                result[model] = param
            return result
        
        if 'all' in param:
            for model in result:
                result[model] = param['all']
    
        for key, value in param.items():
            if key in self.model_categories and key != 'all':
                for model in self.model_categories[key]:
                    result[model] = value
    
        for key, value in param.items():
            if key in result:
                result[key] = value
    
        return result

    def _get_default_param_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Define the default hyperparameter configurations for various models.
        
        Returns:
            Dictionary containing hyperparameter configurations for each model type.
        """
        return {
            "xgboost": {
                "booster": ["gbtree", "gblinear", "dart"],
                "n_estimators": "int",
                "learning_rate": "float",
                "max_depth": "int",
                "min_child_weight": "float",
                "gamma": "float",
                "subsample": "float",
                "colsample_bytree": "float",
                "colsample_bylevel": "float",
                "colsample_bynode": "float",
                "lambda": "float",
                "alpha": "float",
                "scale_pos_weight": "auto_scale",
                "objective": "str",
                "eval_metric": "str",
                "tree_method": ["auto", "exact", "approx", "hist", "gpu_hist"],
                "predictor": "str",
                "verbosity": "int",
                "seed": "int",
                "random_state": "int",
                "grow_policy": ["depthwise", "lossguide"],
                "max_leaves": "int",
                "max_bin": "int"
            },
            "lightgbm": {
                "boosting_type": ["gbdt", "dart", "goss", "rf"],
                "num_leaves": "int",
                "max_depth": "int",
                "learning_rate": "float",
                "n_estimators": "int",
                "subsample_for_bin": "int",
                "objective": "str",
                "class_weight": "auto_weight",
                "min_split_gain": "float",
                "min_child_weight": "float",
                "min_child_samples": "int",
                "subsample": "float",
                "subsample_freq": "int",
                "colsample_bytree": "float",
                "reg_alpha": "float",
                "reg_lambda": "float",
                "random_state": "int",
                "n_jobs": "int",
                "importance_type": ["split", "gain"],
                "metric": "str",
                "early_stopping_round": "int",
                "cat_smooth": "float",
                "max_bin": "int"
            },
            "catboost": {
                "iterations": "int",
                "learning_rate": "float",
                "depth": "int",
                "l2_leaf_reg": "float",
                "model_size_reg": "float",
                "rsm": "float",
                "loss_function": "str",
                "border_count": "int",
                "feature_border_type": ["Median", "Uniform", "GreedyLogSum", "MinEntropy", "MaxLogSum"],
                "thread_count": "int",
                "random_seed": "int",
                "logging_level": ["Silent", "Verbose", "Info", "Debug"],
                "od_type": ["IncToDec", "Iter"],
                "od_wait": "int",
                "boosting_type": ["Ordered", "Plain"],
                "bagging_temperature": "float",
                "leaf_estimation_method": ["Newton", "Gradient"],
                "early_stopping_rounds": "int"
            },
            "random_forest": {
                "n_estimators": "int",
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": "int",
                "min_samples_split": "int or float",
                "min_samples_leaf": "int or float",
                "min_weight_fraction_leaf": "float",
                "max_features": ["auto", "sqrt", "log2", None, "int", "float"],
                "max_leaf_nodes": "int",
                "min_impurity_decrease": "float",
                "bootstrap": "bool",
                "oob_score": "bool",
                "n_jobs": "int",
                "random_state": "int",
                "verbose": "int",
                "ccp_alpha": "float",
                "max_samples": "int or float",
                "class_weight": "auto_weight",
            },
            "decision_tree": {
                "criterion": ["gini", "entropy", "log_loss"],
                "splitter": ["best", "random"],
                "max_depth": "int",
                "min_samples_split": "int or float",
                "min_samples_leaf": "int or float",
                "min_weight_fraction_leaf": "float",
                "max_features": ["auto", "sqrt", "log2", None, "int", "float"],
                "random_state": "int",
                "max_leaf_nodes": "int",
                "min_impurity_decrease": "float",
                "class_weight": "auto_weight",
                "ccp_alpha": "float"
            },
            "svc": {
                "C": "float",
                "kernel": ["linear", "poly", "rbf", "sigmoid"],
                "degree": "int",
                "gamma": ["scale", "auto", "float"],
                "coef0": "float",
                "shrinking": "bool",
                "probability": "bool",
                "tol": "float",
                "cache_size": "float",
                "class_weight": "auto_weight",
                "verbose": "bool",
                "max_iter": "int",
                "decision_function_shape": ["ovo", "ovr"],
                "break_ties": "bool",
                "random_state": "int"
            },
            "svr": {
                "kernel": ["linear", "poly", "rbf", "sigmoid"],
                "degree": "int",
                "gamma": ["scale", "auto", "float"],
                "coef0": "float",
                "tol": "float",
                "C": "float",
                "epsilon": "float",
                "shrinking": "bool",
                "cache_size": "float",
                "verbose": "bool",
                "max_iter": "int"
            },
            "linear_regression": {
                "fit_intercept": "bool",
                "normalize": "bool", 
                "copy_X": "bool",
                "n_jobs": "int",
                "positive": "bool"
            },
            "logistic_regression": {
                "penalty": ["l1", "l2", "elasticnet", "none"],
                "dual": "bool",
                "tol": "float",
                "C": "float",
                "fit_intercept": "bool",
                "intercept_scaling": "float",
                "class_weight": "auto_weight",
                "random_state": "int",
                "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                "max_iter": "int",
                "multi_class": ["auto", "ovr", "multinomial"],
                "verbose": "int",
                "warm_start": "bool",
                "n_jobs": "int",
                "l1_ratio": "float"
            }
        }
        
    def _compute_class_weights(self, y: pd.Series) -> Dict[str, Any]:
        """
        Compute different class weight strategies based on the target distribution.
        
        Args:
            y: Target variable
        
        Returns:
            Dictionary of different class weight strategies
        """
        if self.task_type != "classification":
            return {}
            
        class_counts = pd.Series(y).value_counts()
        n_samples = len(y)
        n_classes = len(class_counts)
        
        weights = {}

        balanced = {i: n_samples / (n_classes * count) for i, count in class_counts.items()}
        weights["balanced"] = balanced
        
        balanced_subsample = {i: 1.0 / count for i, count in class_counts.items()}
        weights["balanced_subsample"] = balanced_subsample
        
        sqrt_balanced = {i: np.sqrt(n_samples / (n_classes * count)) for i, count in class_counts.items()}
        weights["sqrt_balanced"] = sqrt_balanced
        
        log_balanced = {i: np.log1p(n_samples / (n_classes * count)) for i, count in class_counts.items()}
        weights["log_balanced"] = log_balanced
        
        if n_classes == 2:
            minority_class = class_counts.idxmin()
            majority_class = class_counts.idxmax()
            
            weights["ratio_1_1"] = {majority_class: 1.0, minority_class: 1.0}
            weights["ratio_1_2"] = {majority_class: 1.0, minority_class: 2.0}
            weights["ratio_1_3"] = {majority_class: 1.0, minority_class: 3.0}
            weights["ratio_1_5"] = {majority_class: 1.0, minority_class: 5.0}
            
            majority_count = class_counts[majority_class]
            minority_count = class_counts[minority_class]
            
            weights["scale_pos_weight"] = majority_count / minority_count
            weights["scale_pos_weight_sqrt"] = np.sqrt(majority_count / minority_count)
        
        return weights
    
    def _suggest_param(self, trial: optuna.Trial, name: str, param_config: Any, y: Optional[pd.Series] = None) -> Any:
        """
        Suggest a parameter value based on its configuration.
        """
        if name.endswith("_type"):
            return None
        
        if param_config == "auto_weight" and y is not None and self.task_type == "classification":
            weights = self._compute_class_weights(y)
            
            if name == "class_weight":
                class_weight_options = []
                
                class_weight_options.append(None)
                
                if len(np.unique(y)) == 2:
                    class_weight_options.extend([
                        weights["balanced"], 
                        weights["sqrt_balanced"], 
                        weights["log_balanced"],
                        weights["ratio_1_1"], 
                        weights["ratio_1_2"], 
                        weights["ratio_1_3"],
                        weights["ratio_1_5"]
                    ])
                else:
                    class_weight_options.extend([
                        weights["balanced"], 
                        weights["sqrt_balanced"], 
                        weights["log_balanced"]
                    ])
                    
                return trial.suggest_categorical(name, class_weight_options)
                
            elif name == "scale_pos_weight" and len(np.unique(y)) == 2:
                scale_options = [
                    1.0,
                    weights["scale_pos_weight"],
                    weights["scale_pos_weight_sqrt"],
                    weights["scale_pos_weight"] * 0.75
                ]
                return trial.suggest_categorical(name, scale_options)
        
        if isinstance(param_config, list):
            if name == "class_weight" and "LGBM" in str(self.base_model.models_to_run):
                options = [x for x in param_config if x != "dict"]
                if "None" in options:
                    options[options.index("None")] = None
                return trial.suggest_categorical(name, options)
            return trial.suggest_categorical(name, param_config)
        
        elif param_config == "int":
            if "n_estimators" in name or "iterations" in name:
                return trial.suggest_int(name, 50, 1000)
            elif "max_depth" in name:
                return trial.suggest_int(name, 3, 32)
            elif "num_leaves" in name or "max_leaves" in name:
                return trial.suggest_int(name, 8, 256)
            elif "min_samples" in name or "min_child_samples" in name:
                return trial.suggest_int(name, 1, 50)
            elif "max_bin" in name:
                return trial.suggest_int(name, 32, 512)
            elif "max_iter" in name:
                return trial.suggest_int(name, 100, 2000)
            elif "degree" in name:
                return trial.suggest_int(name, 1, 5)
            elif "n_jobs" in name or "thread_count" in name:
                return -1
            elif "verbose" in name or "verbosity" in name:
                return 0 if "verbosity" in name else 0
            elif "random_state" in name:
                return self.random_state
            elif "seed" in name and "random_seed" not in name:
                return self.random_state
            else:
                return trial.suggest_int(name, 1, 100)
                
        elif param_config == "float":
            if "learning_rate" in name:
                return trial.suggest_float(name, 0.001, 0.3, log=True)
            elif "subsample" in name or "colsample" in name:
                return trial.suggest_float(name, 0.5, 1.0)
            elif "reg_alpha" in name or "reg_lambda" in name or "alpha" in name or "lambda" in name or "l1" in name or "l2" in name:
                return trial.suggest_float(name, 1e-8, 10.0, log=True)
            elif "min_child_weight" in name:
                return trial.suggest_float(name, 0.1, 10.0, log=True)
            elif "C" in name:
                return trial.suggest_float(name, 1e-4, 1e4, log=True)
            elif "gamma" in name and param_config == "float":
                return trial.suggest_float(name, 1e-8, 10.0, log=True)
            elif "tol" in name:
                return trial.suggest_float(name, 1e-6, 1e-2, log=True)
            elif "epsilon" in name:
                return trial.suggest_float(name, 0.01, 1.0)
            else:
                return trial.suggest_float(name, 0.0, 1.0)
                
        elif param_config == "bool":
            return trial.suggest_categorical(name, [True, False])
            
        elif param_config.startswith("int or float"):
            if trial.suggest_categorical(f"{name}_type", ["int", "float"]) == "int":
                return trial.suggest_int(name, 1, 50)
            else:
                return trial.suggest_float(name, 0.01, 0.5)
                
        elif param_config == "str":
            if "objective" in name:
                if self.task_type == "classification":
                    return "binary:logistic" if hasattr(self.base_model, 'n_classes_') and self.base_model.n_classes_ == 2 else "multi:softprob"
                else:
                    return "reg:squarederror"
            elif "eval_metric" in name:
                if self.task_type == "classification":
                    return "auc" if hasattr(self.base_model, 'n_classes_') and self.base_model.n_classes_ == 2 else "mlogloss"
                else:
                    return "rmse"
            elif "loss_function" in name:
                if self.task_type == "classification":
                    return "Logloss" if hasattr(self.base_model, 'n_classes_') and self.base_model.n_classes_ == 2 else "MultiClass"
                else:
                    return "RMSE"
            else:
                return None
        else:
            return None
    
    def _get_param_space(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """
        Define the hyperparameter search space for a specific model.
        """
        params: Dict[str, Any] = {}
        

        model_config_map = {
            "Random Forest": "random_forest",
            "Decision Tree": "decision_tree",
            "XGBoost": "xgboost",
            "LGBM": "lightgbm",
            "CatBoost": "catboost",
            "SVC": "svc",
            "SVR": "svr",
            "Logistic Regression": "logistic_regression",
            "Linear Regression": "linear_regression",
            "KNN": None
        }
        

        config_key = model_config_map.get(model_name)
        

        if config_key and config_key in self.param_config:
            config = self.param_config[config_key]
            for param_name, param_config in config.items():
                if self.task_type == "classification" and param_name in ["epsilon"]:
                    continue
                if self.task_type == "regression" and param_name in ["class_weight", "probability"]:
                    continue
                
                if model_name == "Logistic Regression":
                    if "penalty" in params and params["penalty"] == "none" and param_name == "C":
                        continue
                    if "penalty" in params and params["penalty"] not in ["l1", "elasticnet"] and param_name == "l1_ratio":
                        continue
                
                # Skip random_state for CatBoost since we'll use random_seed instead
                if model_name == "CatBoost" and param_name == "random_state":
                    continue
                
                if model_name == "LGBM" and param_name == "class_weight":
                    param_value = trial.suggest_categorical(param_name, ["balanced", None])
                elif param_name in ["class_weight", "scale_pos_weight"] and param_config in ["auto_weight", "auto_scale"]:
                    y = getattr(self, '_current_y', None)
                    param_value = self._suggest_param(trial, param_name, param_config, y)
                else:
                    param_value = self._suggest_param(trial, param_name, param_config)
                
                if param_value is not None:
                    params[param_name] = param_value
        

        if model_name == "KNN":
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 40),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "p": trial.suggest_int("p", 1, 2)
            }
        

        if model_name == "XGBoost":
            if self.task_type == "classification":
                boosting_params = self.base_model._get_boosting_params()['xgb']
                params["objective"] = boosting_params['objective']
                params["eval_metric"] = boosting_params['eval_metric']
                if hasattr(self.base_model, 'n_classes_') and self.base_model.n_classes_ > 2:
                    params["num_class"] = self.base_model.n_classes_
            # Ensure verbosity is in acceptable range (0-3)
            if "verbosity" in params:
                params["verbosity"] = min(params.get("verbosity", 0), 3)
            # Make sure we don't have both seed and random_state
            if "seed" in params and "random_state" in params:
                del params["seed"]
        
        elif model_name == "LGBM":
            if self.task_type == "classification":
                boosting_params = self.base_model._get_boosting_params()['lgbm']
                params["objective"] = "binary" if not hasattr(self.base_model, 'n_classes_') or self.base_model.n_classes_ == 2 else "multiclass"
                params["metric"] = boosting_params['metric']
                if hasattr(self.base_model, 'n_classes_') and self.base_model.n_classes_ > 2:
                    params["num_class"] = self.base_model.n_classes_
            params["verbosity"] = -1
        
        elif model_name == "CatBoost":
            if self.task_type == "classification":
                boosting_params = self.base_model._get_boosting_params()['catboost']
                params["loss_function"] = boosting_params['loss_function']
            params["verbose"] = False
            # For CatBoost, use random_seed instead of random_state
            if "random_seed" not in params:
                params["random_seed"] = self.random_state
            # Make sure we don't have both random_seed and random_state
            if "random_state" in params:
                del params["random_state"]


        if "random_state" in params or model_name in ["Decision Tree", "Random Forest", "LGBM", "XGBoost", "Logistic Regression", "SVC"]:
            params["random_state"] = self.random_state
        
        return params
    
    def _objective(
            self, 
            trial: optuna.Trial, 
            X: pd.DataFrame, 
            y: pd.Series, 
            model_name: str, validation_size: float = 0.2
        ) -> float:
        """
        Objective function for Optuna optimization.
        """
        start_time: float = time.time()
        
        try:
            self._current_y = y
            
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
            

            if model_name == "XGBoost":
                if X_val_proc is not None:
                    model_instance.fit(X_train_proc, y_train, eval_set=[(X_val_proc, y_val)], verbose=0)
                else:
                    model_instance.fit(X_train_proc, y_train, verbose=0)
            elif model_name == "LGBM":
                if X_val_proc is not None:
                    model_instance.fit(X_train_proc, y_train, eval_set=[(X_val_proc, y_val)])
                else:
                    model_instance.fit(X_train_proc, y_train)
            else:
                model_instance.fit(X_train_proc, y_train)
            
            score: float = self.base_model._evaluate_model_score(model_instance, X_val_proc, y_val)
            
            elapsed_time: float = time.time() - start_time
            logger.info(f"Trial {trial.number} for {model_name} - {self.metric}: {score:.4f} (Time: {elapsed_time:.2f}s)")
            
            return score
            
        except Exception as e:
            logger.error(f"Error in trial for {model_name}: {e}", exc_info=True)
            raise optuna.exceptions.TrialPruned(f"Trial for {model_name} failed with error: {str(e)}")
        finally:
            if hasattr(self, '_current_y'):
                delattr(self, '_current_y')

    def _clean_params_for_model(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean parameters to ensure they're compatible with the target model.
        
        Removes parameters that aren't accepted by the model constructor.
        """
        model_class = self.base_model.models_to_run[model_name].__class__
        valid_params = {}
        

        from inspect import signature
        valid_param_names = list(signature(model_class.__init__).parameters.keys())
        

        if 'self' in valid_param_names:
            valid_param_names.remove('self')
        
        # Special handling for CatBoost - use only random_seed, not random_state
        if model_name == "CatBoost":
            if "random_seed" in params:
                if "random_state" in params:
                    del params["random_state"]
            elif "random_state" in params:
                # Rename random_state to random_seed for CatBoost
                params["random_seed"] = params["random_state"]
                del params["random_state"]

        for param_name, param_value in params.items():
            if param_name.endswith('_type'):
                continue
 
            if param_name in valid_param_names:
                valid_params[param_name] = param_value
        
        return valid_params

    def optimize(
            self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            model_name: Optional[str] = None, 
            validation_size: float = 0.2, **kwargs
        ) -> Dict[str, Dict[str, Any]]:
        """
        Optimize hyperparameters for a specific model or all models.
        """
        models_to_optimize: List[str] = [model_name] if model_name else self.models_to_tune
        
        if self.task_type == "classification" and not hasattr(self.base_model, 'n_classes_'):
            self.base_model.n_classes_ = len(np.unique(y))
        
        self.base_model.models_to_run = self.base_model._initialize_models()

        optimization_results: Dict[str, Dict[str, Any]] = {}
        
        if self.task_type == "classification":
            self.class_weights = self._compute_class_weights(y)
        
        for model_name in models_to_optimize:
            if model_name not in self.base_model.models_to_run:
                logger.warning(f"Model {model_name} not found in available models. Skipping.")
                continue
                
            model_n_trials = self.n_trials.get(model_name, 50)
            model_timeout = self.timeout.get(model_name, None)
            
            logger.info(f"Optimizing {model_name} hyperparameters with {model_n_trials} trials (timeout: {model_timeout if model_timeout else 'None'})...")
            
            study: optuna.study.Study = optuna.create_study(direction=self.optimize_direction, **kwargs)
            
            objective_func = lambda trial: self._objective(
                trial=trial, X=X, y=y, model_name=model_name, validation_size=validation_size
            )
            
            study.optimize(
                objective_func, 
                n_trials=model_n_trials,
                timeout=model_timeout
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
            
            if model_name in ["Decision Tree", "Random Forest", "LGBM", "XGBoost"]:
                best_params["random_state"] = self.random_state
            elif model_name == "CatBoost":
                best_params["random_seed"] = self.random_state
                # Remove random_state if it exists to avoid conflict
                if "random_state" in best_params:
                    del best_params["random_state"]
                
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
            
            best_params = self._clean_params_for_model(model_name, best_params)
            
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
        n_trials: Union[int, Dict[str, int]] = 30,
        timeout: Optional[Union[int, Dict[str, int]]] = None
    ) -> None:
        """
        Initialize the TunedBaselineModel.
        
        Args:
            task_type: Type of ML task ('classification' or 'regression')
            metric: Evaluation metric to optimize
            random_state: Seed for reproducibility
            models: Which model set to use ('all', 'trees', or 'gbm')
            n_trials: Number of optimization trials. Can be:
                    - An integer (same for all models)
                    - A dictionary mapping model names or categories to trial counts
                    Example: {'XGBoost': 50, 'boosting': 30, 'trees': 20}
            timeout: Maximum optimization time. Can be:
                    - None for no limit
                    - An integer (same for all models) 
                    - A dictionary mapping model names or categories to timeouts
                    Example: {'XGBoost': 600, 'boosting': 300, 'trees': 120}
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
    
    def fit(
            self, 
            X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray],
            validation_size: float = 0.2, 
            optimize_first: bool = True, 
            **kwargs
        ) -> Dict[str, Dict[str, Any]]:
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
                    # Make sure we're using clean parameters
                    clean_params = self.optimizer._clean_params_for_model(model_name, params)
                    self.models_to_run[model_name] = model_class(**clean_params)
    
        results = super().fit(X, y, validation_size, self.random_state, **kwargs)
        
        return results
    
    def optimize_single_model(
            self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            model_name: str, 
            validation_size: float = 0.2, **kwargs
        ) -> Dict[str, Any]:
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
        clean_params = self.optimizer._clean_params_for_model(model_name, self.best_params[model_name])
        self.models_to_run[model_name] = model_class(**clean_params)
        
        return optimization_result[model_name]
