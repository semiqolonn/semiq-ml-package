# tuning.py
from .baseline_model import BaselineModel # Assuming BaselineModel is in the same package
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
import numpy as np
import pandas as pd
import time
import logging

logger = logging.getLogger(__name__)

class BaseOptimizer(BaselineModel):
    """
    Base class for hyperparameter optimizers, handling common logic.
    """
    def __init__(self, search_strategy_cv, task_type="classification", metric=None, random_state=42, cv=5, n_iter_or_None=None):
        super().__init__(task_type=task_type, metric=metric, random_state=random_state)
        self.cv = cv
        # n_iter for RandomizedSearch, None for GridSearch (which is exhaustive)
        self.n_iter_or_None = n_iter_or_None
        self.search_strategy_cv = search_strategy_cv # RandomizedSearchCV or GridSearchCV class
        self.param_config = self._get_default_param_config() # To be implemented by child
        self.tuned_models = {} # Stores results for each tuned model

    def _get_default_param_config(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def update_param_config(self, model_name, param_dict):
        if model_name not in self.models_to_run:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models_to_run.keys())}")
        
        if model_name in self.param_config:
            self.param_config[model_name] = param_dict
            logger.info(f"Updated parameter configuration for {model_name}")
        else: # Should not happen if _get_default_param_config is comprehensive
            self.param_config[model_name] = param_dict
            logger.warning(f"Parameter configuration for {model_name} was not pre-defined. Added new entry.")

    def _prepare_search_estimator_and_params(self, model_name, base_model_instance, X_search_train_raw, current_model_param_config):
        """
        Prepares the estimator forSearchCV (potentially a Pipeline) and adjusts parameter keys.
        """
        preprocessor_type = self._get_model_type(model_name) # From BaselineModel
        search_cv_fit_params = {} # For fit params like cat_features

        if preprocessor_type == 'catboost_internal':
            estimator_for_search = base_model_instance
            param_config_for_searchcv = current_model_param_config # No prefix needed

            if isinstance(X_search_train_raw, pd.DataFrame):
                cat_features_indices = [
                    X_search_train_raw.columns.get_loc(col)
                    for col in X_search_train_raw.select_dtypes(include=['object', 'category']).columns
                    if col in X_search_train_raw.columns
                ]
                if cat_features_indices:
                    logger.info(f"CatBoost: Determined cat_features indices: {cat_features_indices} for search.")
                    search_cv_fit_params["cat_features"] = cat_features_indices
            else:
                logger.warning("CatBoost is selected, but X_search_train_raw is not a DataFrame. Cannot auto-determine cat_features.")
        else: # 'general_ohe' or 'distance_kernel'
            # Build unfitted preprocessor using X_search_train_raw for dtype inference
            unfitted_preprocessor = self._build_preprocessor(X_search_train_raw, preprocessor_type) # From BaselineModel

            if unfitted_preprocessor:
                estimator_for_search = Pipeline([
                    ('preprocessor', unfitted_preprocessor),
                    ('model', base_model_instance)
                ])
                # Prefix params for pipeline: e.g., 'C' -> 'model__C'
                param_config_for_searchcv = {f"model__{k}": v for k, v in current_model_param_config.items()}
            else: # No preprocessor built (e.g., data was all numeric and simple)
                estimator_for_search = base_model_instance
                param_config_for_searchcv = current_model_param_config # No prefix
        
        return estimator_for_search, param_config_for_searchcv, search_cv_fit_params

    def tune_model(self, model_name, X, y, validation_size=0.2, **caller_fit_params):
        if model_name not in self.models_to_run:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models_to_run.keys())}")

        current_model_param_config = self.param_config.get(model_name, {})
        if not current_model_param_config:
            logger.warning(f"No parameter configuration for {model_name}. Using default model from BaselineModel.")
            # Fallback: run the base model from BaselineModel (or skip tuning)
            # For simplicity, we'll try to fit the base model once here, though this is not tuning.
            # A more robust approach would be to skip or raise error if no params.
            base_model_instance = self.models_to_run[model_name]
            # This path needs careful thought: what should happen if no params for tuning?
            # For now, let's assume `_get_default_param_config` always provides something, even empty for LinearRegression
            if model_name == "Linear Regression": # Special case, no tuning needed
                 logger.info(f"Linear Regression requires no hyperparameter tuning. Evaluating base model.")
                 # We need to evaluate it on a validation set consistent with how tuned models are scored.
                 # This part duplicates some logic from BaselineModel.fit() just for this one model.
                 # A more robust approach would be to call a part of BaselineModel.fit logic.
                 # For now, we'll just store it as "tuned" with no params.
                 self.tuned_models[model_name] = {
                     "model": base_model_instance, "score": None, "time": 0, 
                     "best_params": {}, "status": "No tuning (base model)"
                 }
                 # To get a score, we'd need to process data and evaluate.
                 # This is complex here. Better to ensure tune_model is only called if params exist.
                 return base_model_instance


        stratify_opt = y if self.task_type == "classification" else None
        X_search_train_raw, X_search_val_raw, y_search_train, y_search_val = train_test_split(
            X, y, test_size=validation_size, random_state=self.random_state, stratify=stratify_opt
        )

        base_model_instance = self.models_to_run[model_name]
        
        estimator_for_search, param_config_for_searchcv, search_cv_specific_fit_params = \
            self._prepare_search_estimator_and_params(model_name, base_model_instance, X_search_train_raw, current_model_param_config)

        # Combine caller_fit_params (for model) with search_cv_specific_fit_params (e.g. cat_features)
        # Caller fit_params need prefixing if a pipeline is used.
        final_search_cv_fit_params = search_cv_specific_fit_params.copy()
        if isinstance(estimator_for_search, Pipeline):
            for k, v in caller_fit_params.items():
                final_search_cv_fit_params[f"model__{k}"] = v
        else: # Direct model (e.g. CatBoost)
            final_search_cv_fit_params.update(caller_fit_params)
            
        # Ensure scorer uses the metric and maximization direction from BaselineModel
        # self.metric and self.maximize_metric are from BaselineModel's __init__
        # self._metric_functions is also from BaselineModel
        if self.metric in self._metric_functions:
            # For ROC AUC and LogLoss, make_scorer needs predict_proba=True
            needs_proba = self.metric in ("roc_auc", "auc", "log_loss")
            scorer = make_scorer(
                self._metric_functions[self.metric],
                greater_is_better=self.maximize_metric,
                needs_proba=needs_proba
            )
        else: # Should not happen if metric validation in BaselineModel is robust
            scorer = self.metric 
            logger.warning(f"Metric '{self.metric}' not found in _metric_functions. Using it directly as string for scorer.")


        search_cv_params = {
            "estimator": estimator_for_search,
            "cv": self.cv,
            "scoring": scorer,
            "n_jobs": -1,
            "random_state": self.random_state,
            "verbose": 1 # Can be parameterized
        }
        if self.search_strategy_cv == RandomizedSearchCV:
            search_cv_params["param_distributions"] = param_config_for_searchcv
            search_cv_params["n_iter"] = self.n_iter_or_None
        elif self.search_strategy_cv == GridSearchCV:
            search_cv_params["param_grid"] = param_config_for_searchcv
        else:
            raise TypeError("search_strategy_cv is not recognized.")

        logger.info(f"Starting {self.search_strategy_cv.__name__} for {model_name}...")
        if self.n_iter_or_None and self.search_strategy_cv == RandomizedSearchCV:
            logger.info(f"...with {self.n_iter_or_None} iterations and {self.cv}-fold CV.")
        else:
            logger.info(f"...with {self.cv}-fold CV.")
        
        start_time = time.time()
        
        try:
            search_instance = self.search_strategy_cv(**search_cv_params)
            search_instance.fit(X_search_train_raw, y_search_train, **final_search_cv_fit_params)
            
            best_tuned_estimator = search_instance.best_estimator_
            
            # Evaluate the best_tuned_estimator (which is fitted on X_search_train_raw) 
            # on the holdout set X_search_val_raw, y_search_val.
            # _evaluate_model_score should handle the pipeline correctly.
            score_on_holdout = self._evaluate_model_score(best_tuned_estimator, X_search_val_raw, y_search_val)
            elapsed_time = time.time() - start_time

            best_params_from_search = search_instance.best_params_
            # Clean up parameter names if they were prefixed
            if isinstance(estimator_for_search, Pipeline):
                cleaned_best_params = {k.replace("model__", ""): v for k, v in best_params_from_search.items()}
            else:
                cleaned_best_params = best_params_from_search

            self.tuned_models[model_name] = {
                "model": best_tuned_estimator,
                "score_on_holdout": score_on_holdout, # Score on the separate validation set
                "best_cv_score": search_instance.best_score_, # Score from CV on X_search_train_raw
                "time": elapsed_time,
                "best_params": cleaned_best_params,
                "cv_results_summary": pd.DataFrame(search_instance.cv_results_).head() # Example summary
            }
            
            logger.info(f"Tuning {model_name} completed in {elapsed_time:.2f}s")
            logger.info(f"Best CV {self.metric}: {search_instance.best_score_:.4f}")
            logger.info(f"Holdout {self.metric}: {score_on_holdout:.4f}")
            logger.info(f"Best parameters (cleaned): {cleaned_best_params}")
            
            return best_tuned_estimator
            
        except Exception as e:
            logger.error(f"Error tuning {model_name}: {e}", exc_info=True)
            self.tuned_models[model_name] = {
                "model": None, "score_on_holdout": None, "best_cv_score": None,
                "time": None, "error": str(e)
            }
            return None

    def tune_all_models(self, X, y, validation_size=0.2, models_to_tune=None, **fit_params):
        if models_to_tune is None:
            models_to_tune = [name for name, params in self.param_config.items() if params] # Tune only if params exist
        else:
            valid_models = []
            for model_name in models_to_tune:
                if model_name not in self.models_to_run:
                    logger.warning(f"Model '{model_name}' for tuning not found in base models. Skipping.")
                elif not self.param_config.get(model_name):
                    logger.warning(f"Model '{model_name}' has no parameters defined for tuning. Skipping.")
                else:
                    valid_models.append(model_name)
            models_to_tune = valid_models
        
        if not models_to_tune:
            logger.warning("No models eligible for tuning.")
            return {}

        logger.info(f"Attempting to tune {len(models_to_tune)} models: {', '.join(models_to_tune)}")
        
        # Use score_on_holdout for tracking overall best model from this tuning session
        # Initialize based on self.maximize_metric from BaselineModel
        overall_best_holdout_score = -np.inf if self.maximize_metric else np.inf
        overall_best_model_name = None
        
        for model_name in models_to_tune:
            logger.info(f"\n--- Tuning {model_name} ---")
            self.tune_model(model_name, X, y, validation_size, **fit_params)
            
            current_result = self.tuned_models.get(model_name, {})
            current_holdout_score = current_result.get("score_on_holdout")

            if current_holdout_score is not None:
                if (self.maximize_metric and current_holdout_score > overall_best_holdout_score) or \
                   (not self.maximize_metric and current_holdout_score < overall_best_holdout_score):
                    overall_best_holdout_score = current_holdout_score
                    overall_best_model_name = model_name
        
        if overall_best_model_name and overall_best_model_name in self.tuned_models:
            self.best_model_ = self.tuned_models[overall_best_model_name]["model"] # The best estimator (Pipeline or model)
            self.best_score_ = overall_best_holdout_score # Score on holdout set
            logger.info(f"\n--- Overall Best Tuned Model (based on holdout score) ---")
            logger.info(f"Model: {overall_best_model_name}")
            logger.info(f"{self.metric}: {self.best_score_:.4f}")
            logger.info(f"Parameters: {self.tuned_models[overall_best_model_name].get('best_params')}")
        else:
            logger.warning("No models were successfully tuned to determine an overall best model.")
        
        return self.tuned_models

    def get_tuning_results(self):
        if not self.tuned_models:
            logger.info("No tuning results available.")
            return pd.DataFrame()
            
        results_data = []
        for name, info in self.tuned_models.items():
            entry = {
                "model_name": name,
                "holdout_score": info.get("score_on_holdout"),
                "best_cv_score": info.get("best_cv_score"),
                "tuning_time_seconds": info.get("time"),
                "best_params": str(info.get("best_params", {})), # Convert dict to str for easier DF display
                "status": "Success" if info.get("model") is not None else "Failed",
                "error": info.get("error", "")
            }
            results_data.append(entry)
                
        results_df = pd.DataFrame(results_data)
        
        if not results_df.empty and "holdout_score" in results_df.columns:
            # Sort by holdout_score, handling potential None values by placing them last
            results_df = results_df.sort_values(
                by="holdout_score",
                ascending=not self.maximize_metric,
                na_position='last'
            )
        return results_df.reset_index(drop=True)


class RandomSearchOptimizer(BaseOptimizer):
    def __init__(self, task_type="classification", metric=None, random_state=42, n_iter=10, cv=5):
        super().__init__(RandomizedSearchCV, task_type, metric, random_state, cv, n_iter_or_None=n_iter)

    def _get_default_param_config(self): # Renamed from _get_default_param_distributions
        # Parameter distributions from the original RandomSearchOptimizer class
        # These are extensive. For brevity, I'll assume they are the same as provided.
        # Make sure they are appropriate for RandomizedSearchCV (can include distributions).
        if self.task_type == "classification":
            return {
                "Logistic Regression": {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2'], 'solver': ['liblinear']},
                "SVC": {'C': np.logspace(-3, 3, 7), 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto'] + list(np.logspace(-3, 0, 4))}, # Reduced kernels for demo
                "KNN": {'n_neighbors': np.arange(3, 31, 2), 'weights': ['uniform', 'distance'], 'p': [1, 2]},
                "Decision Tree": {'max_depth': [None] + list(np.arange(5, 21, 5)), 'min_samples_split': np.arange(2, 11), 'min_samples_leaf': np.arange(1, 6)}, # Reduced
                "Random Forest": {'n_estimators': np.arange(50, 201, 50), 'max_depth': [None] + list(np.arange(10, 21, 5)), 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}, # Reduced
                "LGBM": {'learning_rate': np.logspace(-3, -1, 3), 'n_estimators': np.arange(50, 201, 50), 'num_leaves': [31, 63], 'max_depth': [-1, 10]}, # Reduced
                "XGBoost": {'learning_rate': np.logspace(-3, -1, 3), 'n_estimators': np.arange(50, 201, 50), 'max_depth': [3, 6]}, # Reduced
                "CatBoost": {'iterations': np.arange(50, 201, 50), 'learning_rate': np.logspace(-3, -1, 3), 'depth': [4, 6, 8]} # Reduced
            }
        else:  # Regression
            return {
                "Linear Regression": {},
                "SVR": {'C': np.logspace(-3, 2, 5), 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto'] + list(np.logspace(-3, 0, 3)), 'epsilon': np.logspace(-3, 0, 3)}, # Reduced
                "KNN": {'n_neighbors': np.arange(3, 21, 2), 'weights': ['uniform', 'distance'], 'p': [1, 2]}, # Reduced
                "Decision Tree": {'max_depth': [None] + list(np.arange(5, 21, 5)), 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}, # Reduced
                "Random Forest": {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 15], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}, # Reduced
                "LGBM": {'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [50, 100, 150], 'num_leaves': [31, 63], 'max_depth': [-1, 10]}, # Reduced
                "XGBoost": {'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 7]}, # Reduced
                "CatBoost": {'iterations': [50, 100, 150], 'learning_rate': [0.01, 0.05, 0.1], 'depth': [4, 6, 8]} # Reduced
            }

class GridSearchOptimizer(BaseOptimizer):
    def __init__(self, task_type="classification", metric=None, random_state=42, cv=5):
        super().__init__(GridSearchCV, task_type, metric, random_state, cv, n_iter_or_None=None)

    def _get_default_param_config(self): # Renamed from _get_default_param_grid
        # Parameter grids from the original GridSearchOptimizer class
        # These should be lists of specific values for GridSearchCV.
        # For brevity, I'll assume they are the same as provided.
        if self.task_type == "classification":
            return {
                "Logistic Regression": {'C': [0.1, 1.0, 10.0], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']},
                "SVC": {'C': [0.1, 1.0, 10.0], 'kernel': ['rbf'], 'gamma': ['scale', 0.1]}, # Highly reduced
                "KNN": {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}, # Reduced
                "Decision Tree": {'max_depth': [5, 10], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}, # Reduced
                "Random Forest": {'n_estimators': [50, 100], 'max_depth': [10, 20], 'min_samples_split': [2, 5]}, # Reduced
                "LGBM": {'learning_rate': [0.05, 0.1], 'n_estimators': [50, 100], 'num_leaves': [31]}, # Reduced
                "XGBoost": {'learning_rate': [0.05, 0.1], 'n_estimators': [50, 100], 'max_depth': [3, 5]}, # Reduced
                "CatBoost": {'iterations': [50, 100], 'learning_rate': [0.05, 0.1], 'depth': [4, 6]} # Reduced
            }
        else:  # Regression
            return {
                "Linear Regression": {},
                "SVR": {'C': [0.1, 1.0], 'kernel': ['rbf'], 'gamma': ['scale', 0.1], 'epsilon': [0.1]}, # Highly reduced
                "KNN": {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}, # Reduced
                "Decision Tree": {'max_depth': [5, 10], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}, # Reduced
                "Random Forest": {'n_estimators': [50, 100], 'max_depth': [10], 'min_samples_split': [2, 5]}, # Reduced
                "LGBM": {'learning_rate': [0.05, 0.1], 'n_estimators': [50, 100], 'num_leaves': [31]}, # Reduced
                "XGBoost": {'learning_rate': [0.05, 0.1], 'n_estimators': [50, 100], 'max_depth': [3, 5]}, # Reduced
                "CatBoost": {'iterations': [50, 100], 'learning_rate': [0.05, 0.1], 'depth': [4, 6]} # Reduced
            }