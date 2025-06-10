# Optuna-based hyperparameter optimization for BaselineModel
import optuna
import numpy as np
import pandas as pd
import time
import logging
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from .baseline_model import BaselineModel

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class OptunaOptimizer:
    """
    Hyperparameter optimizer using Optuna for models in BaselineModel.
    """
    def __init__(self, baseline_model=None, task_type=None, metric=None, random_state=42, n_trials=100, cv=5):
        """
        Initialize OptunaOptimizer.
        
        Args:
            baseline_model: A fitted BaselineModel instance or None (will create new instance)
            task_type: 'classification' or 'regression' (used if baseline_model is None)
            metric: Metric to optimize (used if baseline_model is None)
            random_state: Random seed for reproducibility
            n_trials: Number of Optuna trials for optimization
            cv: Number of cross-validation folds
        """
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state
        
        # Either use provided BaselineModel or create new one
        if baseline_model is not None:
            self.baseline_model = baseline_model
            self.task_type = baseline_model.task_type
            self.metric = baseline_model.metric
            self.maximize_metric = baseline_model.maximize_metric
        else:
            if task_type is None:
                raise ValueError("If no baseline_model provided, task_type must be specified")
            self.task_type = task_type
            self.metric = metric
            self.baseline_model = BaselineModel(task_type=task_type, metric=metric, random_state=random_state)
            self.maximize_metric = self.baseline_model.maximize_metric

        self.tuned_models = {}
        self.study_objects = {}
        self.best_params = {}

    def _define_param_space(self, trial, model_name):
        """Define hyperparameter search space for different models."""
        if model_name == "Logistic Regression":
            return {
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': trial.suggest_categorical('solver', ['liblinear']),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            }
        elif model_name == "Linear Regression":
            return {
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            }
        elif model_name == "SVC" or model_name == "SVR":
            params = {
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']) if trial.suggest_categorical('gamma_choice', ['preset', 'custom']) == 'preset' 
                        else trial.suggest_float('gamma_value', 1e-4, 1e1, log=True),
                'degree': trial.suggest_int('degree', 2, 5) if trial.params.get('kernel') == 'poly' else 3,
                'coef0': trial.suggest_float('coef0', 0.0, 10.0) if trial.params.get('kernel') in ['poly', 'sigmoid'] else 0.0,
                'shrinking': trial.suggest_categorical('shrinking', [True, False]),
                'tol': trial.suggest_float('tol', 1e-5, 1e-2, log=True),
            }
            
            # Add SVR-specific parameters
            if model_name == "SVR":
                params['epsilon'] = trial.suggest_float('epsilon', 0.01, 1.0)
            # Add SVC-specific parameters
            else:
                params['probability'] = True  # Always use probability for classification
                params['class_weight'] = trial.suggest_categorical('class_weight', [None, 'balanced'])
                
            return params
            
        elif model_name == "KNN":
            return {
                'n_neighbors': trial.suggest_int('n_neighbors', 1, 30),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p': trial.suggest_int('p', 1, 2),
                'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                'leaf_size': trial.suggest_int('leaf_size', 10, 50),
            }
        elif model_name == "Decision Tree":
            criterion_choices = ['gini', 'entropy'] if self.task_type == 'classification' else ['squared_error', 'absolute_error', 'friedman_mse']
            
            return {
                'criterion': trial.suggest_categorical('criterion', criterion_choices),
                'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
                'max_depth': trial.suggest_int('max_depth', 3, 30) if trial.suggest_categorical('max_depth_choice', ['custom', 'None']) == 'custom' else None,
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
                'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 100) if trial.suggest_categorical('max_leaf_nodes_choice', ['custom', 'None']) == 'custom' else None,
                'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.2),
                'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.05),
            }
        elif model_name == "Random Forest":
            criterion_choices = ['gini', 'entropy'] if self.task_type == 'classification' else ['squared_error', 'absolute_error', 'friedman_mse']
            
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'criterion': trial.suggest_categorical('criterion', criterion_choices),
                'max_depth': trial.suggest_int('max_depth', 3, 30) if trial.suggest_categorical('max_depth_choice', ['custom', 'None']) == 'custom' else None,
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 100) if trial.suggest_categorical('max_leaf_nodes_choice', ['custom', 'None']) == 'custom' else None,
                'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.2),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'oob_score': trial.suggest_categorical('oob_score', [True, False]) if trial.params.get('bootstrap', True) else False,
                'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.05),
            }
        elif model_name == "LGBM":
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 15) if trial.suggest_categorical('max_depth_choice', ['custom', 'default']) == 'custom' else -1,
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'subsample': trial.suggest_float('subsample', 0.4, 1.0),
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
                'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 15.0),
                'max_bin': trial.suggest_int('max_bin', 200, 300),
            }
            
            # Add parameters specific to boosting_type
            if params['boosting_type'] == 'dart':
                params['drop_rate'] = trial.suggest_float('drop_rate', 0.1, 0.9)
                params['skip_drop'] = trial.suggest_float('skip_drop', 0.1, 0.9)
                params['max_drop'] = trial.suggest_int('max_drop', 20, 100)
            elif params['boosting_type'] == 'goss':
                params['top_rate'] = trial.suggest_float('top_rate', 0.1, 0.9)
                params['other_rate'] = trial.suggest_float('other_rate', 0.1, 0.9)
            
            return params
            
        elif model_name == "XGBoost":
            booster_type = trial.suggest_categorical('booster', ['gbtree', 'dart'])
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'booster': booster_type,
                'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            }
            
            if self.task_type == 'classification':
                params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 0.1, 10.0, log=True)
                
            # Add parameters specific to dart booster
            if booster_type == 'dart':
                params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
                params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
                params['rate_drop'] = trial.suggest_float('rate_drop', 0.0, 0.5)
                params['one_drop'] = trial.suggest_categorical('one_drop', [0, 1])
                params['skip_drop'] = trial.suggest_float('skip_drop', 0.0, 0.5)
                
            return params
            
        elif model_name == "CatBoost":
            params = {
                'iterations': trial.suggest_int('iterations', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10) if trial.suggest_categorical('use_bagging', [True, False]) else 0,
                'random_strength': trial.suggest_float('random_strength', 1e-8, 10, log=True),
                'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
                'rsm': trial.suggest_float('rsm', 0.1, 1.0),
                'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 10),
                'leaf_estimation_method': trial.suggest_categorical('leaf_estimation_method', ['Newton', 'Gradient']),
                'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),
            }
            
            # Add max_leaves for certain grow_policies
            if params['grow_policy'] in ['Lossguide']:
                params['max_leaves'] = trial.suggest_int('max_leaves', 10, 64)
                
            return params
        else:
            logger.warning(f"Model {model_name} not recognized, returning empty parameter space")
            return {}

    def _create_objective(self, model_name, X, y, fit_params=None):
        """Create an Optuna objective function for a specific model."""
        if fit_params is None:
            fit_params = {}
            
        # Get original model instance from baseline_model
        if model_name not in self.baseline_model.models_to_run:
            raise ValueError(f"Model {model_name} not found in baseline_model.models_to_run")
            
        model_type = self.baseline_model._get_model_type(model_name)
        original_model = self.baseline_model.models_to_run[model_name]
        
        # Define objective function
        def objective(trial):
            # Get hyperparameters for this trial
            trial_params = self._define_param_space(trial, model_name)
            if not trial_params:
                logger.info(f"No hyperparameters to tune for {model_name}, using default model")
                trial.set_user_attr('model_params', {})
                model = original_model
            else:
                # Special handling for parameters that need clean-up
                if 'gamma_choice' in trial.params and 'gamma' in trial_params:
                    if trial.params['gamma_choice'] == 'custom':
                        trial_params['gamma'] = trial_params.pop('gamma_value')
                        
                if 'max_depth_choice' in trial.params and 'max_depth' in trial_params:
                    if trial.params['max_depth_choice'] == 'None':
                        trial_params['max_depth'] = None
                
                # Create model with trial parameters
                model_class = original_model.__class__
                model = model_class(**{**original_model.get_params(), **trial_params})
                    
                # Record the actual parameters used
                cleaned_params = {k: v for k, v in trial_params.items() 
                                if k not in ['gamma_choice', 'max_depth_choice', 'use_bagging']}
                trial.set_user_attr('model_params', cleaned_params)
            
            try:
                # For CatBoost models, handle cat_features specially
                model_specific_fit_params = fit_params.copy()
                if model_name == "CatBoost" and isinstance(X, pd.DataFrame):
                    cat_features_indices = [
                        X.columns.get_loc(col)
                        for col in X.select_dtypes(include=['object', 'category']).columns
                        if col in X.columns
                    ]
                    if cat_features_indices:
                        model_specific_fit_params["cat_features"] = cat_features_indices
                
                # Cross-validation with appropriate preprocessing
                if model_type == 'catboost_internal':
                    # CatBoost handles categorical features internally
                    scores = cross_val_score(
                        model, X, y, cv=self.cv, scoring=self.metric, 
                        fit_params=model_specific_fit_params, n_jobs=-1
                    )
                else:
                    # Build preprocessor for other models
                    preprocessor = self.baseline_model._build_preprocessor(X, model_type)
                    if preprocessor:
                        from sklearn.pipeline import Pipeline
                        pipeline = Pipeline([
                            ('preprocessor', preprocessor),
                            ('model', model)
                        ])
                        scores = cross_val_score(
                            pipeline, X, y, cv=self.cv, scoring=self.metric, 
                            fit_params=model_specific_fit_params, n_jobs=-1
                        )
                    else:
                        scores = cross_val_score(
                            model, X, y, cv=self.cv, scoring=self.metric,
                            fit_params=model_specific_fit_params, n_jobs=-1
                        )
                
                avg_score = scores.mean()
                return avg_score if self.maximize_metric else -avg_score
                
            except Exception as e:
                logger.error(f"Error during trial: {e}")
                raise optuna.exceptions.TrialPruned()
                
        return objective

    def _get_direction(self):
        """Determine optimization direction based on metric."""
        return 'maximize' if self.maximize_metric else 'minimize'

    def tune_model(self, model_name, X, y, fit_params=None):
        """
        Tune hyperparameters for a specific model.
        
        Args:
            model_name: Name of the model to tune (must be in baseline_model.models_to_run)
            X: Feature data
            y: Target data
            fit_params: Additional parameters to pass to fit method
            
        Returns:
            Tuple of (best parameters, best score, best model instance)
        """
        logger.info(f"Starting Optuna optimization for {model_name} with {self.n_trials} trials")
        
        if model_name == "Linear Regression":
            logger.info("Linear Regression has no hyperparameters to tune. Returning original model.")
            model = self.baseline_model.models_to_run[model_name]
            self.tuned_models[model_name] = model
            self.best_params[model_name] = {}
            return {}, None, model
            
        # Create study
        study = optuna.create_study(direction=self._get_direction())
        objective = self._create_objective(model_name, X, y, fit_params)
        
        # Run optimization
        start_time = time.time()
        study.optimize(objective, n_trials=self.n_trials)
        elapsed_time = time.time() - start_time
        
        # Get best model params and create optimized model
        best_params = study.best_trial.user_attrs.get('model_params', {})
        best_score = study.best_value
        
        if best_params:
            model_class = self.baseline_model.models_to_run[model_name].__class__
            optimized_model = model_class(
                **{**self.baseline_model.models_to_run[model_name].get_params(), **best_params}
            )
            logger.info(f"Best {self.metric} for {model_name}: {best_score:.4f}")
            logger.info(f"Best parameters for {model_name}: {best_params}")
        else:
            optimized_model = self.baseline_model.models_to_run[model_name]
            logger.info(f"No hyperparameters tuned for {model_name}")
        
        # Store results
        self.study_objects[model_name] = study
        self.best_params[model_name] = best_params
        self.tuned_models[model_name] = optimized_model
        
        logger.info(f"Optimization for {model_name} completed in {elapsed_time:.2f} seconds")
        return best_params, best_score, optimized_model

    def tune_best_model(self, X, y, fit_params=None):
        """
        Tune the best model from the BaselineModel.
        
        Args:
            X: Feature data
            y: Target data
            fit_params: Additional parameters to pass to fit method
            
        Returns:
            Tuple of (best parameters, best score, best model instance)
        """
        if not hasattr(self.baseline_model, 'best_model_') or self.baseline_model.best_model_ is None:
            logger.warning("No best model found in BaselineModel. Running fit() first.")
            self.baseline_model.fit(X, y)
            
        # Find the name of the best model
        best_model_class = self.baseline_model.best_model_.__class__.__name__
        best_model_name = None
        
        for name, model in self.baseline_model.models_to_run.items():
            if model.__class__.__name__ == best_model_class:
                best_model_name = name
                break
                
        if best_model_name is None:
            logger.error("Could not identify best model name")
            return None, None, None
            
        logger.info(f"Tuning best model: {best_model_name}")
        return self.tune_model(best_model_name, X, y, fit_params)

    def tune_all_models(self, X, y, models_to_tune=None, fit_params=None):
        """
        Tune hyperparameters for multiple models.
        
        Args:
            X: Feature data
            y: Target data
            models_to_tune: List of model names to tune (None = all models)
            fit_params: Additional parameters to pass to fit method
            
        Returns:
            Dictionary with results for each model
        """
        if models_to_tune is None:
            models_to_tune = list(self.baseline_model.models_to_run.keys())
        else:
            # Filter only models that exist in baseline_model
            models_to_tune = [m for m in models_to_tune if m in self.baseline_model.models_to_run]
            if not models_to_tune:
                logger.warning("None of the specified models found in baseline_model")
                return {}
                
        results = {}
        for model_name in models_to_tune:
            logger.info(f"\n--- Tuning {model_name} ---")
            best_params, best_score, optimized_model = self.tune_model(model_name, X, y, fit_params)
            results[model_name] = {
                "best_params": best_params,
                "best_score": best_score,
                "model": optimized_model
            }
            
        # Find overall best model
        overall_best_score = -np.inf if self.maximize_metric else np.inf
        overall_best_model_name = None
        
        for name, result in results.items():
            score = result["best_score"]
            if score is not None:
                if (self.maximize_metric and score > overall_best_score) or \
                   (not self.maximize_metric and score < overall_best_score):
                    overall_best_score = score
                    overall_best_model_name = name
        
        if overall_best_model_name:
            logger.info(f"\n--- Overall Best Tuned Model ---")
            logger.info(f"Model: {overall_best_model_name}")
            logger.info(f"{self.metric}: {overall_best_score:.4f}")
            self.best_model_ = self.tuned_models[overall_best_model_name]
        
        return results

    def get_tuning_results(self):
        """Return DataFrame with tuning results for all models."""
        if not self.study_objects:
            logger.info("No tuning results available.")
            return pd.DataFrame()
            
        results_data = []
        for name, study in self.study_objects.items():
            try:
                best_value = study.best_value
                best_params = self.best_params.get(name, {})
                
                entry = {
                    "model_name": name,
                    "best_score": best_value if self.maximize_metric else -best_value,
                    "best_params": str(best_params),
                    "n_trials": len(study.trials),
                    "status": "Success"
                }
                results_data.append(entry)
            except Exception as e:
                results_data.append({
                    "model_name": name,
                    "best_score": None,
                    "best_params": None,
                    "n_trials": len(study.trials) if hasattr(study, 'trials') else 0,
                    "status": f"Failed: {str(e)}"
                })
                
        results_df = pd.DataFrame(results_data)
        
        if not results_df.empty and "best_score" in results_df.columns:
            # Sort by best_score
            results_df = results_df.sort_values(
                by="best_score",
                ascending=not self.maximize_metric,
                na_position='last'
            )
        return results_df.reset_index(drop=True)
    
    def plot_optimization_history(self, model_name):
        """Plot optimization history for a specific model."""
        if model_name not in self.study_objects:
            logger.warning(f"No study object found for {model_name}")
            return
            
        study = self.study_objects[model_name]
        try:
            from optuna.visualization import plot_optimization_history
            fig = plot_optimization_history(study)
            fig.update_layout(title=f"Optimization History for {model_name}")
            fig.show()
        except ImportError:
            logger.warning("Plotly not installed. Cannot generate visualization.")
            
    def plot_param_importances(self, model_name):
        """Plot parameter importances for a specific model."""
        if model_name not in self.study_objects:
            logger.warning(f"No study object found for {model_name}")
            return
            
        study = self.study_objects[model_name]
        try:
            from optuna.visualization import plot_param_importances
            fig = plot_param_importances(study)
            fig.update_layout(title=f"Parameter Importances for {model_name}")
            fig.show()
        except ImportError:
            logger.warning("Plotly not installed. Cannot generate visualization.")
    
    def fit_best_model(self, X, y, fit_params=None):
        """Fit the best tuned model on the entire dataset."""
        if not hasattr(self, 'best_model_') or self.best_model_ is None:
            logger.warning("No best model found. Use tune_best_model() or tune_all_models() first.")
            return None
            
        if fit_params is None:
            fit_params = {}
            
        best_model = self.best_model_
        # Find model name for preprocessing logic
        best_model_name = None
        for name, model in self.tuned_models.items():
            if model == best_model:
                best_model_name = name
                break
        
        if best_model_name is None:
            logger.warning("Could not identify best model name for preprocessing")
            return best_model.fit(X, y, **fit_params)
            
        # Apply appropriate preprocessing
        model_type = self.baseline_model._get_model_type(best_model_name)
        if model_type == 'catboost_internal':
            # Handle CatBoost cat_features
            if isinstance(X, pd.DataFrame):
                cat_features_indices = [
                    X.columns.get_loc(col)
                    for col in X.select_dtypes(include=['object', 'category']).columns
                    if col in X.columns
                ]
                if cat_features_indices:
                    fit_params["cat_features"] = cat_features_indices
            return best_model.fit(X, y, **fit_params)
        else:
            # Other models need preprocessing
            preprocessor = self.baseline_model._build_preprocessor(X, model_type)
            if preprocessor:
                from sklearn.pipeline import Pipeline
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', best_model)
                ])
                return pipeline.fit(X, y, **fit_params)
            else:
                return best_model.fit(X, y, **fit_params)