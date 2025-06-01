# # my_ml_lib/hyperparameter_tuning.py

# import time
# import logging
# import numpy as np
# import pandas as pd
# from scipy.stats import uniform, randint

# from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
# from sklearn.base import clone # To create fresh instances of models

# # Import the BaselineModel
# from .baseline_model import BaselineModel

# # Set up logging for this module
# logger = logging.getLogger(__name__)

# class RandomSearchOptimizer(BaselineModel):
#     """
#     A class that performs RandomizedSearchCV for a set of baseline models,
#     inheriting core functionalities from BaselineModel.
#     """
#     def __init__(self, task_type='classification', metric=None, random_state=42,
#                  n_iter=10, cv=5, n_jobs=-1, param_distributions=None):
#         """
#         Initializes the RandomSearchOptimizer.

#         Args:
#             task_type (str): 'classification' or 'regression'.
#             metric (str, optional): The evaluation metric to optimize for.
#                                     See BaselineModel for valid metrics.
#             random_state (int): Seed for reproducibility.
#             n_iter (int): Number of parameter settings that are sampled.
#                           More iterations lead to a more exhaustive search.
#             cv (int or cross-validation generator): Determines the cross-validation splitting strategy.
#             n_jobs (int): Number of jobs to run in parallel for RandomizedSearchCV.
#                           -1 means using all processors.
#             param_distributions (dict, optional): A dictionary where keys are model names
#                                                   (e.g., 'Logistic Regression') and values are
#                                                   dictionaries of parameter distributions for that model.
#                                                   If None, uses default distributions.
#         """
#         super().__init__(task_type=task_type, metric=metric, random_state=random_state)
#         self.n_iter = n_iter
#         self.cv = cv
#         self.n_jobs = n_jobs
#         self.param_distributions = param_distributions if param_distributions else self._define_default_param_distributions()

#         # Ensure the scoring metric is compatible with sklearn's RandomizedSearchCV
#         self.scoring_metric = self._sklearn_scoring_map.get(self.metric)
#         if not self.scoring_metric:
#             raise ValueError(f"Metric '{self.metric}' is not directly supported by sklearn's scoring parameter. "
#                              "Please choose a compatible metric or provide a custom scorer.")

#         logger.info(f"Initialized RandomSearchOptimizer for {self.task_type} with {self.metric} as scoring metric.")
#         logger.info(f"Random search iterations per model: {self.n_iter}, CV folds: {self.cv}, Parallel jobs: {self.n_jobs}")


#     def _define_default_param_distributions(self):
#         """
#         Defines default hyperparameter distributions for various models.
#         These are examples and should be adjusted based on problem complexity and dataset.
#         """
#         if self.task_type == 'classification':
#             return {
#                 'Logistic Regression': {
#                     'C': uniform(loc=0.1, scale=10), # Inverse of regularization strength
#                     'solver': ['liblinear', 'lbfgs'],
#                     'max_iter': randint(100, 1000)
#                 },
#                 'SVC': {
#                     'C': uniform(loc=0.1, scale=10),
#                     'kernel': ['linear', 'rbf'],
#                     'gamma': ['scale', 'auto', uniform(loc=0.001, scale=0.1)]
#                 },
#                 'KNN': {
#                     'n_neighbors': randint(3, 15),
#                     'weights': ['uniform', 'distance'],
#                     'p': [1, 2] # Manhattan (1) or Euclidean (2) distance
#                 },
#                 'Decision Tree': {
#                     'max_depth': randint(3, 20),
#                     'min_samples_split': randint(2, 20),
#                     'criterion': ['gini', 'entropy']
#                 },
#                 'Random Forest': {
#                     'n_estimators': randint(50, 500),
#                     'max_depth': randint(5, 25),
#                     'min_samples_split': randint(2, 10),
#                     'min_samples_leaf': randint(1, 5)
#                 },
#                 'LGBM': {
#                     'n_estimators': randint(100, 1000),
#                     'learning_rate': uniform(loc=0.01, scale=0.1),
#                     'num_leaves': randint(20, 60),
#                     'max_depth': randint(5, 15),
#                     'reg_alpha': uniform(loc=0, scale=1), # L1 regularization
#                     'reg_lambda': uniform(loc=0, scale=1) # L2 regularization
#                 },
#                 'XGBoost': {
#                     'n_estimators': randint(100, 1000),
#                     'learning_rate': uniform(loc=0.01, scale=0.1),
#                     'max_depth': randint(3, 10),
#                     'subsample': uniform(loc=0.6, scale=0.4), # Subsample ratio of the training instance
#                     'colsample_bytree': uniform(loc=0.6, scale=0.4), # Subsample ratio of columns when constructing each tree
#                     'gamma': uniform(loc=0, scale=0.5) # Minimum loss reduction required to make a further partition
#                 },
#                 'CatBoost': {
#                     'iterations': randint(100, 1000),
#                     'learning_rate': uniform(loc=0.01, scale=0.1),
#                     'depth': randint(4, 10),
#                     'l2_leaf_reg': uniform(loc=1, scale=5), # L2 regularization term on weights
#                     'border_count': randint(32, 255) # Number of splits for numerical features
#                 }
#             }
#         else: # Regression
#             return {
#                 'Linear Regression': {}, # No hyperparameters to tune for basic LinearRegression
#                 'SVR': {
#                     'C': uniform(loc=0.1, scale=10),
#                     'kernel': ['linear', 'rbf'],
#                     'gamma': ['scale', 'auto', uniform(loc=0.001, scale=0.1)]
#                 },
#                 'KNN': {
#                     'n_neighbors': randint(3, 15),
#                     'weights': ['uniform', 'distance'],
#                     'p': [1, 2]
#                 },
#                 'Decision Tree': {
#                     'max_depth': randint(3, 20),
#                     'min_samples_split': randint(2, 20),
#                     'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
#                 },
#                 'Random Forest': {
#                     'n_estimators': randint(50, 500),
#                     'max_depth': randint(5, 25),
#                     'min_samples_split': randint(2, 10),
#                     'min_samples_leaf': randint(1, 5)
#                 },
#                 'LGBM': {
#                     'n_estimators': randint(100, 1000),
#                     'learning_rate': uniform(loc=0.01, scale=0.1),
#                     'num_leaves': randint(20, 60),
#                     'max_depth': randint(5, 15),
#                     'reg_alpha': uniform(loc=0, scale=1),
#                     'reg_lambda': uniform(loc=0, scale=1)
#                 },
#                 'XGBoost': {
#                     'n_estimators': randint(100, 1000),
#                     'learning_rate': uniform(loc=0.01, scale=0.1),
#                     'max_depth': randint(3, 10),
#                     'subsample': uniform(loc=0.6, scale=0.4),
#                     'colsample_bytree': uniform(loc=0.6, scale=0.4),
#                     'gamma': uniform(loc=0, scale=0.5)
#                 },
#                 'CatBoost': {
#                     'iterations': randint(100, 1000),
#                     'learning_rate': uniform(loc=0.01, scale=0.1),
#                     'depth': randint(4, 10),
#                     'l2_leaf_reg': uniform(loc=1, scale=5),
#                     'border_count': randint(32, 255)
#                 }
#             }

#     def fit(self, X, y, **fit_params):
#         """
#         Performs RandomizedSearchCV for each model and returns the overall best model.

#         Args:
#             X (pd.DataFrame or np.array): Features for training and validation.
#             y (pd.Series or np.array): Target variable for training and validation.
#             **fit_params: Additional keyword arguments to pass to the model's .fit() method
#                           during the final fit of the best estimator from RandomizedSearchCV.
#                           Note: `eval_set` and similar parameters for boosting models
#                           should be handled within the `RandomizedSearchCV`'s `fit` call
#                           if they are part of the CV process. For simplicity here,
#                           we'll assume `fit_params` are for the final fit after tuning.

#         Returns:
#             object: The best trained model instance found across all random searches.
#         """
#         self.results = {}
#         self.best_model_ = None
#         self.best_score_ = -np.inf if self.maximize_metric else np.inf

#         logger.info(f"Starting RandomSearchOptimizer for {self.task_type} with scoring metric: {self.metric}")
#         logger.info(f"Each model will run {self.n_iter} iterations with {self.cv}-fold CV.")

#         # Determine CV strategy
#         cv_strategy = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state) \
#             if self.task_type == 'classification' else KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

#         for name, model_base_instance in self.models_to_run.items():
#             if name not in self.param_distributions or not self.param_distributions[name]:
#                 logger.info(f"Skipping {name}: No parameter distributions defined.")
#                 continue

#             param_dist = self.param_distributions[name]
#             logger.info(f"Running RandomizedSearchCV for {name}...")
            
#             start_time = time.time()
#             try:
#                 # Clone the base instance to ensure a fresh start for RandomizedSearchCV
#                 # This is important if model_base_instance has been modified elsewhere
#                 model_for_search = clone(model_base_instance)

#                 # Special handling for boosting models' fit parameters during RandomizedSearchCV
#                 # These parameters are passed to the estimator's fit method during each CV fold
#                 # For simplicity, we're passing `eval_set` to the final fit,
#                 # but for proper early stopping during CV, it's often done via callbacks
#                 # or by passing `eval_set` directly to RandomizedSearchCV's `fit` if the estimator supports it.
#                 # Here, we'll rely on the model's internal defaults or the `fit_params` passed to this method.

#                 # CatBoost specific: silent mode via verbosity
#                 if isinstance(model_for_search, (CatBoostClassifier, CatBoostRegressor)):
#                     model_for_search.set_params(silent=True) # Ensure silent during CV
#                 # XGBoost specific: silent mode via verbosity
#                 if isinstance(model_for_search, (XGBClassifier, XGBRegressor)):
#                     model_for_search.set_params(verbose=False) # Ensure silent during CV

#                 random_search = RandomizedSearchCV(
#                     estimator=model_for_search,
#                     param_distributions=param_dist,
#                     n_iter=self.n_iter,
#                     cv=cv_strategy,
#                     scoring=self.scoring_metric,
#                     random_state=self.random_state,
#                     n_jobs=self.n_jobs,
#                     verbose=0, # Suppress RandomizedSearchCV verbosity
#                     return_train_score=False
#                 )

#                 random_search.fit(X, y) # Fit on the full dataset X, y using CV

#                 best_estimator = random_search.best_estimator_
#                 best_params = random_search.best_params_
#                 best_cv_score = random_search.best_score_
                
#                 elapsed_time = time.time() - start_time

#                 logger.info(f"  {name} - Best CV {self.metric}: {best_cv_score:.4f} (Time: {elapsed_time:.2f}s)")
#                 logger.info(f"  {name} - Best Params: {best_params}")

#                 self.results[name] = {
#                     'model': best_estimator,
#                     'score': best_cv_score,
#                     'time': elapsed_time,
#                     'best_params': best_params,
#                     'status': 'Success'
#                 }

#                 # Update overall best model
#                 if (self.maximize_metric and best_cv_score > self.best_score_) or \
#                    (not self.maximize_metric and best_cv_score < self.best_score_):
#                     self.best_score_ = best_cv_score
#                     self.best_model_ = best_estimator
#                     logger.info(f"  --> NEW OVERALL BEST model: {name} with {self.metric}: {best_cv_score:.4f}")

#             except Exception as e:
#                 logger.error(f"Error during RandomizedSearchCV for {name}: {e}")
#                 self.results[name] = {
#                     'model': None,
#                     'score': None,
#                     'time': None,
#                     'best_params': None,
#                     'status': 'Failed',
#                     'error': str(e)
#                 }
        
#         if not self.best_model_:
#             logger.error("No models were successfully tuned or evaluated to determine an overall best model.")
#             return None

#         logger.info(f"RandomSearchOptimizer run complete. Overall best model: {self.best_model_.__class__.__name__} with {self.metric}: {self.best_score_:.4f}")
#         return self.best_model_

#     def get_results(self):
#         """
#         Returns a Pandas DataFrame summarizing the results of the RandomSearchOptimizer run.
#         Includes best parameters for each model.

#         Returns:
#             pd.DataFrame: A DataFrame with model names, primary score, training time, and best parameters.
#         """
#         if not self.results:
#             logger.info("No results available. Run .fit() first.")
#             return pd.DataFrame()

#         results_data = []
#         for name, info in self.results.items():
#             results_data.append({
#                 'model': name,
#                 'score': info['score'],
#                 'time': info['time'],
#                 'best_params': info['best_params'],
#                 'status': info['status'],
#                 'error': info.get('error', '')
#             })
        
#         results_df = pd.DataFrame(results_data)
        
#         # Sort based on the primary metric and its direction
#         if self.maximize_metric:
#             results_df = results_df.sort_values(by='score', ascending=False)
#         else:
#             results_df = results_df.sort_values(by='score', ascending=True)

#         return results_df.reset_index(drop=True)