import pandas as pd
import numpy as np
import time
import logging
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    log_loss as sk_log_loss, # Renamed to avoid conflict
    roc_curve,
    precision_recall_curve,
    average_precision_score
)


# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BaselineModel:
    """
    A class to automate the training and basic evaluation of multiple baseline
    machine learning models for either classification or regression tasks,
    with model-specific preprocessing considerations.
    """

    def __init__(self, task_type="classification", metric=None, random_state=42):
        """
        Initializes the BaselineModel instance.

        Args:
            task_type (str): 'classification' or 'regression'.
            metric (str, optional): The evaluation metric to optimize for.
                                    For classification: 'accuracy', 'f1_weighted', 'roc_auc', 'precision_weighted', 'recall_weighted', 'log_loss'.
                                    For regression: 'neg_root_mean_squared_error', 'r2', 'neg_mean_absolute_error'.
                                    If None, defaults to 'accuracy' for classification and 'neg_root_mean_squared_error' for regression.
            random_state (int): Seed for reproducibility.
        """
        if task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be 'classification' or 'regression'.")
        self.task_type = task_type
        self.random_state = random_state
        self.results = {}
        self.best_model_ = None
        self.best_score_ = -np.inf
        self.preprocessors_ = {}  # To store fitted preprocessors
        self.models_to_run = self._initialize_models()

        self._set_metric_and_direction(metric)

        self._metric_functions = {
            "accuracy": accuracy_score,
            "f1_weighted": lambda y_true, y_pred, **kwargs: f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "roc_auc": lambda y_true, y_pred_proba, **kwargs: roc_auc_score(
                y_true, y_pred_proba, 
                multi_class=kwargs.get('multi_class', 'raise'),
                average=kwargs.get('average', 'macro')
            ),
            "auc": lambda y_true, y_pred_proba, **kwargs: roc_auc_score(
                y_true, y_pred_proba,
                multi_class=kwargs.get('multi_class', 'raise'), 
                average=kwargs.get('average', 'macro')
            ),  # Alias
            "precision_weighted": lambda y_true, y_pred, **kwargs: precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall_weighted": lambda y_true, y_pred, **kwargs: recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "log_loss": lambda y_true, y_pred_proba, **kwargs: sk_log_loss(
                y_true, y_pred_proba, 
                labels=kwargs.get('labels', None)
            ),
            # Keep other metrics as they are
            "neg_root_mean_squared_error": lambda y_true, y_pred, **kwargs: -np.sqrt(
                mean_squared_error(y_true, y_pred)
            ),
            "r2": lambda y_true, y_pred, **kwargs: r2_score(y_true, y_pred),
            "neg_mean_absolute_error": lambda y_true, y_pred, **kwargs: -mean_absolute_error(
                y_true, y_pred
            ),
        }

    def _set_metric_and_direction(self, metric):
        self.maximize_metric = True
        default_metrics = {
            "classification": "accuracy",
            "regression": "neg_root_mean_squared_error"
        }
        valid_metrics = {
            "classification": ["accuracy", "f1_weighted", "roc_auc", "auc", "precision_weighted", "recall_weighted", "log_loss"],
            "regression": ["neg_root_mean_squared_error", "r2", "neg_mean_absolute_error"]
        }

        self.metric = metric if metric else default_metrics[self.task_type]

        if self.metric not in valid_metrics[self.task_type]:
            raise ValueError(
                f"Invalid metric '{self.metric}' for {self.task_type}. "
                f"Choose from {valid_metrics[self.task_type]}."
            )

        if self.metric == "log_loss" and self.task_type == "classification":
            self.maximize_metric = False # Log loss is minimized
        elif "neg_" in self.metric:
            self.maximize_metric = True # Already negative, so maximize
        elif self.metric == "r2" or self.metric == "accuracy" or "roc_auc" in self.metric or "f1" in self.metric or "precision" in self.metric or "recall" in self.metric:
            self.maximize_metric = True
        else: # For regression metrics like MSE, MAE (if not negated), they would be minimized
            self.maximize_metric = False


    def _get_model_type(self, model_name):
        """Determines the preprocessing strategy for a given model."""
        model_name_lower = model_name.lower()
        if "knn" in model_name_lower or "svc" in model_name_lower or "svr" in model_name_lower:
            return 'distance_kernel' # Needs OrdinalEncoder + Scaling
        elif "catboost" in model_name_lower:
            return 'catboost_internal' # Uses raw data + cat_features
        else: # Linear, Trees, other Boosting (LGBM, XGBoost)
            return 'general_ohe' # Needs OneHotEncoder + Scaling for numeric

    def _build_preprocessor(self, X_ref_for_dtypes, preprocessor_type):
        """Builds a preprocessor based on the required type and data types of X_ref."""
        if not isinstance(X_ref_for_dtypes, pd.DataFrame):
            logger.warning("X_ref_for_dtypes is not a DataFrame. Cannot infer column types for preprocessor. Assuming all numeric.")
            # Fallback: only scale if not a DataFrame (e.g. numpy array)
            if preprocessor_type == 'catboost_internal': return None # No preprocessor for Catboost
            return Pipeline([('scaler', StandardScaler(with_mean=not X_ref_for_dtypes.ndim==2 or X_ref_for_dtypes.shape[1]==0))]) # Basic scaler for numpy

        categorical_cols = X_ref_for_dtypes.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = X_ref_for_dtypes.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Preprocessor type: {preprocessor_type}")
        logger.info(f"Numeric columns: {numeric_cols}")
        logger.info(f"Categorical columns: {categorical_cols}")

        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        if preprocessor_type == 'general_ohe':
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown="ignore", sparse_output=True))
            ])
        elif preprocessor_type == 'distance_kernel':
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)), # Map unknown to -1
                ('scaler', StandardScaler()) # Scale ordinal features
            ])
        else: # catboost_internal or other unknown
            return None


        transformers = []
        if numeric_cols:
            transformers.append(("num", numeric_transformer, numeric_cols))
        if categorical_cols:
            transformers.append(("cat", categorical_transformer, categorical_cols))

        if not transformers: # No numeric or categorical columns identified
             logger.warning("No numeric or categorical columns identified to build preprocessor.")
             return None # Or a "passthrough" pipeline

        return ColumnTransformer(transformers, remainder='drop')


    def _initialize_models(self):
        """Initializes a dictionary of model instances."""
        if self.task_type == "classification":
            return {
                "Logistic Regression": LogisticRegression(random_state=self.random_state, solver="liblinear", max_iter=1000),
                "SVC": SVC(random_state=self.random_state, probability=True), # Enable probability for all metrics
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
                "Random Forest": RandomForestClassifier(random_state=self.random_state),
                "LGBM": LGBMClassifier(random_state=self.random_state, verbosity=-1),
                "XGBoost": XGBClassifier(random_state=self.random_state, eval_metric="logloss"),
                "CatBoost": CatBoostClassifier(random_state=self.random_state, silent=True),
            }
        else: # Regression
            return {
                "Linear Regression": LinearRegression(),
                "SVR": SVR(),
                "KNN": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(random_state=self.random_state),
                "Random Forest": RandomForestRegressor(random_state=self.random_state),
                "LGBM": LGBMRegressor(random_state=self.random_state, verbosity=-1), # verbosity for LGBMRegressor too
                "XGBoost": XGBRegressor(random_state=self.random_state, eval_metric="rmse"),
                "CatBoost": CatBoostRegressor(random_state=self.random_state, silent=True),
            }

    def _evaluate_model_score(self, model, X_val, y_val):
        """Calculates the score for a given model and primary metric."""
        metric_fn = self._metric_functions[self.metric]
        
        if self.task_type == "classification":
            if self.metric in ("roc_auc", "auc", "log_loss"):
                if hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_val)
                    kwargs = {}
                    # For roc_auc, handle multiclass properly
                    if self.metric in ("roc_auc", "auc") and y_pred_proba.shape[1] > 2: # Multiclass
                        kwargs = {"multi_class": "ovr", "average": "weighted", "labels": getattr(model, 'classes_', np.unique(y_val))}
                    elif self.metric in ("roc_auc", "auc"): # Binary
                         y_pred_proba = y_pred_proba[:, 1]

                    # For log_loss, ensure labels are passed if needed by sk_log_loss
                    if self.metric == "log_loss":
                         kwargs["labels"] = getattr(model, 'classes_', np.unique(y_val))
                    
                    return metric_fn(y_val, y_pred_proba, **kwargs)
                else:
                    logger.warning(
                        f"Model {model.__class__.__name__} does not support predict_proba. "
                        f"Cannot use '{self.metric}'. Falling back to 'accuracy'."
                    )
                    return accuracy_score(y_val, model.predict(X_val)) # Fallback
            else: # accuracy, f1, precision, recall
                y_pred = model.predict(X_val)
                return metric_fn(y_val, y_pred)
        else:  # Regression
            y_pred = model.predict(X_val)
            return metric_fn(y_val, y_pred)


    def fit(self, X, y, validation_size=0.2, **fit_params):
        """
        Trains all initialized models, applying appropriate preprocessing, and evaluates them.
        """
        self.results = {} # Reset results for a new fit
        self.preprocessors_ = {} # Reset fitted preprocessors
        self._fitted_preprocessed_data_cache = {'train': {}, 'val': {}} # Cache for transformed data within this fit

        self.original_X_for_catboost = X.copy() if isinstance(X, pd.DataFrame) else None
        
        # Stratify for classification, not for regression
        stratify_opt = y if self.task_type == "classification" else None

        # Split raw data first
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X, y, test_size=validation_size, random_state=self.random_state, stratify=stratify_opt
        )
        
        original_X_train_catboost, original_X_val_catboost = (None, None)
        if self.original_X_for_catboost is not None:
            original_X_train_catboost, original_X_val_catboost, _, _ = train_test_split(
                self.original_X_for_catboost, y, test_size=validation_size, random_state=self.random_state, stratify=stratify_opt
            )

        self.best_model_ = None
        self.best_score_ = -np.inf if self.maximize_metric else np.inf

        logger.info(f"Starting BaselineModel training for {self.task_type} with metric: {self.metric} (Maximize: {self.maximize_metric})")
        logger.info(f"Validation set size: {validation_size * 100:.0f}%")

        for name, model_instance in self.models_to_run.items():
            start_time = time.time()
            model_specific_fit_params = fit_params.copy()
            
            preprocessor_key = self._get_model_type(name)
            self.results[name] = {'preprocessor_used': preprocessor_key} # Store which preprocessor type

            current_X_train, current_X_val = None, None

            if preprocessor_key == 'catboost_internal':
                current_X_train = original_X_train_catboost
                current_X_val = original_X_val_catboost
                if isinstance(current_X_train, pd.DataFrame):
                    cat_features_indices = [current_X_train.columns.get_loc(col) for col in current_X_train.select_dtypes(include=['object', 'category']).columns if col in current_X_train.columns]
                    if cat_features_indices:
                        model_specific_fit_params["cat_features"] = cat_features_indices
                        logger.info(f"CatBoost: Using categorical feature indices: {cat_features_indices}")
                if "eval_set" not in model_specific_fit_params and current_X_val is not None:
                     model_specific_fit_params["eval_set"] = [(current_X_val, y_val)]

            else: # 'general_ohe' or 'distance_kernel'
                # Check cache for already processed data
                if preprocessor_key in self._fitted_preprocessed_data_cache['train']:
                    current_X_train = self._fitted_preprocessed_data_cache['train'][preprocessor_key]
                    current_X_val = self._fitted_preprocessed_data_cache['val'][preprocessor_key]
                    logger.info(f"Using cached preprocessed data for '{preprocessor_key}' for model {name}")
                else:
                    if preprocessor_key not in self.preprocessors_:
                        # Pass X_train_raw (DataFrame) to infer dtypes for building preprocessor
                        preprocessor_instance = self._build_preprocessor(X_train_raw, preprocessor_key)
                        if preprocessor_instance is not None:
                             self.preprocessors_[preprocessor_key] = preprocessor_instance.fit(X_train_raw) # Fit on X_TRAIN_RAW
                        else: # No preprocessor needed or could be built (e.g. all numeric data for distance)
                             self.preprocessors_[preprocessor_key] = None # Mark as None
                    
                    fitted_preprocessor = self.preprocessors_.get(preprocessor_key)
                    if fitted_preprocessor:
                        current_X_train = fitted_preprocessor.transform(X_train_raw)
                        current_X_val = fitted_preprocessor.transform(X_val_raw)
                    else: # No preprocessor was fit (e.g. data was already all numeric for distance)
                        current_X_train = X_train_raw.to_numpy() if isinstance(X_train_raw, pd.DataFrame) else X_train_raw
                        current_X_val = X_val_raw.to_numpy() if isinstance(X_val_raw, pd.DataFrame) else X_val_raw
                    
                    # Cache the processed data
                    self._fitted_preprocessed_data_cache['train'][preprocessor_key] = current_X_train
                    self._fitted_preprocessed_data_cache['val'][preprocessor_key] = current_X_val
                    logger.info(f"Processed and cached data using '{preprocessor_key}' for model {name}")

                # Common eval_set for other boosting models if X_val is available
                if isinstance(model_instance, (LGBMClassifier, LGBMRegressor, XGBClassifier, XGBRegressor)):
                    if "eval_set" not in model_specific_fit_params and current_X_val is not None:
                        model_specific_fit_params["eval_set"] = [(current_X_val, y_val)]
                    if isinstance(model_instance, (XGBClassifier, XGBRegressor)) and "verbose" not in model_specific_fit_params:
                        model_specific_fit_params["verbose"] = False

            if current_X_train is None:
                logger.error(f"Training data for {name} is None. Skipping.")
                self.results[name].update({"model": None, "score": None, "time": 0, "error": "Training data is None"})
                continue
            
            try:
                model_instance.fit(current_X_train, y_train, **model_specific_fit_params)
                score = self._evaluate_model_score(model_instance, current_X_val, y_val)
                elapsed_time = time.time() - start_time

                logger.info(f"  {name} {self.metric}: {score:.4f} (Training Time: {elapsed_time:.2f}s)")
                self.results[name].update({"model": model_instance, "score": score, "time": elapsed_time})

                if (self.maximize_metric and score > self.best_score_) or \
                   (not self.maximize_metric and score < self.best_score_):
                    self.best_score_ = score
                    self.best_model_ = model_instance
                    logger.info(f"  --> NEW BEST model: {name} with {self.metric}: {score:.4f}")

            except Exception as e:
                logger.error(f"Error training {name}: {e}", exc_info=True)
                self.results[name].update({"model": None, "score": None, "time": time.time() - start_time, "error": str(e)})
        
        del self._fitted_preprocessed_data_cache # Clear cache after fit

        if not self.best_model_:
            logger.error("No models were successfully trained.")
            return None
        logger.info(f"BaselineModel run complete. Best model: {self.best_model_.__class__.__name__} with {self.metric}: {self.best_score_:.4f}")
        return self.best_model_

    def get_model(self, model_name):
        if model_name in self.results and "model" in self.results[model_name] and self.results[model_name]["model"] is not None:
            return self.results[model_name]["model"]
        else:
            raise ValueError(
                f"Model '{model_name}' not found or failed to train. Available models: "
                f"{[k for k, v in self.results.items() if v.get('model') is not None]}"
            )

    def _get_processed_data_for_eval(self, X_raw, model_name_in_results):
        """Transforms raw X data using the preprocessor associated with a trained model."""
        if not self.results or model_name_in_results not in self.results:
            raise ValueError(f"Model {model_name_in_results} not found in results. Run fit() first.")

        preprocessor_key = self.results[model_name_in_results].get('preprocessor_used')

        if preprocessor_key == 'catboost_internal':
            return X_raw # CatBoost uses raw data (expects DataFrame with original categorical types)
        
        fitted_preprocessor = self.preprocessors_.get(preprocessor_key)
        if fitted_preprocessor:
            return fitted_preprocessor.transform(X_raw)
        elif preprocessor_key is None and not fitted_preprocessor : # No preprocessor was needed (e.g. all numeric for distance)
             return X_raw.to_numpy() if isinstance(X_raw, pd.DataFrame) else X_raw
        else:
            raise ValueError(f"Preprocessor '{preprocessor_key}' for model '{model_name_in_results}' not found or not fitted.")


    def evaluate_all(self, X, y):
        """Evaluates all trained models on a given dataset (X, y)."""
        evaluation_metrics_summary = {}
        if not self.results:
            logger.warning("No models have been trained yet. Run .fit() first.")
            return pd.DataFrame()

        for name, model_info in self.results.items():
            model = model_info.get("model")
            if model is None:
                logger.warning(f"Skipping evaluation for {name} as it failed to train.")
                evaluation_metrics_summary[name] = {"error": model_info.get("error", "Failed to train")}
                continue
            
            try:
                # Ensure X is in the correct format (DataFrame for CatBoost if it was trained on one)
                is_catboost = model_info.get('preprocessor_used') == 'catboost_internal'
                eval_X_raw = X
                if is_catboost and not isinstance(X, pd.DataFrame) and isinstance(self.original_X_for_catboost, pd.DataFrame): # self.original_X_for_catboost from fit()
                     logger.warning(f"For CatBoost model '{name}', input X for evaluate_all should ideally be a DataFrame if trained on one. Attempting conversion if possible or using as is.")
                     # This part is tricky - if CatBoost expects specific column names/dtypes from training
                     # it's best if X matches that structure. Here we just pass X.

                processed_eval_X = self._get_processed_data_for_eval(eval_X_raw, name)
                
                current_metrics = {}
                if self.task_type == "classification":
                    preds = model.predict(processed_eval_X)
                    current_metrics["accuracy"] = accuracy_score(y, preds)
                    current_metrics["f1_weighted"] = f1_score(y, preds, average="weighted", zero_division=0)
                    current_metrics["precision_weighted"] = precision_score(y, preds, average="weighted", zero_division=0)
                    current_metrics["recall_weighted"] = recall_score(y, preds, average="weighted", zero_division=0)

                    if hasattr(model, "predict_proba"):
                        y_pred_proba = model.predict_proba(processed_eval_X)
                        labels_for_metrics = getattr(model, 'classes_', np.unique(y))

                        if y_pred_proba.shape[1] == 2: # Binary
                            current_metrics["roc_auc"] = roc_auc_score(y, y_pred_proba[:, 1])
                        else: # Multiclass
                            current_metrics["roc_auc_ovr_weighted"] = roc_auc_score(y, y_pred_proba, multi_class="ovr", average="weighted", labels=labels_for_metrics)
                        
                        try:
                            current_metrics["log_loss"] = sk_log_loss(y, y_pred_proba, labels=labels_for_metrics)
                        except ValueError as e: # e.g. y_true contains labels not in y_pred_proba due to small test set
                            logger.warning(f"Could not compute log_loss for {name} on test data: {e}")
                            current_metrics["log_loss"] = np.nan

                else:  # Regression
                    preds = model.predict(processed_eval_X)
                    current_metrics["mse"] = mean_squared_error(y, preds)
                    current_metrics["mae"] = mean_absolute_error(y, preds)
                    current_metrics["r2"] = r2_score(y, preds)
                    current_metrics["rmse"] = np.sqrt(mean_squared_error(y, preds))
                
                evaluation_metrics_summary[name] = current_metrics
            except Exception as e:
                logger.error(f"Error evaluating {name} on test data: {e}", exc_info=True)
                evaluation_metrics_summary[name] = {"error": str(e)}
        
        results_df = pd.DataFrame.from_dict(evaluation_metrics_summary, orient="index")
        results_df.index.name = "model"
        logger.info("External evaluation complete. Metrics for each model:")
        # logger.info(f"\n{results_df.to_string()}") # Optional: print full df
        return results_df.reset_index()


    def get_results(self):
        """Returns a DataFrame summarizing the primary metric results from the fit process."""
        if not self.results:
            logger.info("No results available. Run .fit() first.")
            return pd.DataFrame()

        results_data = []
        for name, info in self.results.items():
            results_data.append({
                "model": name,
                "score": info.get("score"), # Primary metric score from validation
                "time": info.get("time"),
                "preprocessor_used": info.get("preprocessor_used"),
                "status": "Success" if info.get("model") is not None and "error" not in info else "Failed",
                "error_message": info.get("error", "")
            })
        results_df = pd.DataFrame(results_data)
        if "score" in results_df.columns:
             results_df = results_df.sort_values(by="score", ascending=not self.maximize_metric).reset_index(drop=True)
        return results_df

    def _plot_curves(self, X, y, curve_type="roc"):
        """Helper function to plot ROC or Precision-Recall curves."""
        if self.task_type != "classification":
            raise ValueError(f"{curve_type.upper()} curves only for classification tasks.")
        if not self.results:
            logger.warning("No models trained. Run .fit() first.")
            return

        plt.figure(figsize=(10, 8))
        plotted_anything = False
        for name, model_info in self.results.items():
            model = model_info.get("model")
            if model is None or not hasattr(model, "predict_proba"):
                logger.warning(f"Skipping {curve_type.upper()} curve for {name}: no predict_proba or model failed.")
                continue
            
            try:
                eval_X_processed = self._get_processed_data_for_eval(X, name)
                y_pred_proba = model.predict_proba(eval_X_processed)[:, 1] # Assuming binary or positive class for multiclass

                if curve_type == "roc":
                    fpr, tpr, _ = roc_curve(y, y_pred_proba)
                    auc_score = roc_auc_score(y, y_pred_proba)
                    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")
                elif curve_type == "precision_recall":
                    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
                    ap_score = average_precision_score(y, y_pred_proba)
                    plt.plot(recall, precision, label=f"{name} (AP = {ap_score:.2f})")
                plotted_anything = True
            except Exception as e:
                logger.error(f"Error plotting {curve_type.upper()} curve for {name}: {e}", exc_info=True)

        if not plotted_anything:
            logger.warning(f"No models available to plot {curve_type.upper()} curves.")
            plt.close() # Close the empty figure
            return

        if curve_type == "roc":
            plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
        elif curve_type == "precision_recall":
            # For PR curve, no-skill line depends on class distribution
            # f_score = 2 * (precision * recall) / (precision + recall)
            # no_skill = len(y_true[y_true==1]) / len(y_true)
            # plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves')
        
        plt.legend()
        plt.grid(True)
        plt.show()

    def roc_curves(self, X, y):
        """Plots ROC curves for classification models."""
        self._plot_curves(X, y, curve_type="roc")

    def precision_recall_curves(self, X, y): # Corrected typo
        """Plots Precision-Recall curves for classification models."""
        self._plot_curves(X, y, curve_type="precision_recall")