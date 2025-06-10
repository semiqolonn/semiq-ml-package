import pandas as pd
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Literal, TypeVar, cast

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

# Import boosting libraries
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# Import metrics
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    log_loss as sk_log_loss,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

# Setup Logging
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

    def __init__(
            self, 
            task_type: Literal["classification", "regression"] = "classification",
            metric: Optional[str] = None, 
            random_state: int = 42, 
            models: Literal["all", "trees", "gbm"] = "trees"
        ) -> None:
        """
        Initializes the BaselineModel instance.
    
        Args:
            task_type: 'classification' or 'regression'.
            metric: The evaluation metric to optimize for.
                    For classification: 'accuracy', 'f1_weighted', 'roc_auc', 'precision_weighted', 'recall_weighted', 'log_loss'.
                    For regression: 'neg_root_mean_squared_error', 'r2', 'neg_mean_absolute_error'.
                    If None, defaults to 'accuracy' for classification and 'neg_root_mean_squared_error' for regression.
            random_state: Seed for reproducibility.
            models: Which model set to use: 'all', 'trees' (includes Decision Tree, Random Forest, LGBM, XGBoost, CatBoost), 
                   or 'gbm' (only gradient boosting models - LGBM, XGBoost, CatBoost). Default is 'trees'.
        """
        if task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be 'classification' or 'regression'.")
        if models not in ["all", "trees", "gbm"]:
            raise ValueError("models must be one of: 'all', 'trees', 'gbm'.")
        
        self.task_type = task_type
        self.random_state = random_state
        self.results = {}
        self.best_model_ = None
        self.best_score_ = -np.inf
        self.preprocessors_ = {}
        self.models_selection = models
        
        # Set the metric first so it's available during model initialization
        self._set_metric_and_direction(metric)
        
        # Initialize models after the metric is set
        self.models_to_run = self._initialize_models()

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
            ),
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
            "neg_root_mean_squared_error": lambda y_true, y_pred, **kwargs: -np.sqrt(
                mean_squared_error(y_true, y_pred)
            ),
            "r2": r2_score,
            "neg_mean_absolute_error": lambda y_true, y_pred, **kwargs: -mean_absolute_error(
                y_true, y_pred
            ),
        }

    def _set_metric_and_direction(self, metric: Optional[str]) -> None:
        """Set the metric and determine if it should be maximized or minimized"""
        self.maximize_metric: bool = True
        default_metrics: Dict[str, str] = {
            "classification": "accuracy",
            "regression": "neg_root_mean_squared_error"
        }
        valid_metrics: Dict[str, List[str]] = {
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
        elif self.metric in ["r2", "accuracy", "roc_auc", "auc", "f1_weighted", "precision_weighted", "recall_weighted"]:
            self.maximize_metric = True
        else:
            self.maximize_metric = False

    def _get_boosting_params(self) -> Dict[str, Dict[str, Any]]:
        """Gets appropriate parameters for boosting models based on task type and class count"""
        is_multiclass: bool = hasattr(self, 'n_classes_') and self.n_classes_ > 2
        
        if self.task_type == "classification":
            return {
                'xgb': {
                    'objective': "multi:softprob" if is_multiclass else "binary:logistic",
                    'eval_metric': "mlogloss" if is_multiclass else "logloss"
                },
                'lgbm': {
                    'metric': "multi_error" if is_multiclass else "binary_error" if self.metric == "accuracy" else \
                             "multi_logloss" if is_multiclass else "binary_logloss"
                },
                'catboost': {
                    'loss_function': {
                        "accuracy": "Logloss",
                        "log_loss": "Logloss",
                        "roc_auc": "AUC",
                        "f1_weighted": "F1",
                        "precision_weighted": "Precision",
                        "recall_weighted": "Recall",
                    }.get(self.metric, "Logloss")
                }
            }
        else:  # Regression
            return {
                'xgb': {
                    'objective': "reg:squarederror",
                    'eval_metric': {
                        "neg_root_mean_squared_error": "rmse",
                        "neg_mean_absolute_error": "mae",
                        "r2": "rmse",
                    }.get(self.metric, "rmse")
                },
                'lgbm': {
                    'metric': {
                        "neg_root_mean_squared_error": "rmse",
                        "neg_mean_absolute_error": "mae",
                        "r2": "rmse",
                    }.get(self.metric, "rmse")
                },
                'catboost': {
                    'loss_function': {
                        "neg_root_mean_squared_error": "RMSE",
                        "neg_mean_absolute_error": "MAE",
                        "r2": "RMSE",
                    }.get(self.metric, "RMSE")
                }
            }

    def _get_model_type(self, model_name: str) -> Literal['distance_kernel', 'catboost_internal', 'general_ohe']:
        """Determines the preprocessing strategy for a given model."""
        model_name_lower: str = model_name.lower()
        if "knn" in model_name_lower or "svc" in model_name_lower or "svr" in model_name_lower:
            return 'distance_kernel'
        elif "catboost" in model_name_lower:
            return 'catboost_internal'
        else:
            return 'general_ohe'

    def _build_preprocessor(
            self, 
            X_ref_for_dtypes: Union[pd.DataFrame, np.ndarray], 
            preprocessor_type: Literal['general_ohe', 'distance_kernel', 'catboost_internal']
        ) -> Optional[Union[Pipeline, ColumnTransformer]]:
        """Builds a preprocessor based on the required type and data types of X_ref."""
        if not isinstance(X_ref_for_dtypes, pd.DataFrame):
            logger.warning("X_ref_for_dtypes is not a DataFrame. Cannot infer column types for preprocessor. Assuming all numeric.")
            if preprocessor_type == 'catboost_internal':
                return None
            return Pipeline([('scaler', StandardScaler(with_mean=not X_ref_for_dtypes.ndim==2 or X_ref_for_dtypes.shape[1]==0))])

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
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                ('scaler', StandardScaler())
            ])
        else:
            return None

        transformers = []
        if numeric_cols:
            transformers.append(("num", numeric_transformer, numeric_cols))
        if categorical_cols:
            transformers.append(("cat", categorical_transformer, categorical_cols))

        if not transformers:
            logger.warning("No numeric or categorical columns identified to build preprocessor.")
            return None

        return ColumnTransformer(transformers, remainder='drop')

    def _initialize_models(self) -> Dict[str, Any]:
        """Initializes models based on the selected model group."""
        boosting_params: Dict[str, Dict[str, Any]] = self._get_boosting_params()
        is_multiclass: bool = hasattr(self, 'n_classes_') and self.n_classes_ > 2
        
        if self.task_type == "classification":
            all_models = {
                "Logistic Regression": LogisticRegression(random_state=self.random_state, solver="liblinear", max_iter=1000),
                "SVC": SVC(random_state=self.random_state, probability=True),
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
                "Random Forest": RandomForestClassifier(random_state=self.random_state),
                "LGBM": LGBMClassifier(
                    random_state=self.random_state, 
                    verbosity=-1,
                    metric=boosting_params['lgbm']['metric'] if hasattr(self, 'n_classes_') else None
                ),
                "XGBoost": XGBClassifier(
                    random_state=self.random_state,
                    eval_metric=boosting_params['xgb']['eval_metric'],
                    objective=boosting_params['xgb']['objective'],
                    **({"num_class": self.n_classes_} if is_multiclass else {})
                ),
                "CatBoost": CatBoostClassifier(
                    random_state=self.random_state, 
                    silent=True,
                    loss_function=boosting_params['catboost']['loss_function'] if hasattr(self, 'n_classes_') else None
                )
            }
        else:  # Regression
            all_models = {
                "Linear Regression": LinearRegression(),
                "SVR": SVR(),
                "KNN": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(random_state=self.random_state),
                "Random Forest": RandomForestRegressor(random_state=self.random_state),
                "LGBM": LGBMRegressor(
                    random_state=self.random_state, 
                    verbosity=-1,
                    metric=boosting_params['lgbm']['metric']
                ),
                "XGBoost": XGBRegressor(
                    random_state=self.random_state,
                    eval_metric=boosting_params['xgb']['eval_metric'],
                    objective=boosting_params['xgb']['objective']
                ),
                "CatBoost": CatBoostRegressor(
                    random_state=self.random_state, 
                    silent=True,
                    loss_function=boosting_params['catboost']['loss_function']
                )
            }
        
        # Filter models based on the selected group
        if self.models_selection == "all":
            return all_models
        elif self.models_selection == "trees":
            tree_models = ["Decision Tree", "Random Forest", "LGBM", "XGBoost", "CatBoost"]
            return {name: model for name, model in all_models.items() if name in tree_models}
        elif self.models_selection == "gbm":
            gbm_models = ["LGBM", "XGBoost", "CatBoost"]
            return {name: model for name, model in all_models.items() if name in gbm_models}

    def _evaluate_model_score(
            self, 
            model: Any, 
            X_val: Union[pd.DataFrame, np.ndarray], 
            y_val: Union[pd.Series, np.ndarray]
        ) -> float:
        """Calculates the score for a given model and primary metric."""
        metric_fn: Callable = self._metric_functions[self.metric]
        
        # Handle label encoding if needed
        original_y_val = y_val
        if hasattr(self, 'label_encoder_') and self.label_encoder_ is not None:
            # If we used label encoding during training, ensure y_val matches the encoded format
            if not np.issubdtype(np.array(y_val).dtype, np.number):
                y_val = self.label_encoder_.transform(y_val)
        
        if self.task_type == "classification":
            if self.metric in ("roc_auc", "auc", "log_loss"):
                if hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_val)
                    kwargs = {}
                    # For roc_auc, handle multiclass properly
                    if self.metric in ("roc_auc", "auc") and y_pred_proba.shape[1] > 2: # Multiclass
                        kwargs = {
                            "multi_class": "ovr", 
                            "average": "weighted", 
                            "labels": getattr(model, 'classes_', np.unique(y_val))
                        }
                    elif self.metric in ("roc_auc", "auc"): # Binary
                        y_pred_proba = y_pred_proba[:, 1]
    
                    # For log_loss, ensure labels are passed if needed
                    if self.metric == "log_loss":
                        kwargs["labels"] = getattr(model, 'classes_', np.unique(y_val))
                    
                    return metric_fn(y_val, y_pred_proba, **kwargs)
                else:
                    logger.warning(f"Model {model.__class__.__name__} does not support predict_proba. Using accuracy instead.")
                    y_pred = model.predict(X_val)
                    return accuracy_score(y_val, y_pred)
            else: # accuracy, f1, precision, recall
                y_pred = model.predict(X_val)
                
                # Handle prediction format for models with string classes
                if hasattr(self, 'label_encoder_') and self.label_encoder_ is not None:
                    # Check if model outputs encoded or decoded predictions
                    if hasattr(model, 'classes_') and not isinstance(model.classes_[0], (int, np.integer)):
                        # Model produces string predictions, but we're comparing with encoded y_val
                        y_pred = self.label_encoder_.transform(y_pred)
                        
                return metric_fn(y_val, y_pred)
        else:  # Regression
            y_pred = model.predict(X_val)
            return metric_fn(y_val, y_pred)

    def _preprocess_for_model(
            self, 
            model_name: str, 
            X_train_raw: Union[pd.DataFrame, np.ndarray], 
            X_val_raw: Optional[Union[pd.DataFrame, np.ndarray]] = None
        ) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
        """Preprocesses data for a specific model type"""
        preprocessor_key: str = self._get_model_type(model_name)
        
        if preprocessor_key == 'catboost_internal':
            return X_train_raw, X_val_raw, preprocessor_key
        
        if preprocessor_key not in self.preprocessors_:
            preprocessor_instance = self._build_preprocessor(X_train_raw, preprocessor_key)
            if preprocessor_instance is not None:
                self.preprocessors_[preprocessor_key] = preprocessor_instance.fit(X_train_raw)
            else:
                self.preprocessors_[preprocessor_key] = None
        
        fitted_preprocessor = self.preprocessors_.get(preprocessor_key)
        if fitted_preprocessor:
            X_train = fitted_preprocessor.transform(X_train_raw)
            X_val = fitted_preprocessor.transform(X_val_raw) if X_val_raw is not None else None
        else:
            X_train = X_train_raw.to_numpy() if isinstance(X_train_raw, pd.DataFrame) else X_train_raw
            X_val = X_val_raw.to_numpy() if X_val_raw is not None and isinstance(X_val_raw, pd.DataFrame) else X_val_raw
        
        return X_train, X_val, preprocessor_key

    def fit(
            self, 
            X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray], 
            validation_size: float = 0.2, 
            random_state: Optional[int] = None, 
            **kwargs: Any
        ) -> Dict[str, Dict[str, Any]]:
        """
        Fit the model(s) to the data.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features
        y : pandas.Series or numpy.ndarray
            Target
        validation_size : float, default=0.2
            Size of validation set
        random_state : int, optional
            Random state for reproducibility
        **kwargs : dict
            Additional parameters to pass to the model
            
        Returns
        -------
        dict
            Dictionary with results for each model
        """
        self.results = {}
        self.preprocessors_ = {}
        self._fitted_preprocessed_data_cache = {'train': {}, 'val': {}}
        
        # Handle string class labels for classification tasks
        self.label_encoder_ = None
        if self.task_type == "classification" and not np.issubdtype(np.array(y).dtype, np.number):
            logger.info("Target labels are non-numeric. Applying label encoding.")
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder_ = LabelEncoder()
            y = self.label_encoder_.fit_transform(y)
        
        self.n_classes_ = len(np.unique(y)) if self.task_type == "classification" else None
        
        # Update XGBoost config for multiclass if needed
        if self.task_type == "classification" and self.n_classes_ > 2:
            if "XGBoost" in self.models_to_run:
                params = self._get_boosting_params()['xgb']
                self.models_to_run["XGBoost"] = XGBClassifier(
                    random_state=self.random_state,
                    eval_metric=params['eval_metric'],
                    objective=params['objective'],
                    num_class=self.n_classes_
                )
    
        # Store original X for CatBoost if needed
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
                self.original_X_for_catboost, y, test_size=validation_size, 
                random_state=self.random_state, stratify=stratify_opt
            )

        self.best_model_ = None
        self.best_score_ = -np.inf if self.maximize_metric else np.inf

        logger.info(f"Starting BaselineModel training for {self.task_type} with metric: {self.metric} (Maximize: {self.maximize_metric})")
        logger.info(f"Validation set size: {validation_size * 100:.0f}%")

        for name, model_instance in self.models_to_run.items():
            start_time = time.time()
            model_specific_fit_params = kwargs.copy()
            
            if name == "CatBoost" and self.original_X_for_catboost is not None:
                current_X_train, current_X_val = original_X_train_catboost, original_X_val_catboost
                if isinstance(current_X_train, pd.DataFrame):
                    cat_features_indices = [
                        current_X_train.columns.get_loc(col) 
                        for col in current_X_train.select_dtypes(include=['object', 'category']).columns 
                        if col in current_X_train.columns
                    ]
                    if cat_features_indices:
                        model_specific_fit_params["cat_features"] = cat_features_indices
                if "eval_set" not in model_specific_fit_params and current_X_val is not None:
                    model_specific_fit_params["eval_set"] = [(current_X_val, y_val)]
                preprocessor_key = 'catboost_internal'
            else:
                preprocessor_key = self._get_model_type(name)
                if preprocessor_key in self._fitted_preprocessed_data_cache['train']:
                    current_X_train = self._fitted_preprocessed_data_cache['train'][preprocessor_key]
                    current_X_val = self._fitted_preprocessed_data_cache['val'][preprocessor_key]
                else:
                    current_X_train, current_X_val, preprocessor_key = self._preprocess_for_model(
                        name, X_train_raw, X_val_raw
                    )
                    self._fitted_preprocessed_data_cache['train'][preprocessor_key] = current_X_train
                    self._fitted_preprocessed_data_cache['val'][preprocessor_key] = current_X_val
                
                # Set eval_set for boosting models
                if isinstance(model_instance, (LGBMClassifier, LGBMRegressor, XGBClassifier, XGBRegressor)):
                    if "eval_set" not in model_specific_fit_params and current_X_val is not None:
                        model_specific_fit_params["eval_set"] = [(current_X_val, y_val)]
                    if isinstance(model_instance, (XGBClassifier, XGBRegressor)) and "verbose" not in model_specific_fit_params:
                        model_specific_fit_params["verbose"] = False

            if current_X_train is None:
                logger.error(f"Training data for {name} is None. Skipping.")
                self.results[name] = {
                    "model": None, 
                    "score": None, 
                    "time": 0, 
                    "error": "Training data is None", 
                    "preprocessor_used": preprocessor_key,
                    "status": "Failed",
                    "error_message": "Training data is None",
                    "train_score": None,
                    "val_score": None,
                    "fit_time": 0,
                    "preprocessor": preprocessor_key
                }
                continue
            
            try:
                model_instance.fit(current_X_train, y_train, **model_specific_fit_params)
                
                # Calculate training score
                train_score = self._evaluate_model_score(model_instance, current_X_train, y_train)
                
                # Calculate validation score
                val_score = self._evaluate_model_score(model_instance, current_X_val, y_val)
                
                elapsed_time = time.time() - start_time

                logger.info(f"  {name} {self.metric}: train={train_score:.4f}, val={val_score:.4f} (Time: {elapsed_time:.2f}s)")
                self.results[name] = {
                    "model": model_instance, 
                    "score": val_score,  # Keeping for backward compatibility
                    "time": elapsed_time,
                    "preprocessor_used": preprocessor_key,
                    "status": "Success",
                    "error_message": "",
                    "train_score": train_score,
                    "val_score": val_score,
                    "fit_time": elapsed_time,
                    "preprocessor": preprocessor_key
                }

                if ((self.maximize_metric and val_score > self.best_score_) or 
                   (not self.maximize_metric and val_score < self.best_score_)):
                    self.best_score_ = val_score
                    self.best_model_ = model_instance
                    logger.info(f"  --> NEW BEST model: {name} with {self.metric}: {val_score:.4f}")

            except Exception as e:
                logger.error(f"Error training {name}: {e}", exc_info=True)
                self.results[name] = {
                    "model": None, 
                    "score": None, 
                    "time": time.time() - start_time, 
                    "error": str(e),
                    "preprocessor_used": preprocessor_key,
                    "status": "Failed",
                    "error_message": str(e),
                    "train_score": None,
                    "val_score": None,
                    "fit_time": time.time() - start_time,
                    "preprocessor": preprocessor_key
                }
    
        del self._fitted_preprocessed_data_cache

        if not self.best_model_:
            logger.error("No models were successfully trained.")
            return self.results  # Return results even if empty
        
        logger.info(f"BaselineModel run complete. Best model: {self.best_model_.__class__.__name__} with {self.metric}: {self.best_score_:.4f}")
        return self.results  # Return the results dictionary

    def _generate_evaluation_dataframe(
            self, 
            X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray]
        ) -> pd.DataFrame:
        """
        Generate a DataFrame with evaluation metrics for all models.
    
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features
        y : pandas.Series or numpy.ndarray
            Target
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with evaluation metrics for each model
        """
        if not hasattr(self, 'results') or not self.results:
            raise ValueError("No fitted models available. Call fit() first.")
        
        # Dictionary to store evaluation results
        eval_data = []
    
        # Handle label encoding for classification tasks
        if self.task_type == "classification" and hasattr(self, 'label_encoder_') and self.label_encoder_ is not None:
            if not np.issubdtype(np.array(y).dtype, np.number):
                y = self.label_encoder_.transform(y)
    
        # Define metrics to calculate based on task type
        metrics = {}
        if self.task_type == 'classification':
            metrics = {
                'accuracy': lambda y_true, y_pred, **kwargs: accuracy_score(y_true, y_pred),
                'f1_weighted': lambda y_true, y_pred, **kwargs: f1_score(y_true, y_pred, average='weighted', zero_division=0),
                'precision_weighted': lambda y_true, y_pred, **kwargs: precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall_weighted': lambda y_true, y_pred, **kwargs: recall_score(y_true, y_pred, average='weighted', zero_division=0),
            }
        
            # Add AUC and log_loss for models that support predict_proba
            metrics.update({
                'roc_auc': lambda y_true, y_pred_proba, **kwargs: roc_auc_score(
                    y_true, y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba, 
                    multi_class='ovr' if y_pred_proba.shape[1] > 2 else 'raise',
                    average='weighted' if y_pred_proba.shape[1] > 2 else None
                ),
                'log_loss': lambda y_true, y_pred_proba, **kwargs: sk_log_loss(y_true, y_pred_proba)
            })
        else:  # regression metrics
            metrics = {
                'r2': lambda y_true, y_pred, **kwargs: r2_score(y_true, y_pred),
                'rmse': lambda y_true, y_pred, **kwargs: np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': lambda y_true, y_pred, **kwargs: mean_absolute_error(y_true, y_pred)
            }
    
        # Calculate metrics for each model
        for name, model_info in self.results.items():
            model = model_info.get('model')
            if model is None:
                continue
            try:
                # Get preprocessed X for this model
                X_processed = self._get_processed_data_for_eval(X, name)
                
                # Get predictions and probabilities (if applicable)
                y_pred = model.predict(X_processed)
                y_pred_proba = None
                if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                    try:
                        y_pred_proba = model.predict_proba(X_processed)
                    except:
                        # Some models might not support predict_proba for some configs
                        pass
            
                # Calculate metrics
                model_metrics = {'model': name}
                for metric_name, metric_fn in metrics.items():
                    try:
                        if metric_name in ['roc_auc', 'log_loss'] and y_pred_proba is not None:
                            model_metrics[metric_name] = metric_fn(y, y_pred_proba)
                        else:
                            model_metrics[metric_name] = metric_fn(y, y_pred)
                    except Exception as e:
                        logger.warning(f"Could not calculate {metric_name} for {name}: {e}")
                        model_metrics[metric_name] = None
                
                eval_data.append(model_metrics)
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}", exc_info=True)
    
        if not eval_data:
            logger.warning("No models could be evaluated.")
            return pd.DataFrame()
        
        return pd.DataFrame(eval_data)

    def evaluate_all(
            self, 
            X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray]
        ) -> Dict[str, float]:
        """
        Evaluate all fitted models on new data.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features
        y : pandas.Series or numpy.ndarray
            Target
            
        Returns
        -------
        dict
            Dictionary with model names as keys and their primary metric scores as values
        """
        if not hasattr(self, 'results') or not self.results:
            raise ValueError("No fitted models available. Call fit() first.")
    
        # Generate evaluation DataFrame
        eval_df = self._generate_evaluation_dataframe(X, y)
    
        # Convert DataFrame to dictionary, using only the primary metric as the score
        eval_dict = {}
        for _, row in eval_df.iterrows():
            model_name = row['model']
            # Use the metric that the model was trained to optimize
            if self.metric in row and row[self.metric] is not None:
                eval_dict[model_name] = row[self.metric]
            # Fallback to accuracy for classification or r2 for regression
            elif self.task_type == 'classification' and 'accuracy' in row:
                eval_dict[model_name] = row['accuracy']
            elif self.task_type == 'regression' and 'r2' in row:
                eval_dict[model_name] = row['r2']
            # Last resort: use the first available metric
            else:
                for col in eval_df.columns:
                    if col != 'model' and row[col] is not None:
                        eval_dict[model_name] = row[col]
                        break
    
        return eval_dict

    def get_model(self, model_name: str) -> Any:
        """
        Get a specific model by name.
        
        Parameters
        ----------
        model_name : str
            The name of the model to retrieve
            
        Returns
        -------
        object
            The fitted model instance
            
        Raises
        ------
        ValueError
            If the model is not found or not successfully trained
        """
        if not hasattr(self, 'results') or not self.results:
            raise ValueError("No fitted models available. Call fit() first.")
            
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.results.keys())}")
            
        model = self.results[model_name].get('model')
        if model is None:
            raise ValueError(f"Model '{model_name}' failed to train successfully.")
            
        return model
        
    def get_results(self) -> pd.DataFrame:
        """
        Get results of model fitting as a DataFrame.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with model results
        """
        if not hasattr(self, 'results') or not self.results:
            raise ValueError("No results available. Call fit() first.")
    
        # Extract information from results dictionary
        data = []
        for model_name, model_info in self.results.items():
            entry = {
                'model': model_name,
                'train_score': model_info.get('train_score', None),
                'val_score': model_info.get('val_score', None),
                'fit_time': model_info.get('fit_time', model_info.get('time', None)),
                'preprocessor_used': model_info.get('preprocessor', model_info.get('preprocessor_used', 'unknown')),
                'status': model_info.get('status', 'Success' if model_info.get('model') is not None else 'Failed'),
                'error_message': model_info.get('error_message', model_info.get('error', ''))
            }
            data.append(entry)
    
        # Create and return DataFrame
        return pd.DataFrame(data)
        
    def _get_processed_data_for_eval(
            self, 
            X: Union[pd.DataFrame, np.ndarray], 
            model_name: str
        ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Process input data using the appropriate preprocessor for a specific model.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features to preprocess
        model_name : str
            Name of the model for which to preprocess the data
            
        Returns
        -------
        numpy.ndarray
            Preprocessed features
        """
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.results.keys())}")
            
        model_info = self.results[model_name]
        preprocessor_key = model_info.get('preprocessor', model_info.get('preprocessor_used'))
        
        if preprocessor_key == 'catboost_internal':
            # CatBoost handles preprocessing internally
            return X
        
        preprocessor = self.preprocessors_.get(preprocessor_key)
        if preprocessor is not None:
            return preprocessor.transform(X)
        
        # No preprocessing needed or preprocessor not available
        return X.to_numpy() if isinstance(X, pd.DataFrame) else X

    def _plot_curves(
            self, 
            X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray], 
            curve_type: Literal["roc", "precision_recall"] = "roc"
        ) -> None:
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
                continue
            
            try:
                eval_X_processed = self._get_processed_data_for_eval(X, name)
                y_pred_proba = model.predict_proba(eval_X_processed)[:, 1]

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
                logger.error(f"Error plotting {curve_type.upper()} curve for {name}: {e}")

        if not plotted_anything:
            logger.warning(f"No models available to plot {curve_type.upper()} curves.")
            plt.close()
            return

        if curve_type == "roc":
            plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
        elif curve_type == "precision_recall":
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves')
        
        plt.legend()
        plt.grid(True)
        plt.show()

    def roc_curves(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Plots ROC curves for classification models."""
        self._plot_curves(X, y, curve_type="roc")

    def precision_recall_curves(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Plots Precision-Recall curves for classification models."""
        self._plot_curves(X, y, curve_type="precision_recall")


class Timeseries:
    """
    Placeholder for a future class to handle timeseries-specific baseline models.
    Currently, this class does not implement any functionality.
    """
    def __init__(self) -> None:
        raise NotImplementedError("TimeseriesBaseline is not yet implemented.")