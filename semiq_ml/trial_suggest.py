model_param_suggestions_classification = {
    "XGBoost": {
        "booster": lambda t: t.suggest_categorical("booster", ["gbtree", "dart", "gblinear"]),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": lambda t: t.suggest_int("max_depth", 3, 15),
        "min_child_weight": lambda t: t.suggest_int("min_child_weight", 1, 10),
        "gamma": lambda t: t.suggest_float("gamma", 0, 10),
        "subsample": lambda t: t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": lambda t: t.suggest_float("colsample_bylevel", 0.5, 1.0),
        "colsample_bynode": lambda t: t.suggest_float("colsample_bynode", 0.5, 1.0),
        "lambda": lambda t: t.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": lambda t: t.suggest_float("alpha", 1e-3, 10.0, log=True),
        "scale_pos_weight": lambda t: t.suggest_float("scale_pos_weight", 0.5, 5.0),
        "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 1000),
        "max_delta_step": lambda t: t.suggest_int("max_delta_step", 0, 10),
        "grow_policy": lambda t: t.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        # dart-specific
        "sample_type": lambda t: t.suggest_categorical("sample_type", ["uniform", "weighted"]),
        "normalize_type": lambda t: t.suggest_categorical("normalize_type", ["tree", "forest"]),
        "rate_drop": lambda t: t.suggest_float("rate_drop", 0.0, 0.3),
        "skip_drop": lambda t: t.suggest_float("skip_drop", 0.0, 0.3),
    },

    "LGBM": {
        "boosting_type": lambda t: t.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": lambda t: t.suggest_int("num_leaves", 15, 256),
        "max_depth": lambda t: t.suggest_int("max_depth", -1, 15),
        "min_data_in_leaf": lambda t: t.suggest_int("min_data_in_leaf", 10, 100),
        "min_sum_hessian_in_leaf": lambda t: t.suggest_float("min_sum_hessian_in_leaf", 1e-3, 10.0, log=True),
        "feature_fraction": lambda t: t.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": lambda t: t.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": lambda t: t.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": lambda t: t.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": lambda t: t.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "min_gain_to_split": lambda t: t.suggest_float("min_gain_to_split", 0.0, 1.0),
        "extra_trees": lambda t: t.suggest_categorical("extra_trees", [True, False]),
        "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 1000),
    },

    "CatBoost": {
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": lambda t: t.suggest_int("depth", 4, 10),
        "l2_leaf_reg": lambda t: t.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "border_count": lambda t: t.suggest_int("border_count", 32, 255),
        "bagging_temperature": lambda t: t.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": lambda t: t.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "rsm": lambda t: t.suggest_float("rsm", 0.5, 1.0),
        "iterations": lambda t: t.suggest_int("iterations", 100, 1000),
    },

    "Decision Tree": {
        "criterion": lambda t: t.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": lambda t: t.suggest_int("max_depth", 1, 32),
        "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 20)
    },

    "Random Forest": {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 500),
        "criterion": lambda t: t.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": lambda t: t.suggest_int("max_depth", 1, 32),
        "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 20),
        "bootstrap": lambda t: t.suggest_categorical("bootstrap", [True, False])
    },

    "SVM": {
        "C": lambda t: t.suggest_float("C", 1e-3, 100, log=True),
        "kernel": lambda t: t.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"]),
        "gamma": lambda t: t.suggest_categorical("gamma", ["scale", "auto"]),
        "degree": lambda t: t.suggest_int("degree", 2, 5)
    },

    "KNNClassifier": {
        "n_neighbors": lambda t: t.suggest_int("n_neighbors", 1, 30),
        "weights": lambda t: t.suggest_categorical("weights", ["uniform", "distance"]),
        "p": lambda t: t.suggest_int("p", 1, 2),
        "leaf_size": lambda t: t.suggest_int("leaf_size", 10, 50)
    },

    "Logistic Regression": {
        "C": lambda t: t.suggest_float("C", 1e-3, 100, log=True),
        "penalty": lambda t: t.suggest_categorical("penalty", ["l1", "l2", "elasticnet", "none"]),
        "solver": lambda t: t.suggest_categorical("solver", ["liblinear", "saga"]),
        "l1_ratio": lambda t: t.suggest_float("l1_ratio", 0.0, 1.0)  # used only if penalty = elasticnet
    }
}

regression_param_suggestions_regression = {
    "XGBoost": {
        "booster": lambda t: t.suggest_categorical("booster", ["gbtree", "dart", "gblinear"]),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": lambda t: t.suggest_int("max_depth", 3, 15),
        "min_child_weight": lambda t: t.suggest_int("min_child_weight", 1, 10),
        "gamma": lambda t: t.suggest_float("gamma", 0, 10),
        "subsample": lambda t: t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": lambda t: t.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": lambda t: t.suggest_float("alpha", 1e-3, 10.0, log=True),
        "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 1000),
        "grow_policy": lambda t: t.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        "rate_drop": lambda t: t.suggest_float("rate_drop", 0.0, 0.3),
        "skip_drop": lambda t: t.suggest_float("skip_drop", 0.0, 0.3),
    },

    "LGBM": {
        "boosting_type": lambda t: t.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": lambda t: t.suggest_int("num_leaves", 15, 256),
        "max_depth": lambda t: t.suggest_int("max_depth", -1, 15),
        "min_data_in_leaf": lambda t: t.suggest_int("min_data_in_leaf", 10, 100),
        "feature_fraction": lambda t: t.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": lambda t: t.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": lambda t: t.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": lambda t: t.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": lambda t: t.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "min_gain_to_split": lambda t: t.suggest_float("min_gain_to_split", 0.0, 1.0),
        "extra_trees": lambda t: t.suggest_categorical("extra_trees", [True, False]),
        "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 1000),
    },

    "CatBoost": {
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": lambda t: t.suggest_int("depth", 4, 10),
        "l2_leaf_reg": lambda t: t.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "border_count": lambda t: t.suggest_int("border_count", 32, 255),
        "bagging_temperature": lambda t: t.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": lambda t: t.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "rsm": lambda t: t.suggest_float("rsm", 0.5, 1.0),
        "iterations": lambda t: t.suggest_int("iterations", 100, 1000),
    },

    "Decision Tree": {
        "criterion": lambda t: t.suggest_categorical("criterion", ["squared_error", "friedman_mse", "absolute_error", "poisson"]),
        "max_depth": lambda t: t.suggest_int("max_depth", 1, 32),
        "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 20)
    },

    "Random Forest": {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 500),
        "criterion": lambda t: t.suggest_categorical("criterion", ["squared_error", "absolute_error", "friedman_mse", "poisson"]),
        "max_depth": lambda t: t.suggest_int("max_depth", 1, 32),
        "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 20),
        "bootstrap": lambda t: t.suggest_categorical("bootstrap", [True, False])
    },

    "SVR": {
        "C": lambda t: t.suggest_float("C", 1e-3, 100, log=True),
        "kernel": lambda t: t.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"]),
        "gamma": lambda t: t.suggest_categorical("gamma", ["scale", "auto"]),
        "degree": lambda t: t.suggest_int("degree", 2, 5),
        "epsilon": lambda t: t.suggest_float("epsilon", 0.01, 1.0)
    },

    "KNNRegressor": {
        "n_neighbors": lambda t: t.suggest_int("n_neighbors", 1, 30),
        "weights": lambda t: t.suggest_categorical("weights", ["uniform", "distance"]),
        "p": lambda t: t.suggest_int("p", 1, 2),
        "leaf_size": lambda t: t.suggest_int("leaf_size", 10, 50)
    }
}
