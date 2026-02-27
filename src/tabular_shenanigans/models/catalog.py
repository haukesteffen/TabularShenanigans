from __future__ import annotations


def baseline_specs(task_type: str) -> list[dict]:
    if task_type == "classification":
        return [
            {
                "name": "xgb",
                "model_family": "xgboost",
                "model_params": {
                    "n_estimators": 500,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                },
            },
            {
                "name": "lgbm",
                "model_family": "lightgbm",
                "model_params": {
                    "n_estimators": 600,
                    "num_leaves": 63,
                    "learning_rate": 0.03,
                    "feature_fraction": 0.9,
                    "bagging_fraction": 0.9,
                },
            },
            {
                "name": "cat",
                "model_family": "catboost",
                "model_params": {
                    "iterations": 700,
                    "depth": 6,
                    "learning_rate": 0.04,
                },
            },
        ]

    return [
        {
            "name": "xgb",
            "model_family": "xgboost",
            "model_params": {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
        },
        {
            "name": "lgbm",
            "model_family": "lightgbm",
            "model_params": {
                "n_estimators": 700,
                "num_leaves": 63,
                "learning_rate": 0.03,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
            },
        },
        {
            "name": "cat",
            "model_family": "catboost",
            "model_params": {
                "iterations": 800,
                "depth": 6,
                "learning_rate": 0.04,
            },
        },
    ]


def tune_space(model_family: str) -> dict:
    mf = model_family.lower()
    if mf == "xgboost":
        return {
            "n_estimators": {"type": "int", "low": 300, "high": 1200, "step": 100},
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
        }
    if mf == "lightgbm":
        return {
            "n_estimators": {"type": "int", "low": 300, "high": 1400, "step": 100},
            "num_leaves": {"type": "int", "low": 31, "high": 255},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
            "feature_fraction": {"type": "float", "low": 0.6, "high": 1.0},
            "bagging_fraction": {"type": "float", "low": 0.6, "high": 1.0},
        }
    if mf == "catboost":
        return {
            "iterations": {"type": "int", "low": 300, "high": 1400, "step": 100},
            "depth": {"type": "int", "low": 4, "high": 10},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
            "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0},
        }
    return {}


def meta_learner_specs(task_type: str) -> list[dict]:
    if task_type == "classification":
        return [
            {"name": "mean", "method": "mean", "search_space": {}},
            {
                "name": "linear",
                "method": "linear",
                "search_space": {
                    "C": {"type": "float", "low": 0.05, "high": 20.0, "log": True},
                },
            },
        ]

    return [
        {"name": "mean", "method": "mean", "search_space": {}},
        {
            "name": "linear",
            "method": "linear",
            "search_space": {
                "alpha": {"type": "float", "low": 0.01, "high": 100.0, "log": True},
            },
        },
    ]
