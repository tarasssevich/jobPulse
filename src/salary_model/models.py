"""Salary Modeling & Machine Learning Pipeline for Job Market Research.

Implements baseline (DummyRegressor), regularized linear (Ridge),
and non-linear ensemble (Random Forest) models with k-fold cross-validation
and feature importance extraction for publication-grade salary prediction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

logger = logging.getLogger(__name__)

DEFAULT_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "experience_level",
    "country",
    "work_format",
)
DEFAULT_NUMERICAL_FEATURES: Final[tuple[str, ...]] = ("skills_count",)


@dataclass(frozen=True, slots=True)
class ModelMetrics:
    """Immutable evaluation metrics container for a salary regression model."""

    model_name: str
    r2: float
    mae: float
    rmse: float
    cv_r2_mean: float
    cv_r2_std: float

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "r2": round(self.r2, 4),
            "mae": round(self.mae, 2),
            "rmse": round(self.rmse, 2),
            "cv_r2_mean": round(self.cv_r2_mean, 4),
            "cv_r2_std": round(self.cv_r2_std, 4),
        }


class SalaryModelingPipeline:
    """End-to-end Machine Learning pipeline for IT salary prediction.

    Features:
    - Categorical: `experience_level`, `country`, `work_format` (One-Hot Encoded).
    - Numerical: `skills_count`.
    - Binary: All `skill_*` indicator columns.
    - Target: `salary_mid_byn`.

    Models:
    1. Baseline: DummyRegressor(strategy="median")
    2. Ridge: Ridge(alpha=1.0)
    3. Random Forest: RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=3)
    """

    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.2,
        cv_splits: int = 5,
        rf_estimators: int = 100,
        rf_max_depth: int = 12,
        rf_min_samples_leaf: int = 3,
        ridge_alpha: float = 1.0,
        top_n_features: int = 20,
    ) -> None:
        """Initialize salary modeling pipeline parameters."""
        self.random_state = random_state
        self.test_size = test_size
        self.cv_splits = cv_splits
        self.rf_estimators = rf_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.ridge_alpha = ridge_alpha
        self.top_n_features = top_n_features

        self.models: dict[str, Any] = {}
        self.feature_names_: list[str] = []
        self.results: dict[str, Any] | None = None

    def prepare_data(
        self, df: pd.DataFrame, is_train: bool = True
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix X and target vector y from input vacancy dataset.

        Performs pure filtering, handles missing values, and one-hot encodes categoricals.
        When is_train=False, aligns columns cleanly to self.feature_names_ to prevent
        data leakage and handle unseen or missing categories.

        Args:
            df: Input DataFrame containing vacancies, features, and salary_mid_byn.
            is_train: Whether this is the training split (learns feature names)
                or evaluation split (aligns to learned feature names).

        Returns:
            Tuple of (X: pd.DataFrame, y: pd.Series)

        Raises:
            TypeError: If df is not a DataFrame.
            ValueError: If target column is missing, no valid salary observations exist,
                or no feature columns are present.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if "salary_mid_byn" not in df.columns:
            raise ValueError("Column 'salary_mid_byn' not found in DataFrame.")

        # Clean target: coerce to numeric, filter out non-positive or null values
        sal_numeric = pd.to_numeric(df["salary_mid_byn"], errors="coerce")
        valid_mask = sal_numeric.notnull() & (sal_numeric > 0)
        df_clean = df.loc[valid_mask].copy()
        df_clean["salary_mid_byn"] = sal_numeric.loc[valid_mask]

        if df_clean.empty:
            raise ValueError("No valid records with positive salary_mid_byn found.")

        # 1. Categorical features (One-Hot Encoded)
        cat_cols = [c for c in DEFAULT_CATEGORICAL_FEATURES if c in df_clean.columns]
        if cat_cols:
            cat_df = df_clean[cat_cols].fillna("Missing").astype(str)
            X_cat = pd.get_dummies(cat_df, drop_first=False, dtype=int)
        else:
            X_cat = pd.DataFrame(index=df_clean.index)

        # 2. Numerical features
        num_cols = [c for c in DEFAULT_NUMERICAL_FEATURES if c in df_clean.columns]
        if num_cols:
            X_num = df_clean[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        else:
            X_num = pd.DataFrame(index=df_clean.index)

        # 3. Binary skill features (all columns starting with skill_)
        skill_cols = [c for c in df_clean.columns if c.startswith("skill_")]
        if skill_cols:
            X_skills = (
                df_clean[skill_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
            )
        else:
            X_skills = pd.DataFrame(index=df_clean.index)

        X = pd.concat([X_cat, X_num, X_skills], axis=1)

        if X.shape[1] == 0:
            raise ValueError("No feature columns found in dataset.")

        if is_train:
            self.feature_names_ = list(X.columns)
        elif self.feature_names_:
            # Align strictly with training features (prevent test leakage & handle missing/new categories)
            X = X.reindex(columns=self.feature_names_, fill_value=0)

        y = df_clean["salary_mid_byn"].astype(float)
        return X, y

    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets (80% train, 20% test).

        Args:
            X: Feature matrix.
            y: Target series.

        Returns:
            Tuple (X_train, X_test, y_train, y_test).
        """
        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> dict[str, Any]:
        """Train baseline, Ridge, and Random Forest models and evaluate metrics.

        Args:
            X_train: Training feature matrix.
            X_test: Test feature matrix.
            y_train: Training target.
            y_test: Test target.

        Returns:
            Structured results dictionary with metrics, models, and feature importances.
        """
        candidate_models = {
            "baseline": DummyRegressor(strategy="median"),
            "ridge": Ridge(alpha=self.ridge_alpha),
            "random_forest": RandomForestRegressor(
                n_estimators=self.rf_estimators,
                max_depth=self.rf_max_depth,
                min_samples_leaf=self.rf_min_samples_leaf,
                random_state=self.random_state,
            ),
        }

        # Guarantee test set columns strictly align to train set columns
        X_test_aligned = X_test.reindex(columns=X_train.columns, fill_value=0)

        metrics_dict: dict[str, dict[str, float]] = {}
        fitted_models: dict[str, Any] = {}

        actual_cv_splits = min(self.cv_splits, len(X_train))
        cv = (
            KFold(n_splits=actual_cv_splits, shuffle=True, random_state=self.random_state)
            if actual_cv_splits >= 2
            else None
        )

        for name, model in candidate_models.items():
            logger.info("Training %s regressor...", name)
            model.fit(X_train, y_train)
            fitted_models[name] = model

            y_pred = model.predict(X_test_aligned)
            r2 = float(r2_score(y_test, y_pred))
            if np.isnan(r2):
                r2 = 0.0
            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

            # k-fold cross validation R^2 on train set
            if cv is not None:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
                cv_mean = float(np.mean(cv_scores))
                if np.isnan(cv_mean):
                    cv_mean = 0.0
                cv_std = float(np.std(cv_scores))
                if np.isnan(cv_std):
                    cv_std = 0.0
            else:
                cv_mean = r2
                cv_std = 0.0

            model_metrics = ModelMetrics(
                model_name=name,
                r2=r2,
                mae=mae,
                rmse=rmse,
                cv_r2_mean=cv_mean,
                cv_r2_std=cv_std,
            )

            metrics_dict[name] = model_metrics.to_dict()

        # Provide aliases for capitalization flexibility
        metrics_dict["Dummy"] = metrics_dict["baseline"]
        metrics_dict["Ridge"] = metrics_dict["ridge"]
        metrics_dict["RandomForest"] = metrics_dict["random_forest"]

        # Feature importances for Random Forest (Top N)
        rf_model: RandomForestRegressor = fitted_models["random_forest"]
        importances = rf_model.feature_importances_
        fi_series = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)

        top_fi_df = pd.DataFrame(
            {
                "feature": fi_series.index[: self.top_n_features],
                "importance": fi_series.values[: self.top_n_features],
            }
        )

        top_fi_list = [
            {"feature": feat, "importance": round(float(val), 6)}
            for feat, val in fi_series.head(self.top_n_features).items()
        ]

        self.models = fitted_models
        self.feature_names_ = list(X_train.columns)

        results: dict[str, Any] = {
            "metrics": metrics_dict,
            "models": fitted_models,
            "feature_importances": top_fi_df,
            "feature_importances_list": top_fi_list,
            "feature_names": list(X_train.columns),
            "train_size": len(X_train),
            "test_size": len(X_test_aligned),
        }
        self.results = results
        return results

    def predict(self, df: pd.DataFrame, model_name: str = "random_forest") -> np.ndarray:
        """Predict salaries for new vacancy observations using trained model.

        Args:
            df: Input DataFrame with vacancy features.
            model_name: Model identifier ('baseline', 'ridge', or 'random_forest').

        Returns:
            np.ndarray of predicted salary values in BYN.

        Raises:
            ValueError: If model is not trained or features are uninitialized.
        """
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' has not been trained yet. Available: {list(self.models.keys())}"
            )
        if not self.feature_names_:
            raise ValueError("Pipeline has not been fitted with feature names yet.")

        # Prepare features aligned to training set
        cat_cols = [c for c in DEFAULT_CATEGORICAL_FEATURES if c in df.columns]
        if cat_cols:
            cat_df = df[cat_cols].fillna("Missing").astype(str)
            X_cat = pd.get_dummies(cat_df, drop_first=False, dtype=int)
        else:
            X_cat = pd.DataFrame(index=df.index)

        num_cols = [c for c in DEFAULT_NUMERICAL_FEATURES if c in df.columns]
        if num_cols:
            X_num = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        else:
            X_num = pd.DataFrame(index=df.index)

        skill_cols = [c for c in df.columns if c.startswith("skill_")]
        if skill_cols:
            X_skills = df[skill_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        else:
            X_skills = pd.DataFrame(index=df.index)

        X = pd.concat([X_cat, X_num, X_skills], axis=1)
        X_aligned = X.reindex(columns=self.feature_names_, fill_value=0)
        return self.models[model_name].predict(X_aligned)

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """Execute the end-to-end salary modeling pipeline with strict leakage prevention.

        Splits data first, prepares features strictly on train, and aligns test features.

        Args:
            df: Vacancies dataset.

        Returns:
            Dictionary with evaluation metrics, trained models, and feature importances.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if "salary_mid_byn" not in df.columns:
            raise ValueError("Column 'salary_mid_byn' not found in DataFrame.")

        sal_numeric = pd.to_numeric(df["salary_mid_byn"], errors="coerce")
        valid_mask = sal_numeric.notnull() & (sal_numeric > 0)
        df_valid = df.loc[valid_mask].copy()
        df_valid["salary_mid_byn"] = sal_numeric.loc[valid_mask]

        if len(df_valid) < 2:
            raise ValueError(
                "At least 2 valid salary records are required for train/test splitting."
            )

        train_df, test_df = train_test_split(
            df_valid,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        X_train, y_train = self.prepare_data(train_df, is_train=True)
        X_test, y_test = self.prepare_data(test_df, is_train=False)

        return self.train_and_evaluate(X_train, X_test, y_train, y_test)
