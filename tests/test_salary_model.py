"""Unit and integration tests for JobPulse Step 3: Econometrics, ML Models & Visualizations.

Covers:
- `calculate_skill_premiums` mathematical logic, Mann-Whitney U test, filtering, edge cases.
- `SalaryModelingPipeline` data preparation, one-hot encoding, model fitting, CV, feature importances.
- `AcademicVisualizer` 300 DPI figures generation, styling, valid PNG output.
- `run_salary_analysis` end-to-end integration and artifact persistence.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.analysis.visualizer import AcademicVisualizer, format_feature_label
from src.salary_model.econometrics import (
    PREMIUM_COLUMNS,
    calculate_skill_premiums,
    get_canonical_skill_name,
)
from src.salary_model.models import ModelMetrics, SalaryModelingPipeline
from src.salary_model.runner import run_salary_analysis

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_vacancies_df() -> pd.DataFrame:
    """Generate reproducible synthetic dataset for testing econometrics and modeling."""
    np.random.seed(42)
    n = 100

    grades = np.random.choice(
        ["Junior", "Middle", "Middle+", "Senior"], size=n, p=[0.2, 0.4, 0.3, 0.1]
    )
    countries = np.random.choice(["Беларусь", "Россия", "Казахстан"], size=n)
    work_formats = np.random.choice(["office", "remote", "hybrid"], size=n)

    # Base salary by grade
    base_salaries = {
        "Junior": 1500.0,
        "Middle": 3500.0,
        "Middle+": 5500.0,
        "Senior": 8500.0,
    }

    # Skill indicators
    skill_python = np.random.binomial(1, 0.4, size=n)
    skill_docker = np.random.binomial(1, 0.3, size=n)
    skill_rare = np.array([1, 1] + [0] * (n - 2))  # Only 2 occurrences (below min_vacancies=15)
    skill_ubiquitous = np.ones(n, dtype=int)  # 100% occurrences (no control group)

    salaries: list[float | None] = []
    for i in range(n):
        if i % 10 == 0:
            salaries.append(None)  # 10% null salaries
        else:
            sal = base_salaries[grades[i]]
            # Add premiums
            if skill_python[i] == 1:
                sal += 1000.0
            if skill_docker[i] == 1:
                sal += 800.0
            sal += np.random.normal(0, 200)
            salaries.append(max(500.0, sal))

    skills_count = skill_python + skill_docker + skill_rare

    return pd.DataFrame(
        {
            "vacancy_id": [f"vac_{i}" for i in range(n)],
            "title": [f"Developer {i}" for i in range(n)],
            "salary_mid_byn": salaries,
            "experience_level": grades,
            "country": countries,
            "work_format": work_formats,
            "skills_count": skills_count,
            "skill_python": skill_python,
            "skill_docker": skill_docker,
            "skill_rare": skill_rare,
            "skill_ubiquitous": skill_ubiquitous,
        }
    )


# =============================================================================
# 1. Econometric Analysis Tests
# =============================================================================


def test_get_canonical_skill_name() -> None:
    """Test resolution of raw column names to canonical display names."""
    assert get_canonical_skill_name("skill_python") == "Python"
    assert get_canonical_skill_name("skill_ci_cd") == "CI/CD"
    assert get_canonical_skill_name("skill_docker") == "Docker"
    assert get_canonical_skill_name("skill_custom_test") == "Custom Test"


def test_calculate_skill_premiums_columns_and_sorting(synthetic_vacancies_df: pd.DataFrame) -> None:
    """Test calculate_skill_premiums schema and sort order."""
    df_res = calculate_skill_premiums(synthetic_vacancies_df, min_vacancies=5)

    assert isinstance(df_res, pd.DataFrame)
    for col in PREMIUM_COLUMNS:
        assert col in df_res.columns, f"Missing column {col} in result"

    # Sorted by premium_pct descending
    if len(df_res) > 1:
        assert (df_res["premium_pct"].diff().dropna() <= 0).all()


def test_calculate_skill_premiums_min_vacancies_filter(
    synthetic_vacancies_df: pd.DataFrame,
) -> None:
    """Skills with fewer vacancies than min_vacancies must be filtered out."""
    # skill_rare has only 2 occurrences
    df_res_high = calculate_skill_premiums(synthetic_vacancies_df, min_vacancies=15)
    assert "skill_rare" not in df_res_high["skill"].values

    # skill_ubiquitous has no control group (n_without == 0) and must be skipped safely
    assert "skill_ubiquitous" not in df_res_high["skill"].values


def test_calculate_skill_premiums_math_and_significance() -> None:
    """Test mathematical accuracy of median diff, premium %, and Mann-Whitney U test."""
    # 20 observations with skill, 20 without
    # Group with skill: median 2000
    # Group without: median 1000
    df = pd.DataFrame(
        {
            "salary_mid_byn": [2000.0] * 20 + [1000.0] * 20,
            "skill_test": [1] * 20 + [0] * 20,
        }
    )

    res = calculate_skill_premiums(df, min_vacancies=10)
    assert len(res) == 1
    row = res.iloc[0]

    assert row["skill"] == "skill_test"
    assert row["N_with"] == 20
    assert row["median_with"] == 2000.0
    assert row["median_without"] == 1000.0
    assert row["diff_median"] == 1000.0
    assert row["premium_pct"] == 100.0  # (2000 - 1000) / 1000 * 100
    assert row["p_value"] < 0.001
    assert bool(row["is_significant"]) is True


def test_calculate_skill_premiums_insignificant_effect() -> None:
    """Test that equal salary distributions yield is_significant = False."""
    np.random.seed(99)
    salaries = np.random.normal(3000, 50, size=40)
    df = pd.DataFrame(
        {
            "salary_mid_byn": salaries,
            "skill_neutral": [1] * 20 + [0] * 20,
        }
    )

    res = calculate_skill_premiums(df, min_vacancies=10)
    assert len(res) == 1
    row = res.iloc[0]
    assert row["p_value"] > 0.05
    assert bool(row["is_significant"]) is False


def test_calculate_skill_premiums_immutability(synthetic_vacancies_df: pd.DataFrame) -> None:
    """Input DataFrame must remain unchanged after calling calculate_skill_premiums."""
    copy_before = synthetic_vacancies_df.copy(deep=True)
    _ = calculate_skill_premiums(synthetic_vacancies_df, min_vacancies=5)
    pd.testing.assert_frame_equal(synthetic_vacancies_df, copy_before)


def test_calculate_skill_premiums_edge_cases() -> None:
    """Test edge cases: empty DF, missing salary column, invalid inputs."""
    # Empty DataFrame
    empty_df = pd.DataFrame(columns=["salary_mid_byn", "skill_python"])
    res_empty = calculate_skill_premiums(empty_df)
    assert isinstance(res_empty, pd.DataFrame)
    assert len(res_empty) == 0

    # Missing salary column
    with pytest.raises(ValueError, match="Column 'salary_mid_byn' not found"):
        calculate_skill_premiums(pd.DataFrame({"skill_python": [1, 0]}))

    # Invalid min_vacancies
    with pytest.raises(ValueError, match="min_vacancies must be at least 1"):
        calculate_skill_premiums(pd.DataFrame({"salary_mid_byn": [1000]}), min_vacancies=0)

    # Invalid input type
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        calculate_skill_premiums([1, 2, 3])  # type: ignore[arg-type]


# =============================================================================
# 2. Salary ML Modeling Pipeline Tests
# =============================================================================


def test_model_metrics_dataclass() -> None:
    """Test ModelMetrics dataclass immutability and conversion."""
    m = ModelMetrics(
        model_name="ridge",
        r2=0.4556,
        mae=1666.38,
        rmse=2192.61,
        cv_r2_mean=0.3384,
        cv_r2_std=0.1304,
    )
    d = m.to_dict()
    assert d["r2"] == 0.4556
    assert d["mae"] == 1666.38
    with pytest.raises(AttributeError):
        m.r2 = 0.5  # Immutable frozen dataclass


def test_salary_modeling_pipeline_data_prep(synthetic_vacancies_df: pd.DataFrame) -> None:
    """Test pipeline feature preparation, one-hot encoding, and target filtering."""
    pipe = SalaryModelingPipeline(random_state=42)
    X, y = pipe.prepare_data(synthetic_vacancies_df)

    # Only rows with valid salary (non-null and > 0)
    expected_rows = (
        synthetic_vacancies_df["salary_mid_byn"].notnull()
        & (synthetic_vacancies_df["salary_mid_byn"] > 0)
    ).sum()
    assert len(X) == expected_rows
    assert len(y) == expected_rows

    # Verify one-hot encoded columns exist
    assert any(c.startswith("experience_level_") for c in X.columns)
    assert any(c.startswith("country_") for c in X.columns)
    assert any(c.startswith("work_format_") for c in X.columns)
    assert "skills_count" in X.columns
    assert "skill_python" in X.columns
    assert "skill_docker" in X.columns


def test_salary_modeling_pipeline_train_and_evaluate(synthetic_vacancies_df: pd.DataFrame) -> None:
    """Test model training, CV scoring, and feature importances extraction."""
    pipe = SalaryModelingPipeline(random_state=42, test_size=0.2, cv_splits=3, rf_estimators=10)
    results = pipe.run(synthetic_vacancies_df)

    assert "metrics" in results
    assert "models" in results
    assert "feature_importances" in results
    assert "train_size" in results
    assert "test_size" in results

    metrics = results["metrics"]
    for m_name in ["baseline", "ridge", "random_forest"]:
        assert m_name in metrics
        assert "r2" in metrics[m_name]
        assert "mae" in metrics[m_name]
        assert "rmse" in metrics[m_name]
        assert "cv_r2_mean" in metrics[m_name]
        assert "cv_r2_std" in metrics[m_name]
        assert isinstance(metrics[m_name]["r2"], float)

    # Random forest should outperform baseline on non-trivial synthetic data
    assert metrics["random_forest"]["mae"] < metrics["baseline"]["mae"]

    # Top feature importances check
    fi_df = results["feature_importances"]
    assert isinstance(fi_df, pd.DataFrame)
    assert "feature" in fi_df.columns
    assert "importance" in fi_df.columns
    assert len(fi_df) <= pipe.top_n_features
    # Feature importances must be sorted descending
    assert (fi_df["importance"].diff().dropna() <= 0).all()


def test_salary_modeling_pipeline_invalid_inputs() -> None:
    """Test pipeline exceptions on invalid inputs."""
    pipe = SalaryModelingPipeline()

    # Not a DataFrame
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        pipe.prepare_data("invalid")  # type: ignore[arg-type]

    # Missing target
    with pytest.raises(ValueError, match="Column 'salary_mid_byn' not found"):
        pipe.prepare_data(pd.DataFrame({"a": [1, 2]}))

    # No valid salary records
    with pytest.raises(ValueError, match="No valid records with positive salary_mid_byn"):
        pipe.prepare_data(pd.DataFrame({"salary_mid_byn": [None, 0.0, -100.0]}))


# =============================================================================
# 3. Academic Visualizer Tests
# =============================================================================


def test_format_feature_label() -> None:
    """Test feature label formatter."""
    assert format_feature_label("experience_level_Senior") == "Grade: Senior"
    assert format_feature_label("work_format_office") == "Format: Office"
    assert format_feature_label("country_Беларусь") == "Country: Belarus"
    assert format_feature_label("skills_count") == "Total Skills Count"
    assert format_feature_label("skill_ci_cd") == "Skill: CI/CD"


def test_academic_visualizer_figures_generation(
    tmp_path: Path, synthetic_vacancies_df: pd.DataFrame
) -> None:
    """Test generating all 5 publication figures into a temporary directory."""
    vis = AcademicVisualizer(output_dir=tmp_path / "figs", dpi=100)
    premiums = calculate_skill_premiums(synthetic_vacancies_df, min_vacancies=5)

    pipe = SalaryModelingPipeline(random_state=42, rf_estimators=10)
    res = pipe.run(synthetic_vacancies_df)

    figures = vis.generate_all_figures(
        df=synthetic_vacancies_df,
        df_premiums=premiums,
        feature_importances=res["feature_importances"],
    )

    assert len(figures) == 5
    for fig_id in ["fig1", "fig2", "fig3", "fig4", "fig5"]:
        assert fig_id in figures
        fig_path = figures[fig_id]
        assert fig_path.exists(), f"Figure {fig_id} was not created"
        assert fig_path.stat().st_size > 1000, f"Figure {fig_id} file is empty or too small"
        # Validate PNG magic bytes
        with open(fig_path, "rb") as f:
            header = f.read(8)
            assert header == b"\x89PNG\r\n\x1a\n", f"Figure {fig_id} is not a valid PNG file"


# =============================================================================
# 4. End-to-End Orchestrator Integration Test
# =============================================================================


def test_run_salary_analysis_e2e(tmp_path: Path, synthetic_vacancies_df: pd.DataFrame) -> None:
    """Test run_salary_analysis end-to-end execution on synthetic dataset."""
    in_parquet = tmp_path / "synthetic_vacancies.parquet"
    synthetic_vacancies_df.to_parquet(in_parquet, index=False)

    out_csv = tmp_path / "skill_premiums.csv"
    out_json = tmp_path / "model_metrics.json"
    figs_dir = tmp_path / "figures"

    output = run_salary_analysis(
        input_path=in_parquet,
        premiums_output_path=out_csv,
        metrics_output_path=out_json,
        figures_dir=figs_dir,
        min_vacancies_for_premium=5,
    )

    # Check artifacts created
    assert out_csv.exists()
    assert out_json.exists()
    assert figs_dir.exists()
    assert len(list(figs_dir.glob("*.png"))) == 5

    # Check metrics JSON structure
    with open(out_json, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    assert "models" in metrics_data
    assert "baseline" in metrics_data["models"]
    assert "ridge" in metrics_data["models"]
    assert "random_forest" in metrics_data["models"]
    assert "feature_importances_top20" in metrics_data
    assert "econometrics_summary" in metrics_data

    # Check output dictionary
    assert isinstance(output["df_premiums"], pd.DataFrame)
    assert isinstance(output["model_results"], dict)
    assert len(output["generated_figures"]) == 5


# =============================================================================
# 5. Adversarial Edge Case & Robustness Tests
# =============================================================================


def test_calculate_skill_premiums_zero_variance_ties() -> None:
    """Test that 100% tied observations across groups yield p_value = 1.0 rather than NaN."""
    # Arrange
    df = pd.DataFrame(
        {
            "salary_mid_byn": [2500.0] * 30,
            "skill_constant": [1] * 15 + [0] * 15,
        }
    )

    # Act
    res = calculate_skill_premiums(df, min_vacancies=10)

    # Assert
    assert len(res) == 1
    row = res.iloc[0]
    assert row["p_value"] == 1.0
    assert row["diff_median"] == 0.0
    assert row["premium_pct"] == 0.0
    assert bool(row["is_significant"]) is False


def test_calculate_skill_premiums_boolean_and_numeric_types() -> None:
    """Verify boolean dtype, float dtype, and int dtype indicator columns are handled identically."""
    # Arrange
    salaries = [3000.0] * 20 + [2000.0] * 20
    df = pd.DataFrame(
        {
            "salary_mid_byn": salaries,
            "skill_int": [1] * 20 + [0] * 20,
            "skill_bool": [True] * 20 + [False] * 20,
            "skill_float": [1.0] * 20 + [0.0] * 20,
        }
    )

    # Act
    res = calculate_skill_premiums(df, min_vacancies=15)

    # Assert
    assert len(res) == 3
    for _, row in res.iterrows():
        assert row["diff_median"] == 1000.0
        assert row["premium_pct"] == 50.0
        assert bool(row["is_significant"]) is True


def test_calculate_skill_premiums_ubiquitous_and_zero_prevalence() -> None:
    """Skills with 100% prevalence (no control) or 0% prevalence (no test) are skipped safely."""
    # Arrange
    df = pd.DataFrame(
        {
            "salary_mid_byn": [3000.0] * 20,
            "skill_all_ones": [1] * 20,
            "skill_all_zeros": [0] * 20,
        }
    )

    # Act
    res = calculate_skill_premiums(df, min_vacancies=5)

    # Assert
    assert len(res) == 0


def test_calculate_skill_premiums_negative_and_string_salaries() -> None:
    """Non-positive and non-numeric salary values must be filtered out cleanly without error."""
    # Arrange
    df = pd.DataFrame(
        {
            "salary_mid_byn": [3000.0] * 15 + [1500.0] * 15 + [-500.0, 0.0, "invalid", None],
            "skill_py": [1] * 15 + [0] * 15 + [1, 0, 1, 0],
        }
    )

    # Act
    res = calculate_skill_premiums(df, min_vacancies=10)

    # Assert
    assert len(res) == 1
    row = res.iloc[0]
    assert row["N_with"] == 15
    assert row["median_with"] == 3000.0
    assert row["median_without"] == 1500.0
    assert row["premium_pct"] == 100.0


def test_salary_modeling_pipeline_predict_method(synthetic_vacancies_df: pd.DataFrame) -> None:
    """Pipeline.predict generates valid salary predictions on unseen records."""
    # Arrange
    pipe = SalaryModelingPipeline(random_state=42, rf_estimators=10)
    pipe.run(synthetic_vacancies_df)
    new_data = pd.DataFrame(
        {
            "experience_level": ["Senior", "Junior", "UnknownGrade"],
            "country": ["Беларусь", "НоваяСтрана", "Россия"],
            "work_format": ["remote", "office", "freelance"],
            "skills_count": [5, 1, 3],
            "skill_python": [1, 0, 1],
            "skill_docker": [1, 0, 0],
        }
    )

    # Act
    preds = pipe.predict(new_data, model_name="random_forest")

    # Assert
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 3
    assert (preds > 0).all()


def test_salary_modeling_pipeline_unseen_categories_leakage_free(
    synthetic_vacancies_df: pd.DataFrame,
) -> None:
    """Pipeline handles unseen categories in evaluation without data leakage or crashes."""
    # Arrange
    pipe = SalaryModelingPipeline(random_state=42, rf_estimators=10)
    train_df = synthetic_vacancies_df.iloc[:80].copy()
    test_df = synthetic_vacancies_df.iloc[80:].copy()
    # Inject novel categorical value into test set only
    test_df.loc[test_df.index[0], "experience_level"] = "ExecutiveVP"

    # Act
    X_train, y_train = pipe.prepare_data(train_df, is_train=True)
    X_test, y_test = pipe.prepare_data(test_df, is_train=False)

    # Assert
    assert "experience_level_ExecutiveVP" not in X_train.columns
    assert "experience_level_ExecutiveVP" not in X_test.columns
    assert list(X_train.columns) == list(X_test.columns)

    # Train and evaluate without errors
    results = pipe.train_and_evaluate(X_train, X_test, y_train, y_test)
    assert "random_forest" in results["metrics"]


def test_salary_modeling_pipeline_edge_cases() -> None:
    """Pipeline raises descriptive ValueError when inputs are insufficient or featureless."""
    # Arrange
    pipe = SalaryModelingPipeline()

    # Empty DataFrame
    empty_df = pd.DataFrame(columns=["salary_mid_byn", "skill_python"])
    with pytest.raises(ValueError, match="No valid records with positive salary_mid_byn"):
        pipe.prepare_data(empty_df)

    # Less than 2 samples for run
    single_row = pd.DataFrame(
        {
            "salary_mid_byn": [3000.0],
            "experience_level": ["Senior"],
            "skill_python": [1],
        }
    )
    with pytest.raises(ValueError, match="At least 2 valid salary records are required"):
        pipe.run(single_row)

    # Calling predict before train
    with pytest.raises(ValueError, match="has not been trained yet"):
        pipe.predict(single_row)


def test_academic_visualizer_resource_cleanup_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_vacancies_df: pd.DataFrame
) -> None:
    """Matplotlib figures must be closed via plt.close(fig) even if savefig raises an error."""
    # Arrange
    vis = AcademicVisualizer(output_dir=tmp_path / "figs")

    def mock_savefig(*args: object, **kwargs: object) -> None:
        raise OSError("Disk full simulation")

    monkeypatch.setattr(plt.Figure, "savefig", mock_savefig)

    # Act & Assert
    with pytest.raises(OSError, match="Disk full simulation"):
        vis.plot_salary_distribution_by_grade(synthetic_vacancies_df)

    # Verify no open figures remain in matplotlib state
    assert len(plt.get_fignums()) == 0


def test_academic_visualizer_edge_cases(tmp_path: Path) -> None:
    """AcademicVisualizer properly validates edge case inputs."""
    # Arrange
    vis = AcademicVisualizer(output_dir=tmp_path / "figs")

    # Empty DataFrame for plot_top_skills_demand
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        vis.plot_top_skills_demand(pd.DataFrame(columns=["skill_python"]))

    # Missing required columns in plot_salary_distribution_by_grade
    with pytest.raises(ValueError, match="DataFrame must contain"):
        vis.plot_salary_distribution_by_grade(pd.DataFrame({"salary_mid_byn": [1000]}))

    # Empty df_premiums in plot_skill_premiums
    with pytest.raises(ValueError, match="df_premiums is empty"):
        vis.plot_skill_premiums(pd.DataFrame())

    # Empty feature_importances in plot_feature_importances
    with pytest.raises(ValueError, match="feature_importances is empty"):
        vis.plot_feature_importances(pd.DataFrame())


def test_run_salary_analysis_no_salaries_error(tmp_path: Path) -> None:
    """run_salary_analysis raises ValueError if input dataset has no positive salaries."""
    # Arrange
    empty_parquet = tmp_path / "no_salaries.parquet"
    pd.DataFrame(
        {
            "salary_mid_byn": [None, 0.0, -100.0],
            "skill_python": [1, 0, 1],
        }
    ).to_parquet(empty_parquet)

    # Act & Assert
    with pytest.raises(ValueError, match="No valid records with positive salary_mid_byn found"):
        run_salary_analysis(input_path=empty_parquet, figures_dir=tmp_path / "figs")
