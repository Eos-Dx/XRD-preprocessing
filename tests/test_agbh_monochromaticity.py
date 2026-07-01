from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xrd_preprocessing import (
    AG_KBETA_TO_KALPHA_Q_RATIO,
    AgBHMonochromaticityFilter,
    AgBHMonochromaticityQualityControl,
    AgBHMonochromaticityScorer,
    agbh_filter_statistics,
    agbh_alpha_peaks,
    calculate_agbh_monochromaticity,
)
import xrd_preprocessing.agbh as agbh_module


def _agbh_profile(q: np.ndarray, *, kbeta_scale: float = 0.0) -> np.ndarray:
    y = np.full_like(q, 0.02, dtype=float)
    for order, alpha_q in enumerate(agbh_alpha_peaks(float(np.nanmax(q))), start=1):
        if order < 3:
            continue
        y += np.exp(-0.5 * ((q - alpha_q) / 0.035) ** 2)
        beta_q = alpha_q * AG_KBETA_TO_KALPHA_Q_RATIO
        y += kbeta_scale * np.exp(-0.5 * ((q - beta_q) / 0.055) ** 2)
    return y


def test_calculate_agbh_monochromaticity_scores_left_kbeta_residual():
    q = np.linspace(2.0, 23.0, 1600)
    reference = _agbh_profile(q, kbeta_scale=0.0)
    clean = _agbh_profile(q, kbeta_scale=0.0)
    contaminated = _agbh_profile(q, kbeta_scale=0.6)

    clean_result = calculate_agbh_monochromaticity(
        q,
        clean,
        baseline_q=q,
        baseline_intensity=reference,
        max_score=0.1,
        q_min=2.0,
        q_max=23.0,
    )
    contaminated_result = calculate_agbh_monochromaticity(
        q,
        contaminated,
        baseline_q=q,
        baseline_intensity=reference,
        max_score=0.1,
        q_min=2.0,
        q_max=23.0,
    )

    assert clean_result.score < 0.01
    assert clean_result.passed is True
    assert contaminated_result.score > clean_result.score
    assert contaminated_result.score > 0.1
    assert contaminated_result.passed is False
    assert contaminated_result.n_windows > 0


def test_agbh_monochromaticity_scorer_can_build_baseline_from_all_fit_rows():
    q = np.linspace(2.0, 23.0, 1600)
    agbh_df = pd.DataFrame(
        {
            "sample_id": ["clean", "contaminated"],
            "q_range": [q, q],
            "radial_profile_data": [
                _agbh_profile(q, kbeta_scale=0.0),
                _agbh_profile(q, kbeta_scale=0.6),
            ],
        }
    )

    scored = AgBHMonochromaticityScorer(
        max_score=0.1,
        q_min=2.0,
        q_max=23.0,
    ).fit_transform(agbh_df)
    accepted = AgBHMonochromaticityFilter(max_score=0.1).fit_transform(scored)
    stats = agbh_filter_statistics(scored, accepted)

    assert scored["agbh_monochromaticity_pass"].tolist() == [True, False]
    assert accepted["sample_id"].tolist() == ["clean"]
    assert accepted["agbh_monochromaticity_score"].iloc[0] < 0.1
    assert scored["agbh_baseline_fit_scale"].notna().all()
    assert "agbh_monochromaticity_pass" in scored.columns
    assert stats["filter_type"] == "agbh_monochromaticity"
    assert stats["rows_in"] == 2
    assert stats["rows_pass"] == 1
    assert stats["failed_ids"] == ["contaminated"]


def test_agbh_monochromaticity_scorer_accepts_optional_controlled_baseline():
    q = np.linspace(2.0, 23.0, 1600)
    reference_df = pd.DataFrame(
        {
            "q_range": [q, q],
            "radial_profile_data": [
                _agbh_profile(q, kbeta_scale=0.0),
                _agbh_profile(q, kbeta_scale=0.02),
            ],
        }
    )
    agbh_df = pd.DataFrame(
        {
            "sample_id": ["clean", "contaminated"],
            "q_range": [q, q],
            "radial_profile_data": [
                _agbh_profile(q, kbeta_scale=0.0),
                _agbh_profile(q, kbeta_scale=0.6),
            ],
        }
    )

    scorer = AgBHMonochromaticityScorer(
        reference_df=reference_df,
        max_score=0.1,
        q_min=2.0,
        q_max=23.0,
    )
    scored = scorer.fit_transform(agbh_df)

    assert scorer.stats_["baseline_source"] == "reference_df"
    assert scored["agbh_monochromaticity_pass"].tolist() == [True, False]


def test_agbh_monochromaticity_quality_control_exports_h5_filters_and_manifest():
    q = np.linspace(2.0, 23.0, 1600)
    agbh_df = pd.DataFrame(
        {
            "session_uid": ["agbh-clean-uid", "agbh-bad-uid"],
            "started_at": ["2026-04-22 09:00:00", "2026-04-29 09:00:00"],
            "q_range": [q, q],
            "radial_profile_data": [
                _agbh_profile(q, kbeta_scale=0.0),
                _agbh_profile(q, kbeta_scale=0.6),
            ],
        }
    )

    qc = AgBHMonochromaticityQualityControl(
        id_column="session_uid",
        max_score=0.1,
        q_min=2.0,
        q_max=23.0,
    )
    scored = qc.fit_transform(agbh_df)
    id_filter = qc.selection_.h5_id_filter(column="linked_agbh_session_uid")
    date_filter = qc.selection_.h5_date_filter()
    manifest = qc.selection_.manifest_columns()

    assert scored["agbh_monochromaticity_pass"].tolist() == [True, False]
    assert qc.selection_.accepted_ids == ["agbh-clean-uid"]
    assert qc.selection_.rejected_ids == ["agbh-bad-uid"]
    assert id_filter.column == "linked_agbh_session_uid"
    assert id_filter.op == "in"
    assert id_filter.values == ["agbh-clean-uid"]
    assert date_filter.column == "started_at"
    assert date_filter.op == "date in"
    assert date_filter.values == ["2026-04-22"]
    assert manifest["session_uid"].tolist() == ["agbh-clean-uid", "agbh-bad-uid"]


def test_agbh_monochromaticity_error_and_empty_selection_branches():
    q = np.linspace(2.0, 23.0, 50)
    y = np.ones_like(q)

    with pytest.raises(ValueError, match="baseline_q and baseline_intensity"):
        calculate_agbh_monochromaticity(q, y)

    with pytest.raises(ValueError, match="q_max must be greater"):
        calculate_agbh_monochromaticity(
            q,
            y,
            baseline_q=q,
            baseline_intensity=y,
            q_min=4.0,
            q_max=4.0,
        )

    with pytest.raises(ValueError, match="No AgBH K-beta windows"):
        calculate_agbh_monochromaticity(
            q,
            y,
            baseline_q=q,
            baseline_intensity=y,
            q_min=2.0,
            q_max=2.5,
        )

    with pytest.raises(KeyError, match="agbh_monochromaticity_score"):
        agbh_filter_statistics(pd.DataFrame({"id": ["x"]}), pd.DataFrame())

    with pytest.raises(ValueError, match="drop=False"):
        AgBHMonochromaticityFilter(drop=False)

    selection = agbh_module.AgBHMonochromaticitySelection(
        scored_df=pd.DataFrame({"passed": [False], "day": [pd.NaT]}),
        id_column=None,
        date_column=None,
        score_column="score",
        pass_column="passed",
    )

    assert selection.accepted_ids == []
    assert selection.rejected_ids == []
    assert selection.accepted_dates == []
    with pytest.raises(ValueError, match="id_column is required"):
        selection.h5_id_filter()
    with pytest.raises(ValueError, match="No accepted AgBH dates"):
        selection.h5_date_filter()


def test_agbh_helpers_skip_empty_ids_and_short_profiles():
    grid = np.linspace(1.0, 2.0, 5)

    np.testing.assert_array_equal(
        agbh_module._profile_on_grid(np.asarray([1.0]), np.asarray([1.0]), grid),
        np.full_like(grid, np.nan),
    )
    np.testing.assert_array_equal(
        agbh_module._normalize_profile(np.full(3, np.nan)),
        np.full(3, np.nan),
    )
    assert agbh_module.agbh_kbeta_windows(10.0, 11.0) == []
    assert agbh_module._unique_nonempty_values(pd.Series([None, "", "A", "A"])) == [
        "A"
    ]
    assert agbh_module._infer_id_column(pd.DataFrame({"id": [""]}), ("id",)) is None
