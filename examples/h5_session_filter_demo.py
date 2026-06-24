from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import pandas as pd

from xrd_preprocessing import H5SessionFilter, filter_h5_sessions, h5_to_df, list_h5_sessions


DEFAULT_ARCHIVE = Path(
    "/Users/sad/dev/eos_play/jupyter_notebooks/Clinical_trials/"
    "data/product-aramis-data/combined_archive.h5"
)


def _select_demo_sessions(
    source_archive: Path,
    *,
    accepted_count: int,
    review_count: int,
) -> pd.DataFrame:
    sessions = filter_h5_sessions(source_archive, session_category="SAMPLE")
    return sessions.head(accepted_count + review_count).copy()


def build_tiny_archive(
    source_archive: Path,
    output_archive: Path,
    *,
    accepted_count: int = 3,
    review_count: int = 2,
) -> pd.DataFrame:
    selected = _select_demo_sessions(
        source_archive,
        accepted_count=accepted_count,
        review_count=review_count,
    )
    output_archive.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_archive, "r") as src, h5py.File(output_archive, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        for selected_idx, (_, row) in enumerate(selected.iterrows()):
            archive_group = row["archive_group"]
            session_name = row["archive_session_name"]
            session_path = row["archive_session_path"]
            dst_group = dst.require_group(archive_group)
            for key, value in src[archive_group].attrs.items():
                dst_group.attrs[key] = value
            src.copy(src[session_path], dst_group, name=session_name)
            copied = dst_group[session_name]
            copied.attrs["demo_quality_status"] = (
                "accepted" if selected_idx < accepted_count else "review"
            )

    return selected


def _scalar_preview(df: pd.DataFrame) -> pd.DataFrame:
    preview_columns = [
        "archive_group",
        "archive_session_name",
        "started_at",
        "category",
        "patientId",
        "specimenId",
        "set_name",
        "measurement_data_source",
        "sample_thickness_mm",
        "demo_quality_status",
    ]
    return df[[column for column in preview_columns if column in df.columns]].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a tiny H5 archive and demonstrate H5 attr filtering."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/output/h5_session_filter_demo/tiny_h5_filter_demo.h5"),
    )
    parser.add_argument("--accepted-count", type=int, default=3)
    parser.add_argument("--review-count", type=int, default=2)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    selected_for_tiny = build_tiny_archive(
        args.input,
        args.output,
        accepted_count=args.accepted_count,
        review_count=args.review_count,
    )
    all_sessions = list_h5_sessions(args.output)
    h5_filters = [H5SessionFilter("demo_quality_status", op="==", value="accepted")]
    selected_sessions = filter_h5_sessions(
        args.output,
        h5_filters,
        session_category="SAMPLE",
    )
    _, measurement_df = h5_to_df(
        args.output,
        data_preference="gfrm",
        drop_missing_sample_thickness=True,
        h5_filters=h5_filters,
        session_category="SAMPLE",
        set_category="SAMPLE",
    )

    preview = _scalar_preview(measurement_df)
    preview_path = args.output.parent / "measurement_preview.csv"
    preview.to_csv(preview_path, index=False)

    print(f"source_archive={args.input}")
    print(f"tiny_archive={args.output}")
    print(f"tiny_source_sessions={len(selected_for_tiny)}")
    print(f"h5_sessions_before_filter={len(all_sessions)}")
    print(f"h5_sessions_after_filter={len(selected_sessions)}")
    print(f"measurement_rows_after_h5_to_df={len(measurement_df)}")
    print(f"preview_csv={preview_path}")


if __name__ == "__main__":
    main()
