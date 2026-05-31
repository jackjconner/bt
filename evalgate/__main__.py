"""Entry point: ``python -m evalgate [--save] [--tolerance REL] [--golden PATH]``

Without ``--save``: runs the canonical pipeline, serializes the result,
and if a golden exists at ``--golden`` path diffs current vs golden and
exits non-zero if any field fails tolerance.

With ``--save``: writes the current summary to the golden path and exits 0.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from evalgate._core import (
    additive_only,
    classify_diff,
    diff_summaries,
    format_diff_table,
    load_golden,
    save_golden,
    serialize,
)


def _run_pipeline() -> dict[str, object]:
    """Import lazily so unit tests that only import _core never pay the cost."""
    from etl.datasets import GenSpec
    from pipeline import run_production_pipeline

    spec = GenSpec(n_assets=100, n_dates=252, n_features=20, n_factors=5, seed=0)
    workdir = Path(tempfile.mkdtemp(prefix="bt_evalgate_"))
    try:
        summary = run_production_pipeline(spec, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return serialize(summary)  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evalgate",
        description="Capture and diff the canonical PipelineSummary golden.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write current summary to the golden file and exit.",
    )
    parser.add_argument(
        "--tolerance",
        metavar="REL",
        type=float,
        default=1e-6,
        help="Relative tolerance for float comparisons (default: 1e-6).",
    )
    parser.add_argument(
        "--golden",
        metavar="PATH",
        type=Path,
        default=Path(".oversight/golden.json"),
        help="Path to the golden JSON file (default: .oversight/golden.json).",
    )
    parser.add_argument(
        "--allow-new-fields",
        action="store_true",
        help=(
            "Feature-round gate: pass when the only failing fields are ones the run "
            "*added* (present in current, absent from golden). Existing values must "
            "still hold within tolerance and no field may be dropped. Re-run with "
            "--save afterward to absorb the new fields into the golden."
        ),
    )
    parser.add_argument(
        "--field-tol",
        metavar="FIELD=REL",
        action="append",
        default=[],
        dest="field_tol",
        help=(
            "Per-field relative tolerance override, e.g. --field-tol backtest_p50_s=0.5. "
            "Repeat to set multiple fields. Useful for inherently non-deterministic fields "
            "like wall-clock timings."
        ),
    )
    args = parser.parse_args(argv)

    golden_path: Path = args.golden
    rel_tol: float = args.tolerance
    field_tol: dict[str, float] = {}
    for item in args.field_tol:
        if "=" not in item:
            parser.error(f"--field-tol requires FIELD=REL format, got: {item!r}")
        key, _, val_str = item.partition("=")
        try:
            field_tol[key.strip()] = float(val_str.strip())
        except ValueError:
            parser.error(f"--field-tol value must be a float, got: {val_str!r}")

    print("evalgate: running canonical pipeline …")
    current = _run_pipeline()

    if args.save:
        save_golden(current, golden_path)
        print(f"evalgate: golden written → {golden_path}")
        for field, value in current.items():
            print(f"  {field}: {value}")
        return 0

    # Diff mode.
    if not golden_path.exists():
        print(
            f"evalgate: no golden found at {golden_path}.\n  Run with --save to create one.",
            file=sys.stderr,
        )
        return 1

    golden = load_golden(golden_path)
    rows = diff_summaries(golden, current, rel_tol=rel_tol, field_tol=field_tol or None)
    print(format_diff_table(rows))

    if args.allow_new_fields:
        classification = classify_diff(rows)
        if additive_only(classification):
            if classification.new_fields:
                added = ", ".join(r.field for r in classification.new_fields)
                print(f"\nevalgate: --allow-new-fields — additive growth accepted: {added}")
                print("evalgate: re-run with --save to absorb the new fields into the golden.")
            else:
                print("\nevalgate: all fields within tolerance.")
            return 0
        offenders = classification.moved_existing + classification.missing_fields
        names = ", ".join(r.field for r in offenders)
        print(
            f"\nevalgate: --allow-new-fields — {len(offenders)} existing field(s) moved or "
            f"dropped (not additive): {names} — exiting non-zero.",
            file=sys.stderr,
        )
        return 1

    n_fail = sum(1 for r in rows if not r.passed)
    if n_fail:
        print(
            f"\nevalgate: {n_fail} field(s) failed tolerance — exiting non-zero.", file=sys.stderr
        )
        return 1

    print("\nevalgate: all fields within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
