#!/usr/bin/env python3
"""Interactive console viewer for mined myopia progression data."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "medical_config.json"
Record = Dict[str, Any]


def load_payload(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Cannot find config file: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_control_lookup(records: Sequence[Record]) -> Dict[Tuple[str, int], float]:
    lookup: Dict[Tuple[str, int], float] = {}
    for rec in records:
        if rec.get("treatment_key") != "Natural":
            continue
        ethnicity = rec.get("ethnicity")
        age = rec.get("age")
        mean = rec.get("mean")
        if ethnicity is None or age is None or mean is None:
            continue
        try:
            lookup[(str(ethnicity), int(age))] = float(mean)
        except (TypeError, ValueError):
            continue
    return lookup


def compute_delta(record: Record, controls: Dict[Tuple[str, int], float]) -> Optional[float]:
    if record.get("treatment_key") == "Natural":
        return None
    ethnicity = record.get("ethnicity")
    age = record.get("age")
    mean = record.get("mean")
    if ethnicity is None or age is None or mean is None:
        return None
    control_mean = controls.get((str(ethnicity), int(age)))
    if control_mean is None:
        return None
    try:
        return float(control_mean) - float(mean)
    except (TypeError, ValueError):
        return None


def aggregated_stats(records: Sequence[Record]) -> Optional[Dict[str, float]]:
    cleaned: List[Tuple[float, int, float]] = []
    for rec in records:
        mean = rec.get("mean")
        n_raw = rec.get("total_n") or rec.get("n")
        if mean is None or n_raw is None:
            continue
        try:
            mean_val = float(mean)
            n = int(n_raw)
        except (TypeError, ValueError):
            continue
        if n <= 0:
            continue
        sd_raw = rec.get("sd") or 0.0
        try:
            sd_val = float(sd_raw)
        except (TypeError, ValueError):
            sd_val = 0.0
        cleaned.append((mean_val, n, sd_val))
    if not cleaned:
        return None
    total_n = sum(item[1] for item in cleaned)
    weighted_mean = sum(item[0] * item[1] for item in cleaned) / total_n
    if total_n == 1:
        pooled_sd = cleaned[0][2]
    else:
        numerator = 0.0
        for mean_val, n, sd_val in cleaned:
            numerator += max(n - 1, 0) * (sd_val ** 2) + n * ((mean_val - weighted_mean) ** 2)
        pooled_sd = math.sqrt(max(numerator / (total_n - 1), 0.0))
    return {"mean": weighted_mean, "sd": pooled_sd, "total_n": total_n}


def summarize_by_treatment(records: Sequence[Record]) -> List[Dict[str, float]]:
    buckets: Dict[str, List[Record]] = defaultdict(list)
    for rec in records:
        treatment = rec.get("treatment")
        if not treatment:
            continue
        buckets[treatment].append(rec)
    summary: List[Dict[str, float]] = []
    for treatment, rows in buckets.items():
        stats = aggregated_stats(rows)
        if not stats:
            continue
        summary.append(
            {
                "name": treatment,
                "mean": stats["mean"],
                "sd": stats["sd"],
                "total_n": stats["total_n"],
                "cohorts": len(rows),
            }
        )
    summary.sort(key=lambda item: item["mean"], reverse=True)
    return summary


def highlight_best_delta(records: Sequence[Record], controls: Dict[Tuple[str, int], float]) -> Tuple[Optional[Record], Optional[float]]:
    best_row: Optional[Record] = None
    best_delta: Optional[float] = None
    for rec in records:
        delta = compute_delta(rec, controls)
        if delta is None:
            continue
        if best_delta is None or delta > best_delta:
            best_row = rec
            best_delta = delta
    return best_row, best_delta


def format_entry(rec: Record) -> str:
    total_n = rec.get("total_n") or rec.get("n") or "-"
    mean = rec.get("mean")
    mean_text = "n/a" if mean is None else f"{float(mean):+.3f}"
    return f"{rec.get('ethnicity')} age {rec.get('age')} {rec.get('treatment')} ({mean_text} D/yr, n={total_n})"


def format_sources(rec: Record) -> str:
    sources = rec.get("source_ids")
    if not sources:
        return "-"
    if len(sources) <= 2:
        return ", ".join(sources)
    return ", ".join(sources[:2]) + f", +{len(sources) - 2} more"


def format_list(values: Optional[Sequence[str]]) -> str:
    if not values:
        return "All"
    unique = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    if len(unique) <= 4:
        return ", ".join(unique)
    return f"{unique[0]}, {unique[1]}, ..., {unique[-1]} (total {len(unique)})"


def format_age_list(values: Optional[Sequence[int]]) -> str:
    if not values:
        return "All"
    dedup = sorted({int(v) for v in values})
    if len(dedup) <= 6:
        return ", ".join(str(v) for v in dedup)
    return f"{dedup[0]}-{dedup[-1]} (total {len(dedup)})"


def describe_selection(selection: Dict[str, Any]) -> str:
    return (
        f"Ethnicity: {format_list(selection.get('ethnicities'))} | "
        f"Ages: {format_age_list(selection.get('ages'))} | "
        f"Treatments: {format_list(selection.get('treatments'))} | "
        f"Sort: {selection.get('sort_mode', 'best')} | Rows: {selection.get('top_k', 10)}"
    )


def print_dataset_summary(
    config_path: Path,
    payload: Dict[str, Any],
    records: Sequence[Record],
    controls: Dict[Tuple[str, int], float],
) -> None:
    meta = payload.get("meta_schema") or {}
    generated = payload.get("generated_at", "unknown")
    total_patients = sum(int(rec.get("total_n") or 0) for rec in records)
    eth_age_pairs = {(rec.get("ethnicity"), rec.get("age")) for rec in records}
    print("=" * 78)
    print(f"Loaded mining artifact: {config_path}")
    print(f"Generated at: {generated}")
    print(
        f"Aggregated cohorts: {len(records)} | Unique ethnicity/age pairs: {len(eth_age_pairs)} | "
        f"Patients represented: {total_patients}"
    )

    eth_counts = Counter(rec.get("ethnicity") for rec in records if rec.get("ethnicity"))
    treat_counts = Counter(rec.get("treatment") for rec in records if rec.get("treatment"))
    if eth_counts:
        print("Ethnicity coverage: " + ", ".join(f"{k} ({v})" for k, v in eth_counts.items()))
    if treat_counts:
        print(
            "Treatment coverage: " + ", ".join(f"{k} ({v})" for k, v in treat_counts.items())
        )

    if records:
        best_entry = max(records, key=lambda r: r.get("mean", float("-inf")))
        worst_entry = min(records, key=lambda r: r.get("mean", float("inf")))
        print(f"Slowest progression cohort: {format_entry(best_entry)}")
        print(f"Fastest progression cohort: {format_entry(worst_entry)}")
    uplift_entry, uplift_value = highlight_best_delta(records, controls)
    if uplift_entry and uplift_value is not None:
        print(
            f"Largest improvement vs control: {format_entry(uplift_entry)} -> {uplift_value:+.3f} D/yr"
        )

    leaderboard = summarize_by_treatment(records)
    if leaderboard:
        print("Treatment leaderboard (higher mean = slower progression):")
        for idx, row in enumerate(leaderboard[:3], start=1):
            print(
                f"  {idx}. {row['name']:<20} {row['mean']:+.3f} D/yr  n={int(row['total_n'])}  cohorts={row['cohorts']}"
            )

    if meta:
        print("Available dimensions:")
        if meta.get("ethnicities"):
            print("  Ethnicities: " + ", ".join(str(val) for val in meta["ethnicities"]))
        if meta.get("ages"):
            print("  Ages: " + ", ".join(str(val) for val in meta["ages"]))
        if meta.get("treatments"):
            print(
                "  Treatments: "
                + ", ".join(f"{t['name']} ({t.get('key', '-')})" for t in meta["treatments"])
            )


def parse_multi_choice(raw: str, options: Sequence[str], label: str) -> List[str]:
    if not options:
        return []
    normalized = {opt.lower(): opt for opt in options}
    index_lookup = {str(idx + 1): opt for idx, opt in enumerate(options)}
    if not raw:
        return list(options)
    cleaned_tokens: List[str] = []
    for chunk in raw.replace("，", ",").split(","):
        token = chunk.strip()
        if token:
            cleaned_tokens.append(token)
    selections: List[str] = []
    seen = set()
    for token in cleaned_tokens:
        lowered = token.lower()
        resolved = None
        if lowered in normalized:
            resolved = normalized[lowered]
        elif token in index_lookup:
            resolved = index_lookup[token]
        elif token == "*" or lowered == "all":
            return list(options)
        if resolved:
            if resolved not in seen:
                selections.append(resolved)
                seen.add(resolved)
        else:
            print(f"[WARN] Ignored unknown {label} token: {token}")
    return selections or list(options)


def parse_age_selection(raw: str, available: Sequence[int]) -> List[int]:
    if not available:
        return []
    available_set = {int(v) for v in available}
    if not raw:
        return sorted(available_set)
    tokens = [token.strip() for token in raw.replace("，", ",").split(",") if token.strip()]
    chosen: set[int] = set()
    for token in tokens:
        if token.lower() in {"all", "*"}:
            return sorted(available_set)
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            try:
                start_val = int(start_str)
                end_val = int(end_str)
            except ValueError:
                print(f"[WARN] Could not parse age range: {token}")
                continue
            if start_val > end_val:
                start_val, end_val = end_val, start_val
            for value in range(start_val, end_val + 1):
                if value in available_set:
                    chosen.add(value)
            continue
        try:
            age_val = int(token)
        except ValueError:
            print(f"[WARN] Could not parse age token: {token}")
            continue
        if age_val in available_set:
            chosen.add(age_val)
        else:
            print(f"[WARN] Age {age_val} is outside the available range")
    return sorted(chosen) or sorted(available_set)


def prompt_filters(meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ethnicities = list(meta.get("ethnicities") or [])
    ages = list(meta.get("ages") or [])
    treatments = [item.get("name") for item in meta.get("treatments") or [] if item.get("name")]

    print("\nFilter builder (type 'q' to quit at any prompt).")
    if ethnicities:
        print("Ethnicities:")
        for idx, eth in enumerate(ethnicities, start=1):
            print(f"  {idx}. {eth}")
    eth_raw = input("Select ethnicity (comma, enter for all): ").strip()
    if eth_raw.lower() in {"q", "quit", "exit"}:
        return None
    selected_eth = parse_multi_choice(eth_raw, ethnicities, "ethnicity")

    if ages:
        print("Available ages: " + ", ".join(str(val) for val in ages))
    age_raw = input("Select age or range (8-12, enter for all): ").strip()
    if age_raw.lower() in {"q", "quit", "exit"}:
        return None
    selected_ages = parse_age_selection(age_raw, ages)

    if treatments:
        print("Treatments:")
        for idx, treatment in enumerate(treatments, start=1):
            print(f"  {idx}. {treatment}")
    treat_raw = input("Select treatments (comma, enter for all): ").strip()
    if treat_raw.lower() in {"q", "quit", "exit"}:
        return None
    selected_treatments = parse_multi_choice(treat_raw, treatments, "treatment")

    top_raw = input("How many rows should be displayed? [10]: ").strip()
    if top_raw.lower() in {"q", "quit", "exit"}:
        return None
    try:
        top_value = int(top_raw) if top_raw else 10
    except ValueError:
        top_value = 10

    print("Sort mode options: 1) slow progression first  2) fast progression first  3) biggest gain vs control")
    sort_raw = input("Choose sort option [1]: ").strip()
    sort_map = {"1": "best", "2": "worst", "3": "delta"}
    sort_mode = sort_map.get(sort_raw, "best")

    return {
        "ethnicities": selected_eth,
        "ages": selected_ages,
        "treatments": selected_treatments,
        "top_k": max(top_value, 1),
        "sort_mode": sort_mode,
    }


def filter_records(records: Sequence[Record], selection: Dict[str, Any]) -> List[Record]:
    eth_allowed = set(selection.get("ethnicities") or [])
    age_allowed = {int(val) for val in selection.get("ages") or []}
    treatment_allowed = set(selection.get("treatments") or [])

    result: List[Record] = []
    for rec in records:
        ethnicity = rec.get("ethnicity")
        age = rec.get("age")
        treatment = rec.get("treatment")
        if eth_allowed and ethnicity not in eth_allowed:
            continue
        if age_allowed and age not in age_allowed:
            continue
        if treatment_allowed and treatment not in treatment_allowed:
            continue
        result.append(rec)
    return result


def sort_records(records: Sequence[Record], sort_mode: str, controls: Dict[Tuple[str, int], float]) -> List[Record]:
    mode = sort_mode or "best"
    if mode == "worst":
        return sorted(records, key=lambda rec: rec.get("mean", float("inf")))
    if mode == "delta":
        return sorted(
            records,
            key=lambda rec: compute_delta(rec, controls) if compute_delta(rec, controls) is not None else float("-inf"),
            reverse=True,
        )
    return sorted(records, key=lambda rec: rec.get("mean", float("-inf")), reverse=True)


def render_table(records: Sequence[Record], controls: Dict[Tuple[str, int], float], limit: int) -> None:
    columns = [
        "Age",
        "Ethnicity",
        "Treatment",
        "Mean (D/yr)",
        "SD",
        "Total N",
        "Records",
        "Delta vs Ctrl",
        "Sources",
    ]
    rows: List[List[str]] = []
    for rec in records[: max(limit, 1)]:
        mean = rec.get("mean")
        mean_text = "n/a" if mean is None else f"{float(mean):+.3f}"
        sd = rec.get("sd")
        sd_text = "n/a" if sd is None else f"{float(sd):.3f}"
        total_n = rec.get("total_n") or rec.get("n") or "-"
        records_used = rec.get("records_used") or "-"
        delta = compute_delta(rec, controls)
        delta_text = "--" if delta is None else f"{delta:+.3f}"
        rows.append(
            [
                str(rec.get("age")),
                str(rec.get("ethnicity")),
                str(rec.get("treatment")),
                mean_text,
                sd_text,
                str(total_n),
                str(records_used),
                delta_text,
                format_sources(rec),
            ]
        )
    if not rows:
        print("No rows to display.")
        return
    col_widths = [len(col) for col in columns]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))
    header = " | ".join(col.ljust(col_widths[idx]) for idx, col in enumerate(columns))
    divider = "-+-".join("-" * width for width in col_widths)
    print("\n" + header)
    print(divider)
    for row in rows:
        print(" | ".join(row[idx].ljust(col_widths[idx]) for idx in range(len(columns))))


def run_and_render(records: Sequence[Record], controls: Dict[Tuple[str, int], float], selection: Dict[str, Any]) -> None:
    filtered = filter_records(records, selection)
    if not filtered:
        print("\nNo cohorts matched your filters.")
        return
    print("\n" + describe_selection(selection))
    sorted_rows = sort_records(filtered, selection.get("sort_mode", "best"), controls)
    render_table(sorted_rows, controls, selection.get("top_k", 10))
    stats = aggregated_stats(filtered)
    if stats:
        print(
            f"\nWeighted mean across selection: {stats['mean']:+.3f} D/yr | "
            f"Pooled SD: {stats['sd']:.3f} | Total n={int(stats['total_n'])}"
        )


def interactive_browser(records: Sequence[Record], meta: Dict[str, Any], controls: Dict[Tuple[str, int], float]) -> None:
    if not records:
        print("No data available for interactive browsing.")
        return
    print(
        textwrap.fill(
            "Interactive mode lets you quickly slice the mined cohorts. Use comma-separated "
            "values, ranges (8-12), or press Enter to keep everything. Type 'q' to exit.",
            width=88,
        )
    )
    while True:
        selection = prompt_filters(meta)
        if selection is None:
            print("Leaving interactive mode.")
            break
        run_and_render(records, controls, selection)
        again = input("\nPress Enter for another view or type 'q' to quit: ").strip().lower()
        if again in {"q", "quit", "exit"}:
            break


def resolve_cli_labels(raw_values: Optional[List[str]], options: Sequence[str], label: str) -> List[str]:
    if not options:
        return []
    if not raw_values:
        return list(options)
    combined = ",".join(raw_values)
    return parse_multi_choice(combined, options, label)


def build_cli_selection(args: argparse.Namespace, meta: Dict[str, Any]) -> Dict[str, Any]:
    ethnicities = list(meta.get("ethnicities") or [])
    ages = list(meta.get("ages") or [])
    treatments = [item.get("name") for item in meta.get("treatments") or [] if item.get("name")]
    selection = {
        "ethnicities": resolve_cli_labels(args.ethnicity, ethnicities, "ethnicity"),
        "ages": parse_age_selection(args.ages or "", ages) if ages else [],
        "treatments": resolve_cli_labels(args.treatment, treatments, "treatment"),
        "top_k": max(args.top, 1),
        "sort_mode": args.sort,
    }
    if not selection["ages"] and ages:
        selection["ages"] = list(ages)
    return selection


def ensure_meta_defaults(meta: Dict[str, Any], records: Sequence[Record]) -> Dict[str, Any]:
    if not meta.get("ethnicities"):
        meta["ethnicities"] = sorted(
            {rec.get("ethnicity") for rec in records if rec.get("ethnicity")}
        )
    if not meta.get("ages"):
        meta["ages"] = sorted({int(rec.get("age")) for rec in records if rec.get("age") is not None})
    if not meta.get("treatments"):
        treat_map: Dict[str, str] = {}
        for rec in records:
            name = rec.get("treatment")
            key = rec.get("treatment_key") or name
            if name:
                treat_map[name] = str(key)
        meta["treatments"] = [
            {"name": name, "key": treat_map[name]} for name in sorted(treat_map)
        ]
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display and explore aggregated mining results with quick filters."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to artifacts/medical_config.json (default: repo artifacts folder).",
    )
    parser.add_argument(
        "--ethnicity",
        action="append",
        help="Filter by ethnicity (can be passed multiple times or as comma-separated values).",
    )
    parser.add_argument(
        "--treatment",
        action="append",
        help="Filter by treatment name (multiple values allowed).",
    )
    parser.add_argument(
        "--ages",
        type=str,
        help="Age selector, e.g. '8-12' or '8,10,12'.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Maximum rows to show per view (default: 10).",
    )
    parser.add_argument(
        "--sort",
        choices=["best", "worst", "delta"],
        default="best",
        help="Sort order for the initial view.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip the interactive prompts and only print the initial view.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    payload = load_payload(config_path)
    base_rates = payload.get("base_rates") or []
    meta = ensure_meta_defaults(payload.get("meta_schema") or {}, base_rates)
    controls = build_control_lookup(base_rates)
    print_dataset_summary(config_path, payload, base_rates, controls)
    selection = build_cli_selection(args, meta)
    run_and_render(base_rates, controls, selection)
    if not args.no_interactive:
        interactive_browser(base_rates, meta, controls)


if __name__ == "__main__":
    main()
