"""Contrôle simple de l'export SQL avant entraînement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def number(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def value_counts(frame: pd.DataFrame, column: str, limit: int = 20) -> dict:
    if column not in frame:
        return {}
    counts = frame[column].fillna("VIDE").astype(str).value_counts().head(limit)
    return {str(key): int(value) for key, value in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.data, sep=None, engine="python")
    created = pd.to_datetime(frame.get("request_created_at_utc"), errors="coerce", utc=True)
    target = number(frame, "target_signed")
    eligible = number(frame, "target_price_training_eligible") == 1
    price = number(frame, "price_proposed")
    gross = number(frame, "price_catalog_gross")
    lead_time = number(frame, "event_lead_time_days")

    training = frame.loc[eligible & target.isin([0, 1]) & price.notna()].copy()
    training_target = pd.to_numeric(training["target_signed"], errors="coerce")
    training_price = pd.to_numeric(training["price_proposed"], errors="coerce")

    important_columns = [
        "request_created_at_utc",
        "request_source",
        "event_date",
        "event_postal_code",
        "event_city",
        "event_venue_anonymous_key",
        "price_catalog_gross",
        "price_proposed",
        "target_signed",
        "outcome_class",
    ]
    missing = {}
    for column in important_columns:
        if column in frame:
            missing[column] = {
                "count": int(frame[column].isna().sum()),
                "rate": round(float(frame[column].isna().mean()), 4),
            }

    report = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "duplicate_event_ids": int(
            frame["request_event_id"].duplicated().sum()
            if "request_event_id" in frame
            else 0
        ),
        "request_date_min": created.min().isoformat() if created.notna().any() else None,
        "request_date_max": created.max().isoformat() if created.notna().any() else None,
        "training_eligible_rows": int(len(training)),
        "training_signed_rows": int((training_target == 1).sum()),
        "training_not_signed_rows": int((training_target == 0).sum()),
        "training_signature_rate": round(float(training_target.mean()), 4),
        "price_proposed": {
            "missing_count": int(price.isna().sum()),
            "minimum": float(training_price.min()),
            "median": float(training_price.median()),
            "mean": round(float(training_price.mean()), 2),
            "maximum": float(training_price.max()),
        },
        "data_quality": {
            "negative_lead_time_rows": int((lead_time < 0).sum()),
            "nonpositive_proposed_price_rows": int((price <= 0).sum()),
            "nonpositive_catalog_price_rows": int((gross <= 0).sum()),
            "very_low_price_ratio_rows": int(((price / gross) < 0.5).sum()),
            "very_high_price_ratio_rows": int(((price / gross) > 1.5).sum()),
            "address_available_rows": int(
                (number(frame, "enrichment_address_available") == 1).sum()
            ),
            "unique_venue_keys": int(
                frame["event_venue_anonymous_key"].nunique(dropna=True)
                if "event_venue_anonymous_key" in frame
                else 0
            ),
        },
        "outcomes": value_counts(frame, "outcome_class"),
        "statuses": value_counts(frame, "outcome_current_status"),
        "sources": value_counts(frame, "request_source"),
        "missing_important_columns": missing,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
