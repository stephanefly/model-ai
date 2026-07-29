"""Préparation des variables utilisables avant l'envoi du devis."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


CATEGORICAL_FEATURES = {
    "request_source",
    "event_postal_code",
    "event_city",
    "event_venue_anonymous_key",
    "option_mur_floral_style",
    "venue_type",
}

EXACT_NUMERIC_FEATURES = {
    "request_is_business",
    "prior_request_count",
    "prior_signed_count",
    "prior_is_returning_client",
    "event_month",
    "event_quarter",
    "event_weekday_iso",
    "event_is_weekend",
    "event_lead_time_days",
    "event_has_client_comment",
    "event_client_comment_length",
    "enrichment_address_available",
    "product_count",
    "option_count",
    "price_catalog_gross",
    "price_product_discount",
    "price_global_discount",
    "price_option_discount",
    "price_total_declared_discount",
    "price_proposed",
    "price_proposed_minus_gross",
    "price_proposed_to_gross_ratio",
    "venue_match_confidence",
    "venue_is_private_address",
    "venue_capacity_seated",
    "venue_capacity_cocktail",
    "venue_rating",
    "venue_review_count",
    "venue_price_min",
    "venue_price_max",
    "venue_has_official_website",
    "venue_premium_score",
    "venue_notability_score",
    "venue_enrichment_sources_count",
}


def _numeric_series(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").fillna(default)


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables calculées de façon identique à l'entraînement et en production."""
    data = frame.copy()

    if "event_date" in data.columns:
        # Conversion en UTC puis suppression du fuseau : cela évite l'erreur
        # Windows "tz-naive and tz-aware" lors de la soustraction des dates.
        event_date = pd.to_datetime(
            data["event_date"],
            errors="coerce",
            utc=True,
        ).dt.tz_localize(None)
        if "event_month" not in data:
            data["event_month"] = event_date.dt.month
        if "event_quarter" not in data:
            data["event_quarter"] = event_date.dt.quarter
        if "event_weekday_iso" not in data:
            data["event_weekday_iso"] = event_date.dt.isocalendar().day.astype("Float64")
        if "event_is_weekend" not in data:
            data["event_is_weekend"] = (event_date.dt.weekday >= 5).astype(float)

        if "event_lead_time_days" not in data:
            if "request_created_at_utc" in data:
                request_date = pd.to_datetime(
                    data["request_created_at_utc"],
                    errors="coerce",
                    utc=True,
                ).dt.tz_localize(None)
            else:
                today_utc = (
                    pd.Timestamp.now(tz="UTC")
                    .normalize()
                    .tz_localize(None)
                )
                request_date = pd.Series(
                    today_utc,
                    index=data.index,
                )
            data["event_lead_time_days"] = (event_date - request_date).dt.days

    product_columns = [
        name
        for name in data.columns
        if name.startswith("product_") and name != "product_count"
    ]
    if product_columns and "product_count" not in data:
        data["product_count"] = sum(_numeric_series(data, name) for name in product_columns)

    binary_option_columns = [
        "option_mur_floral",
        "option_phonebooth",
        "option_livre_or",
        "option_fond_360",
        "option_panneau_bienvenue",
        "option_photographe_voguebooth",
        "option_impression_voguebooth",
        "option_decor_voguebooth",
        "option_holo_3d",
        "option_panneau_fontaine",
        "option_video_livre_or",
    ]
    quantity_option_columns = [
        "option_magnets_quantity",
        "option_porte_cles_quantity",
        "option_magnets_simple_quantity",
    ]
    if "option_count" not in data:
        option_count = sum(_numeric_series(data, name) for name in binary_option_columns)
        option_count += sum(
            (_numeric_series(data, name) > 0).astype(int) for name in quantity_option_columns
        )
        data["option_count"] = option_count

    if {"price_proposed", "price_catalog_gross"}.issubset(data.columns):
        proposed = pd.to_numeric(data["price_proposed"], errors="coerce")
        gross = pd.to_numeric(data["price_catalog_gross"], errors="coerce")
        data["price_proposed_minus_gross"] = proposed - gross
        data["price_proposed_to_gross_ratio"] = proposed / gross.replace(0, np.nan)

    for column in data.columns:
        if column in EXACT_NUMERIC_FEATURES or column.startswith(("product_", "option_")):
            if column not in CATEGORICAL_FEATURES:
                data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in CATEGORICAL_FEATURES.intersection(data.columns):
        data[column] = data[column].fillna("INCONNU").astype(str)

    return data


def select_feature_columns(columns: Iterable[str]) -> tuple[list[str], list[str]]:
    """Retourne uniquement les informations connues avant la proposition."""
    available = set(columns)
    categoricals = sorted(CATEGORICAL_FEATURES.intersection(available))

    numerics = set(EXACT_NUMERIC_FEATURES.intersection(available))
    numerics.update(
        name
        for name in available
        if name.startswith(("product_", "option_"))
        and name not in CATEGORICAL_FEATURES
        and not name.endswith("_style")
    )
    numerics = sorted(numerics)

    # Les adresses, commentaires bruts, identifiants et résultats ne sont jamais sélectionnés.
    return numerics, categoricals


def align_features(
    frame: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]
) -> pd.DataFrame:
    """Crée les colonnes manquantes pour accepter une demande CRM partielle."""
    data = prepare_features(frame)
    for column in numeric_features:
        if column not in data:
            data[column] = np.nan
    for column in categorical_features:
        if column not in data:
            data[column] = "INCONNU"
    return data[numeric_features + categorical_features]
