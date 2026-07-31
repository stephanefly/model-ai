"""Préparation des variables utilisables avant l'envoi du devis."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


CATEGORICAL_FEATURES = {
    "request_source",
    "request_daypart",
    "event_postal_code",
    "event_city",
    "event_venue_anonymous_key",
    "event_lead_time_bucket",
    "product_bundle_signature",
    "option_mur_floral_style",
    "venue_type",
}

EXACT_NUMERIC_FEATURES = {
    "request_is_business",
    "request_month",
    "request_weekday_iso",
    "request_is_weekend",
    "request_hour",
    "request_is_outside_business_hours",
    "prior_request_count",
    "prior_signed_count",
    "prior_is_returning_client",
    "prior_days_since_last_request",
    "prior_signature_rate",
    "event_month",
    "event_quarter",
    "event_weekday_iso",
    "event_is_weekend",
    "event_lead_time_days",
    "event_is_very_last_minute",
    "event_is_last_minute",
    "event_is_long_lead",
    "event_has_schedule",
    "event_has_client_comment",
    "event_client_comment_length",
    "event_request_completeness_score",
    "event_missing_information_count",
    "enrichment_address_available",
    "product_count",
    "product_is_multi",
    "product_premium_count",
    "product_has_premium",
    "option_count",
    "option_quantity_total",
    "request_total_items",
    "price_catalog_gross",
    "price_product_discount",
    "price_global_discount",
    "price_option_discount",
    "price_total_declared_discount",
    "price_discount_rate",
    "price_proposed",
    "price_proposed_minus_gross",
    "price_proposed_to_gross_ratio",
    "venue_match_confidence",
    "venue_is_private_address",
    "venue_capacity_seated",
    "venue_capacity_cocktail",
    "venue_capacity_max",
    "venue_rating",
    "venue_review_count",
    "venue_price_min",
    "venue_price_max",
    "venue_price_midpoint",
    "venue_has_official_website",
    "venue_premium_score",
    "venue_notability_score",
    "venue_enrichment_sources_count",
    # Critères opérationnels disponibles avant le premier devis. Ils seront
    # sélectionnés automatiquement dès que l'export historique les contient.
    "bookings_on_date",
    "date_requested_product_bookings",
    "date_utilization_ratio",
    "available_requested_product_ratio",
    "event_distance_km",
    "event_travel_time_minutes",
    "event_toll_cost_eur",
}

PRODUCT_LABELS = {
    "product_photobooth": "PHOTOBOOTH",
    "product_miroirbooth": "MIROIRBOOTH",
    "product_videobooth": "BOOTH_360",
    "product_voguebooth": "VOGUEBOOTH",
    "product_ipadbooth": "IPADBOOTH",
    "product_airbooth": "AIRBOOTH",
}

PREMIUM_PRODUCT_COLUMNS = {
    "product_miroirbooth",
    "product_videobooth",
    "product_voguebooth",
}

BINARY_OPTION_COLUMNS = [
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

QUANTITY_OPTION_COLUMNS = [
    "option_magnets_quantity",
    "option_porte_cles_quantity",
    "option_magnets_simple_quantity",
]


def _numeric_series(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").fillna(default)


def _raw_numeric_series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _build_request_time_features(data: pd.DataFrame) -> None:
    if "request_created_at_utc" not in data.columns:
        return

    request_date = pd.to_datetime(
        data["request_created_at_utc"],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    if "request_month" not in data:
        data["request_month"] = request_date.dt.month
    if "request_weekday_iso" not in data:
        data["request_weekday_iso"] = request_date.dt.isocalendar().day.astype("Float64")
    if "request_is_weekend" not in data:
        data["request_is_weekend"] = (request_date.dt.weekday >= 5).astype(float)
    if "request_hour" not in data:
        data["request_hour"] = request_date.dt.hour
    if "request_is_outside_business_hours" not in data:
        hour = request_date.dt.hour
        data["request_is_outside_business_hours"] = (
            (hour < 8) | (hour >= 19)
        ).astype(float)
    if "request_daypart" not in data:
        hour = request_date.dt.hour
        daypart = pd.Series("INCONNU", index=data.index, dtype="object")
        daypart.loc[hour.between(0, 5)] = "NUIT"
        daypart.loc[hour.between(6, 11)] = "MATIN"
        daypart.loc[hour.between(12, 17)] = "APRES_MIDI"
        daypart.loc[hour.between(18, 23)] = "SOIR"
        daypart.loc[request_date.isna()] = "INCONNU"
        data["request_daypart"] = daypart


def _build_event_time_features(data: pd.DataFrame) -> None:
    if "event_date" in data.columns:
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
                request_date = pd.Series(today_utc, index=data.index)
            data["event_lead_time_days"] = (
                event_date.dt.normalize() - request_date.dt.normalize()
            ).dt.days

    lead_time = _raw_numeric_series(data, "event_lead_time_days")
    if "event_is_very_last_minute" not in data:
        data["event_is_very_last_minute"] = lead_time.between(0, 13).astype(float)
    if "event_is_last_minute" not in data:
        data["event_is_last_minute"] = lead_time.between(0, 29).astype(float)
    if "event_is_long_lead" not in data:
        data["event_is_long_lead"] = (lead_time > 180).astype(float)
    if "event_lead_time_bucket" not in data:
        bucket = pd.Series("INCONNU", index=data.index, dtype="object")
        bucket.loc[lead_time < 0] = "DATE_PASSEE_OU_INVALIDE"
        bucket.loc[lead_time.between(0, 7)] = "0_7_JOURS"
        bucket.loc[lead_time.between(8, 14)] = "8_14_JOURS"
        bucket.loc[lead_time.between(15, 30)] = "15_30_JOURS"
        bucket.loc[lead_time.between(31, 60)] = "31_60_JOURS"
        bucket.loc[lead_time.between(61, 120)] = "61_120_JOURS"
        bucket.loc[lead_time.between(121, 180)] = "121_180_JOURS"
        bucket.loc[lead_time > 180] = "PLUS_180_JOURS"
        data["event_lead_time_bucket"] = bucket


def _build_prior_client_features(data: pd.DataFrame) -> None:
    previous_requests = _numeric_series(data, "prior_request_count")
    previous_signed = _numeric_series(data, "prior_signed_count")

    if "prior_signature_rate" not in data:
        data["prior_signature_rate"] = (
            previous_signed / previous_requests.replace(0, np.nan)
        ).fillna(0.0)

    if (
        "prior_days_since_last_request" not in data
        and "request_created_at_utc" in data.columns
        and "prior_last_request_at_utc" in data.columns
    ):
        current = pd.to_datetime(
            data["request_created_at_utc"], errors="coerce", utc=True
        )
        previous = pd.to_datetime(
            data["prior_last_request_at_utc"], errors="coerce", utc=True
        )
        delta_days = (current - previous).dt.total_seconds() / 86400
        data["prior_days_since_last_request"] = delta_days.clip(lower=0)


def _build_product_features(data: pd.DataFrame) -> None:
    product_columns = [name for name in PRODUCT_LABELS if name in data.columns]
    if product_columns and "product_count" not in data:
        data["product_count"] = sum(
            _numeric_series(data, name) for name in product_columns
        )

    product_count = _numeric_series(data, "product_count")
    if "product_is_multi" not in data:
        data["product_is_multi"] = (product_count >= 2).astype(float)

    premium_columns = [name for name in PREMIUM_PRODUCT_COLUMNS if name in data.columns]
    premium_count = sum(
        (_numeric_series(data, name) > 0).astype(int) for name in premium_columns
    ) if premium_columns else pd.Series(0, index=data.index, dtype=float)
    if "product_premium_count" not in data:
        data["product_premium_count"] = premium_count
    if "product_has_premium" not in data:
        data["product_has_premium"] = (premium_count > 0).astype(float)

    if "product_bundle_signature" not in data:
        selected = []
        for index in data.index:
            labels = [
                label
                for column, label in PRODUCT_LABELS.items()
                if _numeric_series(data.loc[[index]], column).iloc[0] > 0
            ]
            selected.append("+".join(labels) if labels else "AUCUN_PRODUIT")
        data["product_bundle_signature"] = selected


def _build_option_features(data: pd.DataFrame) -> None:
    if "option_count" not in data:
        option_count = sum(_numeric_series(data, name) for name in BINARY_OPTION_COLUMNS)
        option_count += sum(
            (_numeric_series(data, name) > 0).astype(int)
            for name in QUANTITY_OPTION_COLUMNS
        )
        data["option_count"] = option_count

    if "option_quantity_total" not in data:
        data["option_quantity_total"] = sum(
            _numeric_series(data, name) for name in QUANTITY_OPTION_COLUMNS
        )

    if "request_total_items" not in data:
        data["request_total_items"] = _numeric_series(
            data, "product_count"
        ) + _numeric_series(data, "option_count")


def _build_completeness_features(data: pd.DataFrame) -> None:
    address = (_numeric_series(data, "enrichment_address_available") > 0).astype(int)
    schedule = (_numeric_series(data, "event_has_schedule") > 0).astype(int)
    comment = (_numeric_series(data, "event_has_client_comment") > 0).astype(int)
    product = (_numeric_series(data, "product_count") > 0).astype(int)
    completed = address + schedule + comment + product

    if "event_request_completeness_score" not in data:
        data["event_request_completeness_score"] = completed / 4.0
    if "event_missing_information_count" not in data:
        data["event_missing_information_count"] = 4 - completed


def _build_price_features(data: pd.DataFrame) -> None:
    if {"price_proposed", "price_catalog_gross"}.issubset(data.columns):
        proposed = pd.to_numeric(data["price_proposed"], errors="coerce")
        gross = pd.to_numeric(data["price_catalog_gross"], errors="coerce")
        data["price_proposed_minus_gross"] = proposed - gross
        data["price_proposed_to_gross_ratio"] = proposed / gross.replace(0, np.nan)

    if {"price_total_declared_discount", "price_catalog_gross"}.issubset(data.columns):
        discount = pd.to_numeric(
            data["price_total_declared_discount"], errors="coerce"
        )
        gross = pd.to_numeric(data["price_catalog_gross"], errors="coerce")
        data["price_discount_rate"] = discount / gross.replace(0, np.nan)


def _build_venue_features(data: pd.DataFrame) -> None:
    seated = _raw_numeric_series(data, "venue_capacity_seated")
    cocktail = _raw_numeric_series(data, "venue_capacity_cocktail")
    if "venue_capacity_max" not in data:
        data["venue_capacity_max"] = pd.concat(
            [seated, cocktail], axis=1
        ).max(axis=1, skipna=True)

    minimum = _raw_numeric_series(data, "venue_price_min")
    maximum = _raw_numeric_series(data, "venue_price_max")
    if "venue_price_midpoint" not in data:
        data["venue_price_midpoint"] = pd.concat(
            [minimum, maximum], axis=1
        ).mean(axis=1, skipna=True)


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables calculées de façon identique à l'entraînement et en production."""
    data = frame.copy()

    _build_request_time_features(data)
    _build_event_time_features(data)
    _build_prior_client_features(data)
    _build_product_features(data)
    _build_option_features(data)
    _build_completeness_features(data)
    _build_price_features(data)
    _build_venue_features(data)

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

    # Les adresses, commentaires bruts, identifiants, ouvertures d'e-mail,
    # relances et résultats ne sont jamais sélectionnés.
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
