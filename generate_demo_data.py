"""Génère un faux historique pour vérifier l'installation, jamais pour décider un vrai devis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=Path("data/demo_history.csv"), type=Path)
    parser.add_argument("--rows", default=900, type=int)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    n = args.rows
    created = pd.date_range("2023-01-01", "2026-06-30", periods=n)
    lead = rng.integers(20, 480, n)
    event_date = created + pd.to_timedelta(lead, unit="D")

    photobooth = rng.binomial(1, 0.38, n)
    mirror = rng.binomial(1, 0.32, n)
    booth_360 = rng.binomial(1, 0.32, n)
    vogue = rng.binomial(1, 0.17, n)
    empty = (photobooth + mirror + booth_360 + vogue) == 0
    photobooth[empty] = 1

    phone = rng.binomial(1, 0.20, n)
    video_book = rng.binomial(1, 0.16, n)
    floral = rng.binomial(1, 0.14, n)
    premium = np.clip(rng.normal(48, 24, n), 0, 100)
    reviews = np.maximum(0, rng.lognormal(3.7, 1.1, n).astype(int))
    rating = np.clip(rng.normal(4.35, 0.35, n), 2.5, 5)

    gross = (
        450 * photobooth
        + 550 * mirror
        + 500 * booth_360
        + 600 * vogue
        + 50 * phone
        + 200 * video_book
        + 50 * floral
    )
    ratio = np.clip(rng.normal(0.98, 0.13, n), 0.68, 1.35)
    proposed = np.round((gross * ratio - 90) / 100) * 100 + 90
    proposed = np.maximum(proposed, 290)

    source = rng.choice(
        ["Google", "Instagram", "Recommandation", "Salon", "Autre"],
        size=n,
        p=[0.34, 0.27, 0.22, 0.09, 0.08],
    )
    returning = rng.binomial(1, 0.12, n)
    business = rng.binomial(1, 0.18, n)
    venue_type = rng.choice(
        ["wedding_venue", "event_venue", "hotel", "restaurant", "INCONNU"],
        size=n,
        p=[0.28, 0.27, 0.14, 0.12, 0.19],
    )

    logit = (
        -0.75
        - 7.5 * (proposed / gross - 0.92)
        + 0.022 * premium
        + 0.95 * returning
        + 0.70 * (source == "Recommandation")
        + 0.25 * business
        + 0.0012 * np.minimum(lead, 300)
        - 0.35 * (mirror + booth_360 + vogue >= 3)
        + rng.normal(0, 0.30, n)
    )
    probability = 1 / (1 + np.exp(-logit))
    signed = rng.binomial(1, probability)

    frame = pd.DataFrame(
        {
            "request_event_id": np.arange(1, n + 1),
            "request_created_at_utc": created,
            "request_source": source,
            "request_is_business": business,
            "prior_request_count": returning * rng.integers(1, 4, n),
            "prior_signed_count": returning * rng.integers(0, 3, n),
            "prior_is_returning_client": returning,
            "event_date": event_date.date,
            "event_month": event_date.month,
            "event_quarter": event_date.quarter,
            "event_weekday_iso": event_date.dayofweek + 1,
            "event_is_weekend": (event_date.dayofweek >= 5).astype(int),
            "event_lead_time_days": lead,
            "event_postal_code": rng.choice(["75000", "77000", "78000", "91000"], n),
            "event_city": rng.choice(["paris", "meaux", "versailles", "evry"], n),
            "enrichment_event_full_address_raw": [
                f"Salle de démonstration {i % 40}, France" for i in range(n)
            ],
            "enrichment_address_available": 1,
            "event_has_schedule": rng.binomial(1, 0.84, n),
            "event_has_client_comment": rng.binomial(1, 0.66, n),
            "event_client_comment_length": rng.integers(0, 500, n),
            "event_has_internal_comment": rng.binomial(1, 0.42, n),
            "event_internal_comment_length": rng.integers(0, 300, n),
            "event_venue_anonymous_key": [f"DEMO_{i % 40:03d}" for i in range(n)],
            "product_photobooth": photobooth,
            "product_miroirbooth": mirror,
            "product_videobooth": booth_360,
            "product_voguebooth": vogue,
            "product_ipadbooth": 0,
            "product_airbooth": 0,
            "product_count": photobooth + mirror + booth_360 + vogue,
            "option_mur_floral": floral,
            "option_phonebooth": phone,
            "option_livre_or": 0,
            "option_fond_360": 0,
            "option_panneau_bienvenue": 0,
            "option_photographe_voguebooth": 0,
            "option_impression_voguebooth": 0,
            "option_decor_voguebooth": 0,
            "option_holo_3d": 0,
            "option_panneau_fontaine": 0,
            "option_video_livre_or": video_book,
            "option_magnets_quantity": 0,
            "option_porte_cles_quantity": 0,
            "option_magnets_simple_quantity": 0,
            "option_delivery": 0,
            "option_duration_hours": 5,
            "option_count": floral + phone + video_book,
            "price_catalog_gross": gross,
            "price_product_discount": 0,
            "price_global_discount": np.maximum(gross - proposed, 0),
            "price_option_discount": 0,
            "price_total_declared_discount": np.maximum(gross - proposed, 0),
            "price_proposed": proposed,
            "price_proposed_minus_gross": proposed - gross,
            "price_proposed_to_gross_ratio": proposed / gross,
            "venue_match_confidence": rng.uniform(0.62, 0.99, n),
            "venue_is_private_address": (venue_type == "INCONNU").astype(int),
            "venue_type": venue_type,
            "venue_capacity_seated": np.clip(rng.normal(160, 80, n), 30, 600),
            "venue_capacity_cocktail": np.clip(rng.normal(230, 110, n), 40, 900),
            "venue_rating": rating,
            "venue_review_count": reviews,
            "venue_has_official_website": rng.binomial(1, 0.82, n),
            "venue_premium_score": premium,
            "venue_notability_score": np.clip(
                np.log10(reviews + 1) * 24 + (rating - 3.5) * 20, 0, 100
            ),
            "venue_enrichment_sources_count": rng.integers(1, 3, n),
            "target_signed": signed,
            "target_training_eligible": 1,
            "target_price_training_eligible": 1,
            "outcome_class": np.where(signed == 1, "SIGNED", "REFUSED"),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"{len(frame)} fausses lignes écrites dans {args.output}")


if __name__ == "__main__":
    main()
