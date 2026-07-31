"""Méta-modèle tarifaire pour remplir les dates sans sacrifier la marge minimale.

Cette couche s'appuie sur ``quote_optimizer.py`` :
- elle vérifie d'abord la disponibilité des produits à la date demandée ;
- elle détecte les dates vides proches ;
- elle privilégie alors la probabilité de signature, mais uniquement parmi les
  prix qui respectent les garde-fous du moteur tarifaire principal.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime
from pathlib import Path

import joblib

from quote_optimizer import load_json, optimize as base_optimize, read_catalogue


PRODUCT_FIELDS = {
    "product_photobooth": "PHOTOBOOTH",
    "product_miroirbooth": "MIROIRBOOTH",
    "product_videobooth": "BOOTH_360",
    "product_voguebooth": "VOGUEBOOTH",
    "product_ipadbooth": "IPADBOOTH",
    "product_airbooth": "AIRBOOTH",
}

VALID_PRIORITIES = {
    "fill_date",
    "balanced",
    "maximize_expected_margin",
}


def _number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: object, default: int = 0) -> int:
    return int(_number(value, default))


def _parse_date(value: object) -> date | None:
    if not value:
        return None

    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def calculate_days_before_event(request: dict) -> int | None:
    """Retourne le délai avant l'événement de manière reproductible."""
    explicit = request.get("days_before_event")
    if explicit is None:
        explicit = request.get("event_lead_time_days")
    if explicit is not None:
        return max(_integer(explicit), 0)

    event_date = _parse_date(request.get("event_date"))
    request_date = _parse_date(
        request.get("request_created_at_utc") or request.get("request_date")
    )

    if event_date and request_date:
        return max((event_date - request_date).days, 0)
    if event_date:
        return max((event_date - date.today()).days, 0)
    return None


def determine_pricing_strategy(request: dict, config: dict) -> dict:
    """Détermine si la date doit être remplie ou valorisée normalement."""
    explicit_priority = str(
        request.get("commercial_priority") or ""
    ).strip().lower()

    bookings_were_provided = request.get("bookings_on_date") is not None
    bookings_on_date = max(_integer(request.get("bookings_on_date")), 0)
    days_before_event = calculate_days_before_event(request)
    date_status = str(request.get("date_booking_status") or "").strip().lower()

    if not date_status:
        if not bookings_were_provided:
            date_status = "unknown"
        elif bookings_on_date == 0:
            date_status = "empty"
        else:
            date_status = "occupied"

    if explicit_priority in VALID_PRIORITIES:
        priority = explicit_priority
        source = "explicit"
    else:
        date_is_empty = date_status == "empty" or (
            bookings_were_provided and bookings_on_date == 0
        )
        horizon = int(config.get("fill_date_horizon_days", 45))

        if date_is_empty and (
            days_before_event is None or days_before_event <= horizon
        ):
            priority = "fill_date"
        elif date_status in {"busy", "nearly_full", "full"}:
            priority = "maximize_expected_margin"
        else:
            priority = "balanced"
        source = "automatic"

    return {
        "pricing_objective": priority,
        "objective_source": source,
        "date_booking_status": date_status,
        "bookings_on_date": bookings_on_date,
        "days_before_event": days_before_event,
        "fill_date_mode": priority == "fill_date",
    }


def _available_quantity(available: dict, field: str, code: str) -> int | None:
    aliases = (
        code,
        code.lower(),
        field,
        field.removeprefix("product_"),
        field.removeprefix("product_").upper(),
    )
    for alias in aliases:
        if alias in available:
            return max(_integer(available[alias]), 0)
    return None


def analyze_product_availability(request: dict) -> dict:
    """Compare les produits demandés avec le stock encore libre à cette date."""
    raw_available = request.get("available_products")
    if not isinstance(raw_available, dict):
        return {
            "availability_status": "UNKNOWN",
            "requested_products_available": None,
            "available_products": {},
            "requested_products": [],
            "unavailable_requested_products": [],
            "alternative_products": [],
        }

    available_products: dict[str, int] = {}
    requested_products: list[dict] = []
    unavailable_products: list[dict] = []

    for field, code in PRODUCT_FIELDS.items():
        available_quantity = _available_quantity(raw_available, field, code)
        if available_quantity is not None:
            available_products[code] = available_quantity

        requested_quantity = math.ceil(max(_number(request.get(field)), 0))
        if requested_quantity == 0:
            continue

        product = {
            "code": code,
            "requested_quantity": requested_quantity,
            "available_quantity": available_quantity,
        }
        requested_products.append(product)

        if available_quantity is None or available_quantity < requested_quantity:
            unavailable_products.append(product)

    requested_codes = {product["code"] for product in requested_products}
    alternative_products = [
        {"code": code, "available_quantity": quantity}
        for code, quantity in available_products.items()
        if quantity > 0 and code not in requested_codes
    ]

    return {
        "availability_status": "KNOWN",
        "requested_products_available": not unavailable_products,
        "available_products": available_products,
        "requested_products": requested_products,
        "unavailable_requested_products": unavailable_products,
        "alternative_products": alternative_products,
    }


def _eligible_fill_date_candidates(result: dict) -> list[dict]:
    """Conserve uniquement les prix autorisés par les garde-fous existants."""
    minimum_margin = _number(result.get("minimum_direct_margin_eur"), 150)
    probability_floor = _number(
        result.get("minimum_acceptance_probability"), 0.60
    )
    gross_price = _number(result.get("price_catalog_gross_eur"))

    candidates = []
    for candidate in result.get("tested_candidates", []):
        probability = _number(candidate.get("signature_probability"))
        margin = _number(candidate.get("direct_margin_eur"))
        price = _number(candidate.get("price_eur"))

        if probability < probability_floor or margin < minimum_margin:
            continue

        # Même en mode remplissage, on conserve la limitation du moteur principal
        # quand l'effet historique du prix n'est pas suffisamment fiable.
        if (
            result.get("price_signal_reliable") is False
            and gross_price > 0
            and price > gross_price * 1.10
        ):
            continue

        candidates.append(candidate)

    return candidates


def select_fill_date_candidate(result: dict) -> dict | None:
    """Privilégie la conversion, puis la marge attendue, puis le prix le plus bas."""
    candidates = _eligible_fill_date_candidates(result)
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda candidate: (
            _number(candidate.get("signature_probability")),
            _number(candidate.get("expected_direct_margin_eur")),
            -_number(candidate.get("price_eur")),
        ),
    )


def _apply_selected_candidate(result: dict, candidate: dict) -> None:
    result["recommended_price_eur"] = candidate["price_eur"]
    result["estimated_signature_probability"] = candidate[
        "signature_probability"
    ]
    result["estimated_direct_margin_eur"] = candidate["direct_margin_eur"]
    result["estimated_expected_direct_margin_eur"] = candidate[
        "expected_direct_margin_eur"
    ]


def optimize_meta_price(
    request: dict,
    bundle: dict,
    catalogue,
    config: dict,
) -> dict:
    """Exécute la disponibilité, la stratégie de date, puis le moteur de prix."""
    availability = analyze_product_availability(request)
    strategy = determine_pricing_strategy(request, config)

    if availability["requested_products_available"] is False:
        return {
            "recommendation_status": "PRODUCT_UNAVAILABLE",
            "recommended_price_eur": None,
            "event_date": request.get("event_date"),
            **strategy,
            **availability,
            "requires_human_review": True,
            "review_reason": (
                "Au moins un produit demandé est indisponible à cette date. "
                "Proposer un produit disponible ou une autre date avant de calculer le prix."
            ),
            "tested_candidates": [],
        }

    result = dict(base_optimize(request, bundle, catalogue, config))
    result.update(strategy)
    result.update(availability)
    result["event_date"] = request.get("event_date")
    result["recommendation_status"] = "OK"

    if strategy["fill_date_mode"] and result.get("all_constraints_satisfied", True):
        selected = select_fill_date_candidate(result)
        if selected is not None:
            _apply_selected_candidate(result, selected)
            fill_reason = (
                "Date vide ou prioritaire : le prix favorise la signature sans "
                f"descendre sous {result['minimum_direct_margin_eur']:.0f} € "
                "de marge directe."
            )
            previous_reason = str(result.get("review_reason") or "").strip()
            result["review_reason"] = (
                f"{previous_reason} {fill_reason}".strip()
                if previous_reason
                else fill_reason
            )

    return result


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument(
        "--catalogue", default=project_dir / "catalogue_tarifs.csv", type=Path
    )
    parser.add_argument("--config", default=project_dir / "config.json", type=Path)
    args = parser.parse_args()

    request = load_json(args.request)
    config = load_json(args.config)
    catalogue = read_catalogue(args.catalogue)
    bundle = joblib.load(args.model)
    recommendation = optimize_meta_price(request, bundle, catalogue, config)

    result = json.dumps(recommendation, ensure_ascii=False, indent=2)
    print(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
