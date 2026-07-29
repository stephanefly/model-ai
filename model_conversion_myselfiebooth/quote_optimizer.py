"""Calcule le prix X90 qui maximise la marge attendue sous contraintes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import pandas as pd

from feature_engineering import align_features, prepare_features


PRODUCT_TO_CODE = {
    "product_photobooth": "PHOTOBOOTH",
    "product_miroirbooth": "MIROIRBOOTH",
    "product_videobooth": "BOOTH_360",
    "product_voguebooth": "VOGUEBOOTH",
}

OPTION_TO_CODE = {
    "option_phonebooth": "PHONEBOOTH",
    "option_fond_360": "FOND_LED",
    "option_impression_voguebooth": "VOGUE_PRINT",
    "option_mur_floral": "MUR_FLORAL",
    "option_panneau_fontaine": "PANNEAU_FONTAINE",
    "option_video_livre_or": "LIVRE_VIDEO",
}

UNPRICED_FIELDS = {
    "product_ipadbooth": "iPadBooth",
    "product_airbooth": "AirBooth",
    "option_livre_or": "Livre d'or non vidéo",
    "option_panneau_bienvenue": "Panneau de bienvenue",
    "option_photographe_voguebooth": "Photographe VogueBooth",
    "option_decor_voguebooth": "Décor VogueBooth",
    "option_holo_3d": "Holo 3D",
    "option_delivery": "Livraison",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_catalogue(path: Path) -> pd.DataFrame:
    catalogue = pd.read_csv(path)
    for column in ("normal_price_eur", "direct_cost_eur", "floor_price_eur"):
        catalogue[column] = pd.to_numeric(catalogue[column], errors="coerce")
    return catalogue.set_index("code", drop=False)


def _is_selected(request: dict, field: str) -> bool:
    try:
        return float(request.get(field, 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def calculate_direct_cost(request: dict, catalogue: pd.DataFrame) -> tuple[float, list[dict]]:
    """Calcule le coût; refuse de considérer une donnée absente comme zéro."""
    if request.get("direct_cost_eur") is not None:
        return float(request["direct_cost_eur"]), [
            {
                "code": "MANUAL_OVERRIDE",
                "quantity": 1,
                "unit_direct_cost_eur": float(request["direct_cost_eur"]),
            }
        ]

    explicit_code = request.get("catalogue_code")
    items: list[tuple[str, float]] = []
    covered_products: set[str] = set()
    if explicit_code:
        items.append((str(explicit_code), 1))
    elif _is_selected(request, "product_miroirbooth") and _is_selected(
        request, "product_videobooth"
    ):
        items.append(("PACK_DUO", 1))
        covered_products.update({"product_miroirbooth", "product_videobooth"})

    for field, code in PRODUCT_TO_CODE.items():
        if field not in covered_products and _is_selected(request, field):
            items.append((code, float(request.get(field, 1))))

    for field, code in OPTION_TO_CODE.items():
        if _is_selected(request, field):
            items.append((code, 1))

    quantities = {
        "option_porte_cles_quantity": "PORTE_CLES_100",
        "option_magnets_quantity": "MAGNETS_100",
        "option_magnets_simple_quantity": "MAGNETS_100",
    }
    for field, code in quantities.items():
        quantity = float(request.get(field, 0) or 0)
        if quantity > 0:
            items.append((code, math.ceil(quantity / 100)))

    missing_manual = [
        label for field, label in UNPRICED_FIELDS.items() if _is_selected(request, field)
    ]
    if float(request.get("option_duration_hours", 5) or 5) > 5:
        missing_manual.append("Durée supérieure à 5 heures")
    if missing_manual:
        raise ValueError(
            "Coût direct manquant pour : "
            + ", ".join(missing_manual)
            + ". Renseignez direct_cost_eur dans la demande ou complétez le catalogue."
        )
    if not items:
        raise ValueError(
            "Aucune prestation tarifée détectée. Renseignez les product_*, catalogue_code "
            "ou direct_cost_eur."
        )

    details = []
    total = 0.0
    for code, quantity in items:
        if code not in catalogue.index:
            raise ValueError(f"Code absent du catalogue : {code}")
        unit_cost = catalogue.loc[code, "direct_cost_eur"]
        if pd.isna(unit_cost):
            raise ValueError(
                f"Coût direct absent pour {code}. Complétez le catalogue ou fournissez "
                "direct_cost_eur."
            )
        subtotal = float(unit_cost) * quantity
        total += subtotal
        details.append(
            {
                "code": code,
                "quantity": quantity,
                "unit_direct_cost_eur": float(unit_cost),
                "subtotal_direct_cost_eur": subtotal,
            }
        )
    return total, details


def ending_prices(minimum: float, maximum: float, ending: int, step: int) -> list[int]:
    first = math.ceil((minimum - ending) / step) * step + ending
    if first < minimum:
        first += step
    return list(range(int(first), int(math.floor(maximum)) + 1, int(step)))


def optimize(request: dict, bundle: dict, catalogue: pd.DataFrame, config: dict) -> dict:
    direct_cost, cost_details = calculate_direct_cost(request, catalogue)
    minimum_margin = float(config["minimum_direct_margin_eur"])
    probability_floor = float(config["minimum_acceptance_probability"])
    gross = float(request.get("price_catalog_gross") or 0)
    if gross <= 0:
        normal_prices = [
            catalogue.loc[item["code"], "normal_price_eur"]
            for item in cost_details
            if item["code"] in catalogue.index
        ]
        if not normal_prices or any(pd.isna(value) for value in normal_prices):
            raise ValueError("price_catalog_gross doit être renseigné.")
        gross = float(sum(normal_prices))

    minimum_price = max(
        direct_cost + minimum_margin,
        float(request.get("minimum_price_eur") or 0),
    )
    maximum_price = float(
        request.get("maximum_price_eur")
        or max(gross, minimum_price) * float(config["maximum_price_multiplier"])
    )
    prices = ending_prices(
        minimum_price,
        maximum_price,
        int(config["price_ending"]),
        int(config["price_step_eur"]),
    )
    if not prices:
        raise ValueError(
            f"Aucun prix finissant par {config['price_ending']} entre "
            f"{minimum_price:.2f} € et {maximum_price:.2f} €."
        )

    price_model = bundle.get("price_model", bundle["model"])
    price_model_name = bundle.get("price_model_name", bundle["model_name"])
    rows = []
    previous_probability = 1.0
    for price in prices:
        candidate = dict(request)
        candidate["price_catalog_gross"] = gross
        candidate["price_proposed"] = price
        frame = align_features(
            prepare_features(pd.DataFrame([candidate])),
            bundle["numeric_features"],
            bundle["categorical_features"],
        )
        raw_probability = float(price_model.predict_proba(frame)[0, 1])
        # Garde-fou économique : à demande identique, augmenter le prix ne peut
        # pas faire monter la probabilité utilisée par l'optimiseur.
        probability = min(previous_probability, raw_probability)
        previous_probability = probability
        margin = price - direct_cost
        rows.append(
            {
                "price_eur": int(price),
                "signature_probability": round(probability, 4),
                "raw_model_probability": round(raw_probability, 4),
                "direct_margin_eur": round(margin, 2),
                "expected_direct_margin_eur": round(probability * margin, 2),
                "meets_probability_floor": probability >= probability_floor,
            }
        )

    probability_drop = rows[0]["raw_model_probability"] - rows[-1]["raw_model_probability"]
    minimum_drop = float(
        config.get("minimum_probability_drop_across_price_range", 0.05)
    )
    price_signal_reliable = probability_drop >= minimum_drop

    review_reason = None
    selection_rows = rows
    if not price_signal_reliable:
        safe_maximum = gross * float(
            config.get("maximum_unproven_price_multiplier", 1.1)
        )
        selection_rows = [row for row in rows if row["price_eur"] <= safe_maximum]
        if not selection_rows:
            selection_rows = [min(rows, key=lambda row: abs(row["price_eur"] - gross))]
        review_reason = (
            "L'historique ne prouve pas encore suffisamment l'effet du prix. "
            "La recommandation est limitée à +10 % du prix catalogue."
        )

    feasible = [
        row for row in selection_rows if row["meets_probability_floor"]
    ]
    if feasible:
        pool = feasible
    else:
        pool = selection_rows
        probability_warning = (
            f"Aucun candidat n'atteint la probabilité minimale de {probability_floor:.0%}."
        )
        review_reason = (
            f"{review_reason} {probability_warning}".strip()
            if review_reason
            else probability_warning
        )

    best_expected = max(row["expected_direct_margin_eur"] for row in pool)
    tolerance = float(config["near_optimal_expected_margin_tolerance"])
    near_best = [
        row
        for row in pool
        if row["expected_direct_margin_eur"] >= best_expected * (1 - tolerance)
    ]
    selected = max(near_best, key=lambda row: row["price_eur"])

    return {
        "recommended_price_eur": selected["price_eur"],
        "estimated_signature_probability": selected["signature_probability"],
        "estimated_direct_margin_eur": selected["direct_margin_eur"],
        "estimated_expected_direct_margin_eur": selected["expected_direct_margin_eur"],
        "price_catalog_gross_eur": gross,
        "direct_cost_eur": direct_cost,
        "minimum_direct_margin_eur": minimum_margin,
        "minimum_acceptance_probability": probability_floor,
        "model_name": bundle["model_name"],
        "price_model_name": price_model_name,
        "price_signal_reliable": price_signal_reliable,
        "probability_drop_across_tested_range": round(probability_drop, 4),
        "requires_human_review": True,
        "review_reason": review_reason or "Validation humaine obligatoire pendant la phase pilote.",
        "cost_details": cost_details,
        "tested_candidates": rows,
    }


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
    recommendation = optimize(request, bundle, catalogue, config)
    result = json.dumps(recommendation, ensure_ascii=False, indent=2)
    print(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
