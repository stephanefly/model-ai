import unittest
from unittest.mock import patch

import meta_price_optimizer as meta


BASE_RESULT = {
    "recommended_price_eur": 690,
    "all_constraints_satisfied": True,
    "estimated_signature_probability": 0.62,
    "estimated_direct_margin_eur": 340,
    "estimated_expected_direct_margin_eur": 210.8,
    "price_catalog_gross_eur": 750,
    "minimum_direct_margin_eur": 150,
    "minimum_acceptance_probability": 0.60,
    "price_signal_reliable": True,
    "requires_human_review": True,
    "review_reason": "Validation humaine obligatoire pendant la phase pilote.",
    "tested_candidates": [
        {
            "price_eur": 490,
            "signature_probability": 0.74,
            "direct_margin_eur": 140,
            "expected_direct_margin_eur": 103.6,
        },
        {
            "price_eur": 590,
            "signature_probability": 0.70,
            "direct_margin_eur": 240,
            "expected_direct_margin_eur": 168,
        },
        {
            "price_eur": 690,
            "signature_probability": 0.62,
            "direct_margin_eur": 340,
            "expected_direct_margin_eur": 210.8,
        },
    ],
}


class MetaPriceOptimizerTests(unittest.TestCase):
    def test_unavailable_product_stops_price_calculation(self):
        request = {
            "product_miroirbooth": 1,
            "available_products": {
                "MIROIRBOOTH": 0,
                "PHOTOBOOTH": 2,
            },
        }
        with patch.object(meta, "base_optimize") as mocked:
            result = meta.optimize_meta_price(request, {}, None, {})

        mocked.assert_not_called()
        self.assertEqual(result["recommendation_status"], "PRODUCT_UNAVAILABLE")
        self.assertEqual(result["alternative_products"][0]["code"], "PHOTOBOOTH")

    def test_empty_near_date_selects_conversion_price_above_margin_floor(self):
        request = {
            "product_miroirbooth": 1,
            "available_products": {"MIROIRBOOTH": 1},
            "bookings_on_date": 0,
            "days_before_event": 20,
        }
        config = {"fill_date_horizon_days": 45}
        with patch.object(meta, "base_optimize", return_value=BASE_RESULT):
            result = meta.optimize_meta_price(request, {}, None, config)

        self.assertTrue(result["fill_date_mode"])
        self.assertEqual(result["recommended_price_eur"], 590)
        self.assertGreaterEqual(result["estimated_direct_margin_eur"], 150)

    def test_empty_far_date_keeps_base_price(self):
        request = {
            "product_miroirbooth": 1,
            "available_products": {"MIROIRBOOTH": 1},
            "bookings_on_date": 0,
            "days_before_event": 120,
        }
        config = {"fill_date_horizon_days": 45}
        with patch.object(meta, "base_optimize", return_value=BASE_RESULT):
            result = meta.optimize_meta_price(request, {}, None, config)

        self.assertFalse(result["fill_date_mode"])
        self.assertEqual(result["recommended_price_eur"], 690)


if __name__ == "__main__":
    unittest.main()
