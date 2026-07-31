import unittest

import pandas as pd

from feature_engineering import prepare_features, select_feature_columns


class FeatureEngineeringTests(unittest.TestCase):
    def test_builds_pre_quote_timing_and_history_features(self):
        frame = pd.DataFrame([
            {
                "request_created_at_utc": "2026-07-01T20:30:00Z",
                "prior_last_request_at_utc": "2026-06-01T20:30:00Z",
                "prior_request_count": 2,
                "prior_signed_count": 1,
                "event_date": "2026-07-20",
            }
        ])

        result = prepare_features(frame).iloc[0]

        self.assertEqual(result["request_daypart"], "SOIR")
        self.assertEqual(result["request_is_outside_business_hours"], 1)
        self.assertEqual(result["event_lead_time_days"], 19)
        self.assertEqual(result["event_lead_time_bucket"], "15_30_JOURS")
        self.assertEqual(result["event_is_last_minute"], 1)
        self.assertEqual(result["prior_days_since_last_request"], 30)
        self.assertEqual(result["prior_signature_rate"], 0.5)

    def test_builds_product_bundle_and_completeness_features(self):
        frame = pd.DataFrame([
            {
                "product_photobooth": 0,
                "product_miroirbooth": 1,
                "product_videobooth": 1,
                "product_voguebooth": 0,
                "option_phonebooth": 1,
                "option_porte_cles_quantity": 100,
                "enrichment_address_available": 1,
                "event_has_schedule": 1,
                "event_has_client_comment": 0,
            }
        ])

        result = prepare_features(frame).iloc[0]

        self.assertEqual(result["product_count"], 2)
        self.assertEqual(result["product_is_multi"], 1)
        self.assertEqual(result["product_premium_count"], 2)
        self.assertEqual(result["product_bundle_signature"], "MIROIRBOOTH+BOOTH_360")
        self.assertEqual(result["option_count"], 2)
        self.assertEqual(result["option_quantity_total"], 100)
        self.assertEqual(result["request_total_items"], 4)
        self.assertEqual(result["event_request_completeness_score"], 0.75)
        self.assertEqual(result["event_missing_information_count"], 1)

    def test_post_quote_fields_are_never_selected(self):
        columns = {
            "request_source",
            "event_has_schedule",
            "product_bundle_signature",
            "email_open_count",
            "followup_count",
            "target_signed",
            "outcome_class",
        }

        numerics, categoricals = select_feature_columns(columns)

        self.assertIn("event_has_schedule", numerics)
        self.assertIn("request_source", categoricals)
        self.assertIn("product_bundle_signature", categoricals)
        self.assertNotIn("email_open_count", numerics + categoricals)
        self.assertNotIn("followup_count", numerics + categoricals)
        self.assertNotIn("target_signed", numerics + categoricals)
        self.assertNotIn("outcome_class", numerics + categoricals)


if __name__ == "__main__":
    unittest.main()
