"""Identifie les salles à partir d'un nom, d'une adresse, ou des deux."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import html
import json
import math
import os
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd


PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
FIELD_MASK = ",".join(
    [
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.primaryType",
        "places.types",
        "places.rating",
        "places.userRatingCount",
        "places.websiteUri",
        "places.googleMapsUri",
        "places.priceLevel",
        "places.location",
    ]
)
VENUE_TYPES = {
    "wedding_venue",
    "event_venue",
    "banquet_hall",
    "convention_center",
    "hotel",
    "restaurant",
    "night_club",
}
PREMIUM_WORDS = {
    "chateau",
    "domaine",
    "manoir",
    "palace",
    "prestige",
    "abbaye",
    "orangerie",
    "relais",
}


def normalize(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def json_request(url: str, payload: dict, headers: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def search_places(query: str, api_key: str) -> list[dict]:
    response = json_request(
        PLACES_URL,
        {"textQuery": query, "languageCode": "fr", "regionCode": "FR", "pageSize": 5},
        {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": FIELD_MASK,
        },
    )
    return response.get("places", [])


def place_name(place: dict) -> str:
    display_name = place.get("displayName") or {}
    return display_name.get("text", "") if isinstance(display_name, dict) else str(display_name)


def match_score(query: str, place: dict) -> float:
    query_norm = normalize(query)
    candidate = normalize(f"{place_name(place)} {place.get('formattedAddress', '')}")
    sequence = difflib.SequenceMatcher(None, query_norm, candidate).ratio()
    query_tokens = set(query_norm.split())
    candidate_tokens = set(candidate.split())
    overlap = len(query_tokens & candidate_tokens) / max(1, len(query_tokens))
    return round(0.55 * sequence + 0.45 * overlap, 4)


def select_place(query: str, places: list[dict]) -> tuple[dict | None, float]:
    if not places:
        return None, 0.0
    scored = [(place, match_score(query, place)) for place in places]
    return max(scored, key=lambda item: item[1])


def is_private_address(place: dict) -> int:
    types = set(place.get("types") or [])
    if types & VENUE_TYPES:
        return 0
    if types & {"street_address", "premise", "subpremise", "route"}:
        return 1
    return 0


def venue_scores(place: dict) -> tuple[int, int]:
    rating = float(place.get("rating") or 0)
    reviews = int(place.get("userRatingCount") or 0)
    price_level = str(place.get("priceLevel") or "")
    name = normalize(place_name(place))
    premium_price_points = {
        "PRICE_LEVEL_FREE": 0,
        "PRICE_LEVEL_INEXPENSIVE": 5,
        "PRICE_LEVEL_MODERATE": 12,
        "PRICE_LEVEL_EXPENSIVE": 24,
        "PRICE_LEVEL_VERY_EXPENSIVE": 32,
    }.get(price_level, 8)
    rating_points = max(0, min(30, (rating - 3.5) * 20))
    review_points = min(25, math.log10(reviews + 1) * 9)
    keyword_points = 18 if set(name.split()) & PREMIUM_WORDS else 0
    website_points = 8 if place.get("websiteUri") else 0
    premium = round(min(100, premium_price_points + rating_points + keyword_points + website_points))
    notability = round(
        min(100, min(65, math.log10(reviews + 1) * 23) + max(0, rating - 3.5) * 20)
    )
    return premium, notability


class PageTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text: list[str] = []
        self.links: list[str] = []

    def handle_data(self, data: str) -> None:
        self.text.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def fetch_page(url: str) -> tuple[str, list[str]]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MySelfieBoothVenueResearch/1.0 (+human-reviewed)"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return "", []
        raw = response.read(1_500_000).decode("utf-8", errors="replace")
    parser = PageTextParser()
    parser.feed(raw)
    text = re.sub(r"\s+", " ", html.unescape(" ".join(parser.text)))
    return text, parser.links


def capacity_from_text(text: str) -> tuple[int | None, int | None]:
    patterns = [
        re.compile(
            r"(?:capacite|accueillir|jusqu.?a|maximum|max)[^0-9]{0,35}([1-9][0-9]{1,3})"
            r"\s*(?:personnes|convives|places)",
            re.IGNORECASE,
        ),
        re.compile(
            r"([1-9][0-9]{1,3})\s*(?:personnes|convives|places)[^.!?]{0,30}"
            r"(?:assis|assises|cocktail|reception)",
            re.IGNORECASE,
        ),
    ]
    seated: list[int] = []
    cocktail: list[int] = []
    normalized = normalize(text)
    for pattern in patterns:
        for match in pattern.finditer(normalized):
            value = int(match.group(1))
            if not 20 <= value <= 3000:
                continue
            context = normalized[max(0, match.start() - 50) : match.end() + 50]
            if any(word in context for word in ("assis", "assise", "banquet", "repas")):
                seated.append(value)
            elif any(word in context for word in ("cocktail", "debout")):
                cocktail.append(value)
            else:
                cocktail.append(value)
    return (max(seated) if seated else None, max(cocktail) if cocktail else None)


def scrape_official_capacity(website: str) -> tuple[int | None, int | None, int]:
    parsed_home = urllib.parse.urlparse(website)
    if parsed_home.scheme not in {"http", "https"} or not parsed_home.netloc:
        return None, None, 0
    urls = [website]
    seen = set()
    seated_values: list[int] = []
    cocktail_values: list[int] = []
    fetched = 0
    while urls and fetched < 3:
        url = urls.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            text, links = fetch_page(url)
        except (OSError, urllib.error.URLError, ValueError):
            continue
        fetched += 1
        seated, cocktail = capacity_from_text(text)
        if seated:
            seated_values.append(seated)
        if cocktail:
            cocktail_values.append(cocktail)
        for link in links:
            absolute = urllib.parse.urljoin(url, link)
            parsed = urllib.parse.urlparse(absolute)
            if parsed.netloc != parsed_home.netloc:
                continue
            if any(word in normalize(parsed.path) for word in ("mariage", "reception", "salle", "capacite")):
                urls.append(absolute)
    return (
        max(seated_values) if seated_values else None,
        max(cocktail_values) if cocktail_values else None,
        fetched,
    )


def load_overrides(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {
            str(row["match_key"]).strip(): row
            for row in csv.DictReader(stream)
            if row.get("match_key") and not row["match_key"].startswith("EXEMPLE_")
        }


def nullable_number(value: object, integer: bool = False):
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    return int(number) if integer else number


def enrich_row(
    row: pd.Series,
    api_key: str | None,
    cache: dict,
    overrides: dict[str, dict],
    scrape_websites: bool,
) -> dict:
    query = str(row.get("enrichment_event_full_address_raw") or "").strip()
    anonymous_key = str(row.get("event_venue_anonymous_key") or "").strip()
    override = overrides.get(anonymous_key)
    if override:
        return {
            "venue_name": override.get("venue_name"),
            "venue_match_confidence": 1.0,
            "venue_is_private_address": 0,
            "venue_type": override.get("venue_type"),
            "venue_capacity_seated": nullable_number(
                override.get("venue_capacity_seated"), integer=True
            ),
            "venue_capacity_cocktail": nullable_number(
                override.get("venue_capacity_cocktail"), integer=True
            ),
            "venue_rating": None,
            "venue_review_count": None,
            "venue_price_min": nullable_number(override.get("venue_price_min")),
            "venue_price_max": nullable_number(override.get("venue_price_max")),
            "venue_has_official_website": int(bool(override.get("source_url"))),
            "venue_premium_score": nullable_number(
                override.get("venue_premium_score"), integer=True
            ),
            "venue_notability_score": nullable_number(
                override.get("venue_notability_score"), integer=True
            ),
            "venue_enrichment_sources_count": 1,
            "venue_enrichment_status": "MANUAL_OVERRIDE",
        }
    if not query:
        return {"venue_enrichment_status": "NO_ADDRESS_OR_NAME"}
    if not api_key:
        return {"venue_enrichment_status": "NO_GOOGLE_PLACES_API_KEY"}

    cache_key = hashlib.sha256(normalize(query).encode()).hexdigest()
    if cache_key not in cache:
        cache[cache_key] = {"query": query, "places": search_places(query, api_key)}
        time.sleep(0.08)
    place, confidence = select_place(query, cache[cache_key]["places"])
    if not place:
        return {"venue_enrichment_status": "NO_MATCH"}

    premium, notability = venue_scores(place)
    seated = cocktail = None
    fetched_pages = 0
    website = place.get("websiteUri")
    if scrape_websites and website:
        seated, cocktail, fetched_pages = scrape_official_capacity(website)

    return {
        "venue_name": place_name(place),
        "venue_match_confidence": confidence,
        "venue_is_private_address": is_private_address(place),
        "venue_type": place.get("primaryType"),
        "venue_capacity_seated": seated,
        "venue_capacity_cocktail": cocktail,
        "venue_rating": place.get("rating"),
        "venue_review_count": place.get("userRatingCount"),
        "venue_price_min": None,
        "venue_price_max": None,
        "venue_has_official_website": int(bool(website)),
        "venue_premium_score": premium,
        "venue_notability_score": notability,
        "venue_enrichment_sources_count": 1 + int(fetched_pages > 0),
        "venue_enrichment_status": (
            "MATCH_REVIEW_REQUIRED" if confidence < 0.55 else "MATCHED"
        ),
    }


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", default=Path("venue_cache.json"), type=Path)
    parser.add_argument(
        "--overrides", default=project_dir / "venue_overrides.csv", type=Path
    )
    parser.add_argument("--scrape-official-websites", action="store_true")
    parser.add_argument("--keep-raw-addresses", action="store_true")
    args = parser.parse_args()

    data = pd.read_csv(args.input, sep=None, engine="python")
    api_key = os.environ.get("GOOGLE_PLACES_API_KEY")
    cache = {}
    if args.cache.exists():
        cache = json.loads(args.cache.read_text(encoding="utf-8"))
    overrides = load_overrides(args.overrides)

    enriched = []
    for _, row in data.iterrows():
        try:
            enriched.append(
                enrich_row(
                    row,
                    api_key=api_key,
                    cache=cache,
                    overrides=overrides,
                    scrape_websites=args.scrape_official_websites,
                )
            )
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            enriched.append({"venue_enrichment_status": f"HTTP_ERROR_{type(error).__name__}"})

    result = pd.concat([data.reset_index(drop=True), pd.DataFrame(enriched)], axis=1)
    if not args.keep_raw_addresses:
        raw_columns = [
            column
            for column in result.columns
            if column.startswith("enrichment_event_")
        ]
        result = result.drop(columns=raw_columns)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    args.cache.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(result)} lignes enregistrées dans {args.output}")
    print(result["venue_enrichment_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
