#!/usr/bin/env python3
"""
adapters/yelp.py

TrueRating dataset adapter for the Yelp Open Dataset
(yelp_academic_dataset_business.json + yelp_academic_dataset_review.json).

Performance note (this is the one adapter that needs it): the review file
is 5GB+, far too big to load into memory. So this adapter still does the
same two-pass trick the original standalone phase1_data.py used:
  1. Read the (much smaller) business file fully, filter to restaurant
     candidates, and pre-sample down to `args.sample_size` business_ids.
     This bounds how many businesses the expensive second pass has to
     track, without which streaming a multi-GB file would still mean
     holding reviews for hundreds of thousands of businesses in memory.
  2. Stream the review file line-by-line, keeping only reviews whose
     business_id is in that pre-sampled set, batching them straight into
     RestaurantRecord/ReviewRecord objects -- never materializing the
     whole file.

Because of that pre-sampling, ingest.py's shared (dataset-agnostic)
filter-and-sample step mostly acts as a validation pass for this adapter
(dropping the rare business that turned out to have fewer real reviews in
the review file than --min-reviews requires) rather than doing the "real"
random selection -- that's a deliberate, documented tradeoff for the sake
of not reading a 5GB file twice at full scale. Zomato's adapter, working
from a much smaller single file, can afford to hand ingest.py the full
candidate pool and let it do the real sampling.
"""

import json
import random
import sys
from typing import Dict, List

from adapters.base import BaseDatasetAdapter, RestaurantRecord, ReviewRecord

CATEGORY_FILTER = "Restaurants"   # case-sensitive substring match, matches Yelp's own casing
MIN_BUSINESS_REVIEW_COUNT = 15    # quality bar on Yelp's own review_count metadata field
RANDOM_SEED = 42


class YelpAdapter(BaseDatasetAdapter):
    name = "yelp"
    description = "Yelp Open Dataset (business.json + review.json)"

    def add_cli_arguments(self, parser) -> None:
        parser.add_argument(
            "--business-file",
            default="yelp_academic_dataset_business.json",
            help="Path to yelp_academic_dataset_business.json",
        )
        parser.add_argument(
            "--review-file",
            default="yelp_academic_dataset_review.json",
            help="Path to yelp_academic_dataset_review.json (5GB+, streamed line-by-line)",
        )

    def load(self, args) -> List[RestaurantRecord]:
        candidates = self._load_and_filter_businesses(args.business_file)
        if not candidates:
            print("[YELP][ERROR] No businesses matched the filter criteria.", file=sys.stderr)
            return []

        pre_sampled = self._pre_sample(candidates, args.sample_size)
        target_ids = {b["business_id"] for b in pre_sampled}
        print(f"[YELP] Pre-sampled {len(target_ids)} candidate restaurants "
              f"before streaming the review file.")

        records_by_id = {
            b["business_id"]: RestaurantRecord(
                name=b.get("name"),
                categories=b.get("categories"),
                city=b.get("city"),
            )
            for b in pre_sampled
        }

        self._stream_matching_reviews(args.review_file, target_ids, records_by_id)
        return list(records_by_id.values())

    # -- internal helpers -----------------------------------------------
    def _load_and_filter_businesses(self, business_file: str) -> List[dict]:
        candidates = []
        total_lines = 0
        try:
            with open(business_file, "r", encoding="utf-8") as f:
                for line in f:
                    total_lines += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    categories = record.get("categories") or ""
                    is_open = record.get("is_open")
                    review_count = record.get("review_count") or 0

                    if (
                        CATEGORY_FILTER in categories
                        and is_open == 1
                        and review_count >= MIN_BUSINESS_REVIEW_COUNT
                    ):
                        candidates.append(record)
        except FileNotFoundError:
            print(f"[YELP][ERROR] File not found: {business_file}", file=sys.stderr)
            raise
        except OSError as exc:
            print(f"[YELP][ERROR] Could not read {business_file}: {exc}", file=sys.stderr)
            raise

        print(f"[YELP] Scanned {total_lines:,} business records, "
              f"{len(candidates):,} matched filters (categories contains "
              f"'{CATEGORY_FILTER}', is_open=1, review_count>={MIN_BUSINESS_REVIEW_COUNT}).")
        return candidates

    def _pre_sample(self, candidates: List[dict], sample_size: int) -> List[dict]:
        if len(candidates) <= sample_size:
            print(f"[YELP][WARN] Only {len(candidates)} candidates available; "
                  f"requested {sample_size}. Using all candidates.", file=sys.stderr)
            return candidates
        random.seed(RANDOM_SEED)
        return random.sample(candidates, sample_size)

    def _stream_matching_reviews(
        self, review_file: str, target_ids: set, records_by_id: Dict[str, RestaurantRecord]
    ) -> None:
        total_lines = 0
        total_matches = 0
        try:
            with open(review_file, "r", encoding="utf-8") as f:
                for line in f:
                    total_lines += 1
                    if total_lines % 1_000_000 == 0:
                        print(f"[YELP] Processed {total_lines:,} review lines, "
                              f"found {total_matches:,} matches so far...")

                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    business_id = record.get("business_id")
                    if business_id not in target_ids:
                        continue

                    text = record.get("text")
                    if text is None or not text.strip():
                        continue

                    records_by_id[business_id].reviews.append(
                        ReviewRecord(
                            text=text.strip(),
                            stars=record.get("stars"),
                            date=record.get("date"),
                        )
                    )
                    total_matches += 1
        except FileNotFoundError:
            print(f"[YELP][ERROR] File not found: {review_file}", file=sys.stderr)
            raise
        except OSError as exc:
            print(f"[YELP][ERROR] Could not read {review_file}: {exc}", file=sys.stderr)
            raise

        print(f"[YELP] Finished streaming {total_lines:,} review lines, "
              f"{total_matches:,} matched a pre-sampled restaurant.")
