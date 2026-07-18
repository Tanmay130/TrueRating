#!/usr/bin/env python3
"""
adapters/base.py

The contract every TrueRating dataset adapter must satisfy.

TrueRating is a platform, not a Yelp-specific tool: Phase 1 is the only
part of the pipeline that knows anything about a particular dataset's raw
file format. Phases 2-5, app.py, and evaluate.py only ever read the
canonical `restaurants` / `reviews` SQLite schema, so they work unchanged
regardless of which dataset produced it.

An adapter's ONLY job is turning its dataset's raw input file(s) into a
list of RestaurantRecord objects (each carrying its own de-duplicated
ReviewRecord list). Everything dataset-agnostic -- minimum-review
filtering, random sampling down to a dev-scale size, assigning stable
IDs, writing to SQLite, building indexes -- is handled exactly once, in
ingest.py, so adapters never reimplement it and never drift out of sync
with each other.

To add support for a new dataset:
  1. Create adapters/<name>.py with a class implementing BaseDatasetAdapter.
  2. Register it in adapters/__init__.py's ADAPTERS dict.
  3. Run: python ingest.py <name> --db truerating_<name>.db ...
No other file needs to change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ReviewRecord:
    """One review, already cleaned (non-empty text) and de-duplicated."""

    text: str
    stars: Optional[float] = None
    date: Optional[str] = None


@dataclass
class RestaurantRecord:
    """
    One restaurant candidate. `id` is intentionally NOT set here -- stable
    IDs are assigned once, in ingest.py, only for restaurants that survive
    filtering and sampling, so adapters never need to worry about ID
    collisions across datasets or across runs.
    """

    name: str
    categories: Optional[str] = None
    city: Optional[str] = None
    reference_rating: Optional[float] = None
    votes: Optional[int] = None
    approx_cost: Optional[float] = None
    reviews: List[ReviewRecord] = field(default_factory=list)


class BaseDatasetAdapter(ABC):
    """Base class every dataset adapter (adapters/yelp.py, adapters/zomato.py,
    ...) must subclass."""

    #: Short identifier used on the CLI, e.g. "yelp", "zomato". Must be
    #: unique across all registered adapters.
    name: str = "base"

    #: Human-readable one-liner shown in `python ingest.py --help`.
    description: str = ""

    @abstractmethod
    def add_cli_arguments(self, parser) -> None:
        """
        Register this adapter's own dataset-specific CLI arguments (e.g.
        --business-file/--review-file for Yelp, --input for Zomato) onto
        the subparser ingest.py creates for this adapter. Do NOT add --db,
        --sample-size, or --min-reviews here -- those are shared flags
        ingest.py already provides for every adapter.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, args) -> List[RestaurantRecord]:
        """
        Parse this dataset's raw input file(s) (paths available via the
        parsed `args` namespace, using whatever attribute names this
        adapter's add_cli_arguments() registered) and return every
        candidate restaurant found, each with its full de-duplicated
        review list.

        Do NOT enforce --min-reviews or --sample-size here -- ingest.py
        applies those uniformly after every adapter returns, so results
        stay comparable across datasets. It's fine (and encouraged) to
        skip rows that are malformed/unparseable; it is NOT fine to
        silently drop valid restaurants because they look "too small" --
        that's a filtering decision, not a parsing one.
        """
        raise NotImplementedError
