"""
adapters/__init__.py

Registry of every dataset adapter TrueRating currently supports. To add a
new dataset:
    1. Write adapters/<name>.py with a class implementing BaseDatasetAdapter.
    2. Import it below and add it to ADAPTERS.
That's the entire integration surface -- ingest.py, and everything
downstream of it, needs no other changes.
"""

from adapters.yelp import YelpAdapter
from adapters.zomato import ZomatoAdapter

ADAPTERS = {
    "yelp": YelpAdapter(),
    "zomato": ZomatoAdapter(),
}
