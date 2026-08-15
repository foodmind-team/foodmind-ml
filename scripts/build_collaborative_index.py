"""Create a serving CF index from a Backend training snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from foodmind_ml.collaborative.index import write_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-neighbor-support", type=int, default=3)
    parser.add_argument("--min-item-support", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=25)
    options = parser.parse_args()
    write_index(
        options.snapshot,
        options.output,
        min_neighbor_support=options.min_neighbor_support,
        min_item_support=options.min_item_support,
        top_k=options.top_k,
    )
