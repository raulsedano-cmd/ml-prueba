"""Automatic product price adjustment tool.

This module provides a command line program capable of calculating
new prices for a catalogue of products.  The program reads a CSV file
with product data and applies a configurable set of rules to obtain
updated prices that balance profitability, demand and competition.

Usage example
-------------
```
python price_adjuster.py products.csv --output adjusted.csv \
    --demand-sensitivity 0.4 --competitor-weight 0.25 \
    --stock-threshold 15 --stock-sensitivity 0.1
```

CSV columns
-----------
The input CSV must contain at least the following columns:

* ``name``: Name of the product.
* ``price``: Current selling price.
* ``cost``: Production or acquisition cost.
* ``target_margin``: Desired profit margin expressed as a decimal (e.g. 0.25 for 25%).
* ``demand_index``: Relative demand indicator where ``1`` is neutral, values > 1
  denote higher demand and values < 1 denote lower demand.
* ``stock``: Available stock units.

Optional columns:

* ``competitor_price``: Price of a comparable product sold by a competitor.
* ``min_price`` and ``max_price``: Hard bounds for the adjusted price.

The output CSV contains the original data plus a ``new_price`` column with the
recommended price.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class Product:
    """Represents product information required to adjust the price."""

    name: str
    price: float
    cost: float
    target_margin: float
    demand_index: float
    stock: float
    competitor_price: Optional[float] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None


@dataclass
class AdjustmentConfig:
    """Parameters that control the price adjustment algorithm."""

    demand_sensitivity: float = 0.3
    competitor_weight: float = 0.2
    stock_threshold: float = 20.0
    stock_sensitivity: float = 0.1


def clamp(value: float, lower: Optional[float], upper: Optional[float]) -> float:
    """Restrict ``value`` to the interval defined by ``lower`` and ``upper``."""

    if lower is not None and value < lower:
        return lower
    if upper is not None and value > upper:
        return upper
    return value


def adjust_price(product: Product, config: AdjustmentConfig) -> float:
    """Compute the new price for a single product.

    The algorithm combines several elements:

    1. Base price derived from cost and desired margin.
    2. Demand adjustment: positive demand raises price, negative demand lowers it.
    3. Competition adjustment: align towards competitor price when available.
    4. Inventory adjustment: boost price if stock is scarce, reduce if abundant.
    5. Optional bounds defined in the product data.
    """

    # Base price ensures that the target margin is respected.
    base_price = product.cost * (1 + product.target_margin)

    # Demand adjustment: demand_index > 1 should increase price, < 1 decrease it.
    demand_delta = product.demand_index - 1.0
    demand_adjustment = 1 + config.demand_sensitivity * demand_delta
    price_after_demand = base_price * max(demand_adjustment, 0)

    # Competition adjustment: move price towards competitor price proportionally.
    if product.competitor_price is not None:
        competitor_gap = product.competitor_price - product.price
        competition_adjustment = config.competitor_weight * competitor_gap
    else:
        competition_adjustment = 0.0
    price_after_competition = price_after_demand + competition_adjustment

    # Inventory adjustment: encourage sales when stock is high, protect when low.
    if product.stock <= 0:
        # Avoid division by zero and keep a premium price when out of stock.
        stock_factor = 1 + config.stock_sensitivity
    else:
        stock_ratio = product.stock / config.stock_threshold
        stock_factor = 1 - config.stock_sensitivity * (stock_ratio - 1)
    price_after_stock = price_after_competition * max(stock_factor, 0)

    # Enforce optional limits.
    final_price = clamp(price_after_stock, product.min_price, product.max_price)

    return round(final_price, 2)


def parse_product(row: dict) -> Product:
    """Create a :class:`Product` instance from a CSV row."""

    def optional_float(key: str) -> Optional[float]:
        value = row.get(key)
        if value in (None, ""):
            return None
        return float(value)

    return Product(
        name=row["name"],
        price=float(row["price"]),
        cost=float(row["cost"]),
        target_margin=float(row["target_margin"]),
        demand_index=float(row["demand_index"]),
        stock=float(row["stock"]),
        competitor_price=optional_float("competitor_price"),
        min_price=optional_float("min_price"),
        max_price=optional_float("max_price"),
    )


def read_products(csv_path: Path) -> List[Product]:
    """Read product data from ``csv_path``."""

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"name", "price", "cost", "target_margin", "demand_index", "stock"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        return [parse_product(row) for row in reader]


def write_products(csv_path: Path, fieldnames: Iterable[str], rows: Iterable[dict]) -> None:
    """Write rows to ``csv_path`` using ``fieldnames`` order."""

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def adjust_catalogue(products: Iterable[Product], config: AdjustmentConfig) -> List[dict]:
    """Adjust prices for all products, returning dictionaries ready to export."""

    adjusted_rows = []
    for product in products:
        new_price = adjust_price(product, config)
        row = {
            "name": product.name,
            "price": product.price,
            "cost": product.cost,
            "target_margin": product.target_margin,
            "demand_index": product.demand_index,
            "stock": product.stock,
            "competitor_price": product.competitor_price,
            "min_price": product.min_price,
            "max_price": product.max_price,
            "new_price": new_price,
        }
        adjusted_rows.append(row)
    return adjusted_rows


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adjust product prices based on demand, competition and inventory."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="CSV file containing product data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("adjusted_prices.csv"),
        help="Destination CSV for adjusted prices (default: adjusted_prices.csv)",
    )
    parser.add_argument(
        "--demand-sensitivity",
        type=float,
        default=0.3,
        help="Multiplier applied to demand deviation from neutral (default: 0.3)",
    )
    parser.add_argument(
        "--competitor-weight",
        type=float,
        default=0.2,
        help="Weight applied to the gap between competitor and current price (default: 0.2)",
    )
    parser.add_argument(
        "--stock-threshold",
        type=float,
        default=20.0,
        help="Stock level considered balanced (default: 20 units)",
    )
    parser.add_argument(
        "--stock-sensitivity",
        type=float,
        default=0.1,
        help="Strength of the adjustment based on inventory levels (default: 0.1)",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    config = AdjustmentConfig(
        demand_sensitivity=args.demand_sensitivity,
        competitor_weight=args.competitor_weight,
        stock_threshold=args.stock_threshold,
        stock_sensitivity=args.stock_sensitivity,
    )

    products = read_products(args.input)
    adjusted_rows = adjust_catalogue(products, config)

    # Ensure consistent column order for the output.
    fieldnames = [
        "name",
        "price",
        "cost",
        "target_margin",
        "demand_index",
        "stock",
        "competitor_price",
        "min_price",
        "max_price",
        "new_price",
    ]

    write_products(args.output, fieldnames, adjusted_rows)


if __name__ == "__main__":
    main()
