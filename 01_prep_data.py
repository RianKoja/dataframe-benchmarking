import os
from typing import Any, Dict

import numpy as np
import pandas as pd


def create_datasets() -> None:
    """Create comprehensive datasets for benchmarking pandas, fireducks, and polars"""

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Dataset sizes - make them large enough for meaningful benchmarks
    n_customers = 100_000
    n_orders = 500_000
    n_products = 10_000
    n_order_items = 1_200_000
    n_reviews = 300_000

    print("Creating customers dataset...")
    # Customers table
    customers = pd.DataFrame(
        {
            "customer_id": range(1, n_customers + 1),
            "name": [f"Customer_{i}" for i in range(1, n_customers + 1)],
            "email": [f"customer_{i}@email.com" for i in range(1, n_customers + 1)],
            "age": np.random.randint(18, 80, n_customers),
            "city": np.random.choice(
                [
                    "New York",
                    "Los Angeles",
                    "Chicago",
                    "Houston",
                    "Phoenix",
                    "Philadelphia",
                    "San Antonio",
                    "San Diego",
                    "Dallas",
                    "Austin",
                ],
                n_customers,
            ),
            "registration_date": pd.date_range(
                "2020-01-01", periods=n_customers, freq="1h"
            ),
            "annual_income": np.random.normal(50000, 20000, n_customers),
            "customer_segment": np.random.choice(
                ["Premium", "Standard", "Basic"], n_customers, p=[0.2, 0.5, 0.3]
            ),
        }
    )

    print("Creating products dataset...")
    # Products table
    products = pd.DataFrame(
        {
            "product_id": range(1, n_products + 1),
            "product_name": [f"Product_{i}" for i in range(1, n_products + 1)],
            "category": np.random.choice(
                ["Electronics", "Clothing", "Books", "Home", "Sports"], n_products
            ),
            "price": np.random.uniform(10, 1000, n_products),
            "cost": np.random.uniform(5, 500, n_products),
            "weight": np.random.uniform(0.1, 50, n_products),
            "supplier_id": np.random.randint(1, 1000, n_products),
            "in_stock": np.random.choice([True, False], n_products, p=[0.8, 0.2]),
        }
    )

    print("Creating orders dataset...")
    # Orders table
    orders = pd.DataFrame(
        {
            "order_id": range(1, n_orders + 1),
            "customer_id": np.random.randint(1, n_customers + 1, n_orders),
            "order_date": pd.date_range("2020-01-01", periods=n_orders, freq="30min"),
            "shipping_date": pd.date_range(
                "2020-01-02", periods=n_orders, freq="30min"
            ),
            "total_amount": np.random.uniform(20, 5000, n_orders),
            "discount_amount": np.random.uniform(0, 500, n_orders),
            "shipping_cost": np.random.uniform(5, 50, n_orders),
            "status": np.random.choice(
                ["Pending", "Shipped", "Delivered", "Cancelled"],
                n_orders,
                p=[0.1, 0.2, 0.6, 0.1],
            ),
            "payment_method": np.random.choice(
                ["Credit Card", "PayPal", "Bank Transfer"], n_orders
            ),
        }
    )

    print("Creating order items dataset...")
    # Order Items table (junction table)
    order_items = pd.DataFrame(
        {
            "order_item_id": range(1, n_order_items + 1),
            "order_id": np.random.randint(1, n_orders + 1, n_order_items),
            "product_id": np.random.randint(1, n_products + 1, n_order_items),
            "quantity": np.random.randint(1, 10, n_order_items),
            "unit_price": np.random.uniform(10, 1000, n_order_items),
            "discount_percentage": np.random.uniform(0, 0.3, n_order_items),
        }
    )

    print("Creating reviews dataset...")
    # Reviews table
    reviews = pd.DataFrame(
        {
            "review_id": range(1, n_reviews + 1),
            "customer_id": np.random.randint(1, n_customers + 1, n_reviews),
            "product_id": np.random.randint(1, n_products + 1, n_reviews),
            "order_id": np.random.randint(1, n_orders + 1, n_reviews),
            "rating": np.random.randint(1, 6, n_reviews),
            "review_text": [f"Review text {i}" for i in range(1, n_reviews + 1)],
            "review_date": pd.date_range("2020-02-01", periods=n_reviews, freq="2H"),
            "helpful_votes": np.random.randint(0, 100, n_reviews),
        }
    )

    print("Creating time series dataset...")
    # Time series data for more complex operations
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    time_series = pd.DataFrame(
        {
            "date": dates,
            "sales": np.random.normal(10000, 2000, len(dates))
            + 1000
            * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25),  # seasonal pattern
            "marketing_spend": np.random.normal(5000, 1000, len(dates)),
            "temperature": 20
            + 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
            + np.random.normal(0, 5, len(dates)),
            "website_visits": np.random.poisson(50000, len(dates)),
            "conversion_rate": np.random.beta(2, 8, len(dates)),
        }
    )

    # Save all datasets as parquet files
    print("Saving datasets to parquet files...")
    customers.to_parquet("data/customers.parquet", index=False)
    products.to_parquet("data/products.parquet", index=False)
    orders.to_parquet("data/orders.parquet", index=False)
    order_items.to_parquet("data/order_items.parquet", index=False)
    reviews.to_parquet("data/reviews.parquet", index=False)
    time_series.to_parquet("data/time_series.parquet", index=False)

    # Create some additional datasets with different characteristics
    print("Creating additional datasets for diverse benchmarks...")

    # Wide dataset (many columns)
    wide_data = pd.DataFrame({f"col_{i}": np.random.randn(50000) for i in range(100)})
    wide_data["id"] = range(50000)
    wide_data.to_parquet("data/wide_data.parquet", index=False)

    # Text-heavy dataset
    text_data = pd.DataFrame(
        {
            "id": range(100000),
            "text_col_1": [
                f"This is a long text string number {i} with many words" * 10
                for i in range(100000)
            ],
            "text_col_2": [f"Another text column {i}" * 5 for i in range(100000)],
            "category": np.random.choice(["A", "B", "C", "D", "E"], 100000),
            "value": np.random.randn(100000),
        }
    )
    text_data.to_parquet("data/text_data.parquet", index=False)

    print("Data preparation completed successfully!")
    print("Created datasets:")
    print(f"- customers: {len(customers):,} rows")
    print(f"- products: {len(products):,} rows")
    print(f"- orders: {len(orders):,} rows")
    print(f"- order_items: {len(order_items):,} rows")
    print(f"- reviews: {len(reviews):,} rows")
    print(f"- time_series: {len(time_series):,} rows")
    print(f"- wide_data: {len(wide_data):,} rows")
    print(f"- text_data: {len(text_data):,} rows")


def load_data(df_lib: Any) -> Dict[str, Any]:
    """Load all datasets"""
    print(f"Loading data with {df_lib.__name__}")
    data = {}
    data["customers"] = df_lib.read_parquet("data/customers.parquet")
    data["products"] = df_lib.read_parquet("data/products.parquet")
    data["orders"] = df_lib.read_parquet("data/orders.parquet")
    data["order_items"] = df_lib.read_parquet("data/order_items.parquet")
    data["reviews"] = df_lib.read_parquet("data/reviews.parquet")
    data["time_series"] = df_lib.read_parquet("data/time_series.parquet")
    data["wide_data"] = df_lib.read_parquet("data/wide_data.parquet")
    data["text_data"] = df_lib.read_parquet("data/text_data.parquet")
    return data


if __name__ == "__main__":
    create_datasets()
