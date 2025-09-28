# Standard imports
import argparse
import importlib.util
import os
from typing import Any, Callable, Dict, List

# PyPI imports
import polars as pl

# Local imports using importlib for numbered modules
tools_module = importlib.import_module("00_tools")
prep_data_module = importlib.import_module("01_prep_data")
# Import specific functions from the modules
time_operation: Callable[[str, Any, Callable[..., Any]], Dict[str, Any]] = getattr(
    tools_module, "time_operation"
)
load_data: Callable[[Any], Dict[str, Any]] = getattr(prep_data_module, "load_data")


def run_benchmarks(
    data: Dict[str, Any], use_cache: bool = False
) -> List[Dict[str, Any]]:
    """Run comprehensive benchmarks using polars (including lazy evaluation)"""
    results = []

    # Extract dataframes
    customers = data["customers"]
    products = data["products"]
    orders = data["orders"]
    order_items = data["order_items"]
    reviews = data["reviews"]
    time_series = data["time_series"]
    wide_data = data["wide_data"]
    text_data = data["text_data"]

    print("Running benchmarks with polars...")

    # Basic operations (eager)
    results.append(
        time_operation(
            "basic_filtering", pl, lambda: customers.filter(pl.col("age") > 30)
        )
    )

    results.append(
        time_operation(
            "groupby_aggregation",
            pl,
            lambda: customers.group_by("city").agg(
                [
                    pl.col("annual_income").mean().alias("mean"),
                    pl.col("annual_income").std().alias("std"),
                    pl.col("annual_income").count().alias("count"),
                ]
            ),
        )
    )

    results.append(
        time_operation(
            "sorting",
            pl,
            lambda: orders.sort(
                ["order_date", "total_amount"], descending=[False, True]
            ),
        )
    )

    # Join operations
    results.append(
        time_operation(
            "simple_inner_join",
            pl,
            lambda: orders.join(customers, on="customer_id", how="inner"),
        )
    )

    results.append(
        time_operation(
            "left_join",
            pl,
            lambda: orders.join(customers, on="customer_id", how="left"),
        )
    )

    results.append(
        time_operation(
            "complex_multi_join",
            pl,
            lambda: orders.join(customers, on="customer_id")
            .join(order_items, on="order_id")
            .join(products, on="product_id"),
        )
    )

    results.append(
        time_operation(
            "four_table_join",
            pl,
            lambda: customers.join(orders, on="customer_id")
            .join(order_items, on="order_id")
            .join(products, on="product_id")
            .join(reviews, on=["customer_id", "product_id"]),
        )
    )

    # Window functions
    results.append(
        time_operation(
            "window_functions",
            pl,
            lambda: orders.with_columns(
                [
                    pl.col("total_amount")
                    .cum_sum()
                    .over("customer_id")
                    .alias("running_total"),
                    pl.col("total_amount")
                    .rank(method="dense")
                    .over("customer_id")
                    .alias("rank"),
                ]
            ),
        )
    )

    # String operations
    results.append(
        time_operation(
            "string_operations",
            pl,
            lambda: text_data.with_columns(
                [
                    pl.col("text_col_1").str.len_chars().alias("text_length"),
                    pl.col("text_col_1").str.to_uppercase().alias("text_upper"),
                    pl.col("text_col_1").str.contains(r"\d+").alias("contains_number"),
                ]
            ),
        )
    )

    # Datetime operations
    results.append(
        time_operation(
            "datetime_operations",
            pl,
            lambda: orders.with_columns(
                [
                    pl.col("order_date").dt.year().alias("year"),
                    pl.col("order_date").dt.month().alias("month"),
                    pl.col("order_date").dt.weekday().alias("day_of_week"),
                    (pl.col("shipping_date") - pl.col("order_date"))
                    .dt.total_days()
                    .alias("days_to_ship"),
                ]
            ),
        )
    )

    # Complex aggregations
    results.append(
        time_operation(
            "complex_groupby",
            pl,
            lambda: orders.group_by(["status", pl.col("order_date").dt.year()]).agg(
                [
                    pl.col("total_amount").sum().alias("total_amount_sum"),
                    pl.col("total_amount").mean().alias("total_amount_mean"),
                    pl.col("total_amount").count().alias("total_amount_count"),
                    pl.col("discount_amount").sum().alias("discount_amount_sum"),
                    pl.col("discount_amount").mean().alias("discount_amount_mean"),
                    pl.col("shipping_cost").mean().alias("shipping_cost_mean"),
                ]
            ),
        )
    )

    # Pivot operations (using polars pivot)
    results.append(
        time_operation(
            "pivot_table",
            pl,
            lambda: orders.pivot(
                values="total_amount",
                index="customer_id",
                on="status",
                aggregate_function="sum",
            ),
        )
    )

    # Statistical operations
    results.append(
        time_operation("statistical_operations", pl, lambda: customers.describe())
    )

    # Correlation matrix (select numeric columns)
    numeric_cols = [
        col
        for col in time_series.columns
        if time_series[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
    ]
    if numeric_cols:
        results.append(
            time_operation(
                "correlation_matrix",
                pl,
                lambda: time_series.select(numeric_cols).corr(),
            )
        )

    # Rolling window operations
    results.append(
        time_operation(
            "rolling_operations",
            pl,
            lambda: time_series.sort("date").with_columns(
                [
                    pl.col("sales").rolling_mean(window_size=7).alias("sales_ma_7"),
                    pl.col("sales").rolling_mean(window_size=30).alias("sales_ma_30"),
                    pl.col("sales").rolling_std(window_size=7).alias("sales_std_7"),
                ]
            ),
        )
    )

    results.append(
        time_operation(
            "wide_data_transpose", pl, lambda: wide_data.head(1000).transpose()
        )
    )

    # Memory intensive operations
    results.append(
        time_operation("large_concat", pl, lambda: pl.concat([customers] * 5))
    )

    # Advanced filtering
    results.append(
        time_operation(
            "conditional_join",
            pl,
            lambda: customers.join(orders, on="customer_id").filter(
                (pl.col("age") > 25) & (pl.col("total_amount") > 100)
            ),
        )
    )

    # Complex filtering
    results.append(
        time_operation(
            "complex_filtering",
            pl,
            lambda: orders.filter(
                (pl.col("total_amount") > orders["total_amount"].quantile(0.75))
                & (pl.col("status") == "Delivered")
                & (pl.col("order_date") >= pl.datetime(2021, 1, 1))
            ),
        )
    )

    # Cross tabulation (using group_by and pivot)
    results.append(
        time_operation(
            "crosstab",
            pl,
            lambda: customers.group_by(["city", "customer_segment"])
            .len()
            .pivot(values="len", index="city", on="customer_segment"),
        )
    )

    # Multi-level groupby
    results.append(
        time_operation(
            "multilevel_groupby",
            pl,
            lambda: order_items.group_by(["order_id", "product_id"]).agg(
                [
                    pl.col("quantity").sum().alias("quantity_sum"),
                    pl.col("unit_price").mean().alias("unit_price_mean"),
                    pl.col("discount_percentage")
                    .max()
                    .alias("discount_percentage_max"),
                ]
            ),
        )
    )

    # Time series resampling
    results.append(
        time_operation(
            "time_series_resample",
            pl,
            lambda: time_series.group_by_dynamic("date", every="1mo").agg(
                [
                    pl.col("sales").sum().alias("sales_sum"),
                    pl.col("marketing_spend").sum().alias("marketing_spend_sum"),
                    pl.col("website_visits").mean().alias("website_visits_mean"),
                ]
            ),
        )
    )

    # Quantile operations
    results.append(
        time_operation(
            "quantile_operations",
            pl,
            lambda: customers.group_by("customer_segment").agg(
                [
                    pl.col("annual_income").quantile(0.25).alias("q25"),
                    pl.col("annual_income").quantile(0.5).alias("q50"),
                    pl.col("annual_income").quantile(0.75).alias("q75"),
                ]
            ),
        )
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Use cached data")
    args = parser.parse_args()

    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)

    # Load data
    data = load_data(pl)

    # Run benchmarks
    results = run_benchmarks(data, use_cache=args.cache)

    # Convert results to pandas DataFrame for consistency
    results_df = pl.DataFrame(results).to_pandas()
    results_df["cache_used"] = args.cache

    # Save results
    cache_suffix = "_cache" if args.cache else "_no_cache"
    output_file = f"outputs/polars{cache_suffix}_results.parquet"
    results_df.to_parquet(output_file, index=False)

    print(f"Benchmarks completed. Results saved to {output_file}")
    print(f"Total operations: {len(results)}")
    print(f"Successful operations: {sum(1 for r in results if r['success'])}")
    print(f"Failed operations: {sum(1 for r in results if not r['success'])}")

    # Print summary statistics
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        times = [r["execution_time"] for r in successful_results]
        print(f"Average execution time: {sum(times) / len(times):.4f} seconds")
        print(f"Total execution time: {sum(times):.4f} seconds")


if __name__ == "__main__":
    main()
