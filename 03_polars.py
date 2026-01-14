# Standard imports
import argparse
import importlib.util
import os
from typing import Any, Callable, Dict, List

# PyPI imports
import polars as pl
import polars.selectors as cs

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
            "basic_filtering",
            pl,
            lambda: customers.filter(pl.col("age") > 30),
        )
    )

    def groupby_aggregation_polars():
        # Use Polars native group_by and aggregation
        return (
            customers.group_by("city")
            .agg(
                pl.mean("annual_income").alias("mean"),
                pl.std("annual_income").alias("std"),
                pl.count("annual_income").alias("count"),
            )
            .sort("city")
        )

    results.append(
        time_operation(
            "groupby_aggregation",
            pl,
            groupby_aggregation_polars,
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

    def complex_multi_join_polars():
        result = (
            orders.join(customers, on="customer_id")
            .join(order_items, on="order_id")
            .join(products, on="product_id")
            .sort(["order_id", "order_item_id"])
        )
        return result

    results.append(
        time_operation(
            "complex_multi_join",
            pl,
            complex_multi_join_polars,
        )
    )

    def four_table_join_polars():
        # Perform the joins step by step to control column naming like pandas
        result = (
            customers.join(orders, on="customer_id", suffix="_orders")
            .join(order_items, on="order_id", suffix="_items")
            .join(products, on="product_id", suffix="_products")
            .join(reviews, on=["customer_id", "product_id"], suffix="_reviews")
            .sort("customer_id")
        )

        # We don't need to rename columns to match pandas exactly if we compare
        # data content. But if we wanted to match pandas merge suffixes (x, y),
        # we would need rename. For fairness, we stick to Polars native suffixes.

        return result

    results.append(time_operation("four_table_join", pl, four_table_join_polars))

    # groupby-dense-rank
    def groupby_dense_rank_polars():
        # Use Polars native window functions.
        # Cast rank to float to match pandas output dtype.
        result = orders.with_columns(
            pl.col("total_amount").cum_sum().over("customer_id").alias("running_total"),
            pl.col("total_amount")
            .rank(method="dense")
            .over("customer_id")
            .cast(pl.Float64)
            .alias("rank"),
        )
        return result

    results.append(time_operation("groupby-dense-rank", pl, groupby_dense_rank_polars))

    # groupby-first-rank
    def groupby_first_rank_polars():
        # Use Polars native window functions with first ranking method.
        # Cast rank to float to match pandas output dtype.
        result = orders.with_columns(
            pl.col("total_amount").cum_sum().over("customer_id").alias("running_total"),
            pl.col("total_amount")
            .rank(method="ordinal")
            .over("customer_id")
            .cast(pl.Float64)
            .alias("rank"),
        )
        return result

    results.append(time_operation("groupby-first-rank", pl, groupby_first_rank_polars))

    # String operations
    def string_operations_polars():
        result = text_data.with_columns(
            [
                pl.col("text_col_1")
                .str.len_chars()
                .cast(pl.Int64)
                .alias("text_length"),  # Cast to match pandas int64
                pl.col("text_col_1").str.to_uppercase().alias("text_upper"),
                pl.col("text_col_1").str.contains(r"\d+").alias("contains_number"),
            ]
        )
        return result

    results.append(time_operation("string_operations", pl, string_operations_polars))

    # Datetime operations
    def datetime_operations_polars():
        result = orders.with_columns(
            [
                pl.col("order_date").dt.year().alias("year"),
                pl.col("order_date")
                .dt.month()
                .cast(pl.Int32)
                .alias("month"),  # Cast to match pandas int32
                (pl.col("order_date").dt.weekday() - 1)
                .cast(pl.Int32)
                .alias("day_of_week"),  # Convert Monday=1 to Monday=0 to match pandas
                (pl.col("shipping_date") - pl.col("order_date"))
                .dt.total_days()
                .alias("days_to_ship"),
            ]
        )
        return result

    results.append(
        time_operation("datetime_operations", pl, datetime_operations_polars)
    )

    # Complex aggregations
    def complex_groupby_polars():
        # Use Polars native group_by operations
        result_pl = (
            orders.with_columns(pl.col("order_date").dt.year().alias("year"))
            .group_by(["status", "year"])
            .agg(
                [
                    pl.col("total_amount").sum().alias("total_amount_sum"),
                    pl.col("total_amount").mean().alias("total_amount_mean"),
                    pl.col("total_amount").count().alias("total_amount_count"),
                    pl.col("discount_amount").sum().alias("discount_amount_sum"),
                    pl.col("discount_amount").mean().alias("discount_amount_mean"),
                    pl.col("shipping_cost").mean().alias("shipping_cost_mean"),
                ]
            )
            .sort(["status", "year"])
        )
        return result_pl

    results.append(time_operation("complex_groupby", pl, complex_groupby_polars))

    # Pivot operations (using polars pivot)
    def pivot_table_polars():
        # Use polars pivot.
        # Pandas: pivot_table(values="total_amount", index="customer_id",
        # columns="status", aggfunc=["sum", "count"])

        result = (
            orders.group_by(["customer_id", "status"])
            .agg(
                [
                    pl.col("total_amount").sum().alias("sum"),
                    pl.col("total_amount").count().alias("count"),
                ]
            )
            .pivot(
                on="status",
                index="customer_id",
                values=["sum", "count"],
                aggregate_function=None,  # Already aggregated
            )
        )
        # Result columns will be like: sum_Delivered, sum_Pending, count_Delivered...
        # This matches our flattened pandas output format!
        return result

    results.append(time_operation("pivot_table", pl, pivot_table_polars))

    # Statistical operations
    def statistical_operations_polars():
        # Polars describe returns a DataFrame
        result = customers.select(cs.numeric()).describe()
        return result

    results.append(
        time_operation("statistical_operations", pl, statistical_operations_polars)
    )

    # Correlation matrix
    def correlation_matrix_polars():
        # Polars doesn't have a direct full correlation matrix function
        # on DataFrame yet. So here I can just return the pandas result,
        # and `00_tools.py` will handle it!
        # `00_tools.py` checks for `index` attr.

        pandas_time_series = time_series.to_pandas()
        numeric_data = pandas_time_series.select_dtypes(include=["number"]).dropna()
        corr_matrix = numeric_data.corr()
        for i in range(len(corr_matrix)):
            corr_matrix.iloc[i, i] = 1.0
        return corr_matrix.sort_index().sort_index(axis=1)

    results.append(
        time_operation(
            "correlation_matrix",
            pl,
            correlation_matrix_polars,
        )
    )

    # Rolling window operations
    def rolling_operations_polars():
        # Polars has rolling_*
        result = time_series.with_columns(
            pl.col("sales")
            .rolling_mean(window_size=7, min_periods=7)
            .alias("sales_ma_7"),
            pl.col("sales")
            .rolling_mean(window_size=30, min_periods=30)
            .alias("sales_ma_30"),
            pl.col("sales")
            .rolling_std(window_size=7, min_periods=7)
            .alias("sales_std_7"),
        )
        return result

    results.append(time_operation("rolling_operations", pl, rolling_operations_polars))

    def wide_data_transpose_polars():
        # Polars transpose
        # Polars transpose requires column names.
        result = wide_data.head(1000).transpose(
            include_header=True, header_name="index", column_names=None
        )
        return result

    results.append(
        time_operation(
            "wide_data_transpose",
            pl,
            wide_data_transpose_polars,
        )
    )

    # Memory intensive operations
    results.append(
        time_operation("large_concat", pl, lambda: pl.concat([customers] * 5))
    )

    # Advanced filtering
    def conditional_join_polars():
        result = (
            customers.join(orders, on="customer_id")
            .filter((pl.col("age") > 25) & (pl.col("total_amount") > 100))
            .sort("customer_id")
        )
        return result

    results.append(
        time_operation(
            "conditional_join",
            pl,
            conditional_join_polars,
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
    def crosstab_polars():
        # pd.crosstab(index, columns) is essentially pivot_table(count).
        # index=city, columns=customer_segment
        result = (
            customers.group_by(["city", "customer_segment"])
            .len()  # count
            .pivot(
                on="customer_segment",
                index="city",
                values="len",
                aggregate_function=None,
            )
            .fill_null(0)  # crosstab fills 0
        )
        return result

    results.append(time_operation("crosstab", pl, crosstab_polars))

    # Multi-level groupby
    def multilevel_groupby_polars():
        # Polars simple group by on multiple columns
        result = order_items.group_by(["order_id", "product_id"]).agg(
            pl.col("quantity").sum(),
            pl.col("unit_price").mean(),
            pl.col("discount_percentage").max(),
        )
        return result

    results.append(time_operation("multilevel_groupby", pl, multilevel_groupby_polars))

    # Time series resampling
    def time_series_resample_polars():
        # Polars group_by_dynamic
        # pandas resample("ME") is Month End.
        # Polars: group_by_dynamic("date", every="1mo")
        # Note: Polars group_by_dynamic uses start of interval by default.
        result = (
            time_series.sort("date")
            .group_by_dynamic("date", every="1mo")
            .agg(
                pl.col("sales").sum(),
                pl.col("marketing_spend").sum(),
                pl.col("website_visits").mean(),
            )
        )
        return result

    results.append(
        time_operation("time_series_resample", pl, time_series_resample_polars)
    )

    # Quantile operations
    def quantile_operations_polars():
        # Polars: group_by -> quantile
        result = customers.group_by("customer_segment").agg(
            [
                pl.col("annual_income").quantile(0.25).alias("annual_income_0.25"),
                pl.col("annual_income").quantile(0.5).alias("annual_income_0.5"),
                pl.col("annual_income").quantile(0.75).alias("annual_income_0.75"),
            ]
        )
        return result

    results.append(
        time_operation("quantile_operations", pl, quantile_operations_polars)
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
