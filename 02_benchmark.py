# Standard imports:
import importlib
import os
import sys
from types import ModuleType
from typing import Any, Callable, Dict, List

# Local imports using importlib for numbered modules
tools_module = importlib.import_module("00_tools")
prep_data_module = importlib.import_module("01_prep_data")
# Import specific functions from the modules
time_operation: Callable[[str, ModuleType, Callable[..., Any]], Dict[str, Any]] = (
    getattr(tools_module, "time_operation")
)
load_data: Callable[[ModuleType], Dict[str, Any]] = getattr(
    prep_data_module, "load_data"
)

# Dynamic import based on command line argument
if "fireducks" in sys.argv:
    import fireducks.pandas as df_lib

    framework = "fireducks"

elif "pandas" in sys.argv:
    import pandas as df_lib

    framework = "pandas"

else:
    raise ValueError("Please specify 'pandas' or 'fireducks' as argument")


def run_benchmarks() -> List[Dict[str, Any]]:
    """Run comprehensive benchmarks"""
    results = []

    data = load_data(df_lib)

    # Extract dataframes
    customers = data["customers"]
    products = data["products"]
    orders = data["orders"]
    order_items = data["order_items"]
    reviews = data["reviews"]
    time_series = data["time_series"]
    wide_data = data["wide_data"]
    text_data = data["text_data"]

    print(f"Running benchmarks with {framework}...")

    # Basic operations
    results.append(
        time_operation(
            "basic_filtering", df_lib, lambda: customers[customers["age"] > 30]
        )
    )

    def groupby_aggregation_operation():
        # Ensure consistent groupby behavior by sorting result
        result = customers.groupby("city")["annual_income"].agg(
            ["mean", "std", "count"]
        )
        return result.sort_index()

    results.append(
        time_operation(
            "groupby_aggregation",
            df_lib,
            groupby_aggregation_operation,
        )
    )

    results.append(
        time_operation(
            "sorting",
            df_lib,
            lambda: orders.sort_values(
                ["order_date", "total_amount"], ascending=[True, False]
            ),
        )
    )

    # Join operations (multiple types)
    results.append(
        time_operation(
            "simple_inner_join",
            df_lib,
            lambda: orders.merge(customers, on="customer_id", how="inner"),
        )
    )

    results.append(
        time_operation(
            "left_join",
            df_lib,
            lambda: orders.merge(customers, on="customer_id", how="left"),
        )
    )

    results.append(
        time_operation(
            "complex_multi_join",
            df_lib,
            lambda: orders.merge(customers, on="customer_id")
            .merge(order_items, on="order_id")
            .merge(products, on="product_id")
            .sort_values(["order_id", "order_item_id"])
            .reset_index(drop=True),
        )
    )

    results.append(
        time_operation(
            "four_table_join",
            df_lib,
            lambda: customers.merge(orders, on="customer_id")
            .merge(order_items, on="order_id")
            .merge(products, on="product_id")
            .merge(reviews, on=["customer_id", "product_id"])
            .sort_values("customer_id")
            .reset_index(drop=True),
        )
    )

    # Window functions
    results.append(
        time_operation(
            "window_functions",
            df_lib,
            lambda: orders.assign(
                running_total=orders.groupby("customer_id")["total_amount"].cumsum(),
                rank=orders.groupby("customer_id")["total_amount"].rank(method="dense"),
            ),
        )
    )

    # String operations
    results.append(
        time_operation(
            "string_operations",
            df_lib,
            lambda: text_data.assign(
                text_length=text_data["text_col_1"].str.len(),
                text_upper=text_data["text_col_1"].str.upper(),
                contains_number=text_data["text_col_1"].str.contains(r"\d+"),
            ),
        )
    )

    # Datetime operations
    results.append(
        time_operation(
            "datetime_operations",
            df_lib,
            lambda: orders.assign(
                year=orders["order_date"].dt.year,
                month=orders["order_date"].dt.month,
                day_of_week=orders["order_date"].dt.dayofweek,
                days_to_ship=(orders["shipping_date"] - orders["order_date"]).dt.days,
            ),
        )
    )

    # Complex aggregations
    def complex_groupby_operation():
        # Ensure consistent groupby behavior by explicitly handling missing groups
        result = orders.groupby(["status", orders["order_date"].dt.year]).agg(
            {
                "total_amount": ["sum", "mean", "count"],
                "discount_amount": ["sum", "mean"],
                "shipping_cost": "mean",
            }
        )
        # Sort by index to ensure consistent ordering
        return result.sort_index()

    results.append(
        time_operation(
            "complex_groupby",
            df_lib,
            complex_groupby_operation,
        )
    )

    # Pivot operations
    results.append(
        time_operation(
            "pivot_table",
            df_lib,
            lambda: orders.pivot_table(
                values="total_amount",
                index="customer_id",
                columns="status",
                aggfunc=["sum", "count"],
                fill_value=0,
            ),
        )
    )

    # Statistical operations
    results.append(
        time_operation(
            "statistical_operations",
            df_lib,
            lambda: customers.select_dtypes(include=["number"]).describe(),
        )
    )

    def correlation_matrix_operation():
        # Ensure consistent correlation matrix by explicitly dropping NaN and sorting
        numeric_data = time_series.select_dtypes(include=["number"]).dropna()
        corr_matrix = numeric_data.corr()
        # Fill diagonal with 1.0 explicitly to ensure consistency
        for i in range(len(corr_matrix)):
            corr_matrix.iloc[i, i] = 1.0
        return corr_matrix.sort_index().sort_index(axis=1)

    results.append(
        time_operation(
            "correlation_matrix",
            df_lib,
            correlation_matrix_operation,
        )
    )

    # Rolling window operations
    def rolling_operations_func():
        # Ensure consistent rolling operations by explicitly handling NaN values
        result = time_series.assign(
            sales_ma_7=time_series["sales"].rolling(window=7, min_periods=7).mean(),
            sales_ma_30=time_series["sales"].rolling(window=30, min_periods=30).mean(),
            sales_std_7=time_series["sales"].rolling(window=7, min_periods=7).std(),
        )
        return result

    results.append(
        time_operation(
            "rolling_operations",
            df_lib,
            rolling_operations_func,
        )
    )

    results.append(
        time_operation("wide_data_transpose", df_lib, lambda: wide_data.head(1000).T)
    )

    # Memory intensive operations
    results.append(
        time_operation(
            "large_concat",
            df_lib,
            lambda: df_lib.concat([customers] * 5, ignore_index=True),
        )
    )

    # Advanced joins with conditions
    results.append(
        time_operation(
            "conditional_join",
            df_lib,
            lambda: customers.merge(orders, on="customer_id")
            .query("age > 25 and total_amount > 100")
            .sort_values("customer_id")
            .reset_index(drop=True),
        )
    )

    # Complex filtering
    results.append(
        time_operation(
            "complex_filtering",
            df_lib,
            lambda: orders[
                (orders["total_amount"] > orders["total_amount"].quantile(0.75))
                & (orders["status"] == "Delivered")
                & (orders["order_date"] >= "2021-01-01")
            ],
        )
    )

    # Cross tabulation (if supported)
    crosstab_func = getattr(df_lib, "crosstab", None)
    if crosstab_func is not None:
        results.append(
            time_operation(
                "crosstab",
                df_lib,
                lambda: crosstab_func(customers["city"], customers["customer_segment"]),
            )
        )

    # Multi-level groupby
    results.append(
        time_operation(
            "multilevel_groupby",
            df_lib,
            lambda: order_items.groupby(["order_id", "product_id"]).agg(
                {"quantity": "sum", "unit_price": "mean", "discount_percentage": "max"}
            ),
        )
    )

    # Time series resampling
    results.append(
        time_operation(
            "time_series_resample",
            df_lib,
            lambda: time_series.set_index("date")
            .resample("ME")
            .agg({"sales": "sum", "marketing_spend": "sum", "website_visits": "mean"}),
        )
    )

    # Quantile operations
    results.append(
        time_operation(
            "quantile_operations",
            df_lib,
            lambda: customers.groupby("customer_segment")["annual_income"].quantile(
                [0.25, 0.5, 0.75]
            ),
        )
    )

    return results


def main() -> None:
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)

    cache_used = "--cache" in sys.argv

    # Run benchmarks
    results = run_benchmarks()

    # Convert results to DataFrame
    results_df = df_lib.DataFrame(results)
    results_df["cache_used"] = cache_used

    # Save results
    cache_suffix = "_cache" if cache_used else "_no_cache"
    output_file = f"outputs/{framework}{cache_suffix}_results.parquet"
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
