import hashlib
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict


def normalize_result(result: Any) -> Any:
    """
    Normalize the result DataFrame to ensure consistent format across frameworks.
    - Resets index (moving index to columns)
    - Flattens MultiIndex columns
    """
    # Check if it looks like a pandas/fireducks DataFrame
    if (
        hasattr(result, "index")
        and hasattr(result, "columns")
        and hasattr(result, "reset_index")
    ):
        # 1. Reset index if it's not a RangeIndex
        # This moves grouping keys from index to columns
        is_range_index = False
        # Check for RangeIndex (has start/stop/step attributes)
        if (
            hasattr(result.index, "start")
            and hasattr(result.index, "stop")
            and hasattr(result.index, "step")
        ):
            is_range_index = True

        if not is_range_index:
            result = result.reset_index()

        # 2. Flatten MultiIndex columns
        # Check if columns is a MultiIndex (has nlevels > 1)
        if hasattr(result.columns, "nlevels") and result.columns.nlevels > 1:
            new_columns = []
            for col in result.columns.values:
                if isinstance(col, tuple):
                    # Join non-empty parts with underscore
                    # E.g. ('total_amount', 'sum') -> 'total_amount_sum'
                    name = "_".join([str(c) for c in col if str(c) != ""]).strip("_")
                    new_columns.append(name)
                else:
                    new_columns.append(str(col))
            result.columns = new_columns

    return result


def time_operation(
    operation_name: str,
    df_lib: Any,
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Time a function and return the result and execution info"""

    framework = df_lib.__name__
    if framework == "fireducks.pandas":
        framework = "fireducks"

    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    # Force evaluation for lazy operations
    if hasattr(result, "_evaluate"):
        result = result._evaluate()
    elif hasattr(result, "compute"):
        result = result.compute()
    elif hasattr(result, "values"):
        _ = result.values  # Access values to force computation
    elif isinstance(result, df_lib.DataFrame):
        _ = len(result)  # Force evaluation by accessing length
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    success = True
    error_msg = None

    # Save result to parquet file and compute hash
    result_hash = None
    if result is None:
        raise ValueError("Operation returned None, which is not allowed.")

    # Create outputs/results directory if it doesn't exist
    results_dir = "outputs/results"
    os.makedirs(results_dir, exist_ok=True)

    # Save result to parquet file
    output_filename = f"{results_dir}/{operation_name}_{framework}.parquet"

    if hasattr(result, "to_frame"):
        result = result.to_frame(name=operation_name)

    # Normalize result (reset index, flatten columns) before saving
    # This ensures consistency between Pandas (which uses Index/MultiIndex)
    # and Polars (which uses flat DataFrames)
    try:
        result = normalize_result(result)
    except Exception as e:
        print(f"Warning: Failed to normalize result for {operation_name}: {e}")

    if hasattr(result, "to_parquet"):
        result.to_parquet(output_filename, index=False)
    elif hasattr(result, "write_parquet"):
        result.write_parquet(output_filename)
    else:
        # Handle other types (scalars, arrays, etc.)
        # by converting to DataFrame
        if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
            # Iterable but not string
            temp_df = df_lib.DataFrame({"result": list(result)})
        else:
            # Scalar value
            temp_df = df_lib.DataFrame({"result": [result]})
        temp_df.to_parquet(output_filename, index=False)

    # Compute hash of the saved file for consistency verification
    with open(output_filename, "rb") as f:
        result_hash = hashlib.sha256(f.read()).hexdigest()

    return {
        "operation": operation_name,
        "framework": framework,
        "execution_time": execution_time,
        "success": success,
        "error": error_msg,
        "timestamp": datetime.now(),
        "result_hash": result_hash,
    }
