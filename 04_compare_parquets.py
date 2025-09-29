from pathlib import Path

import pandas as pd
import pandas.testing as pd_testing

# Define the relative path to the results folder
folder_path = Path(__file__).parent / "outputs" / "results"


def group_files_by_operation(path):
    """Groups parquet files by their base operation name."""
    # Define supported frameworks
    frameworks = ["fireducks", "pandas", "polars"]

    files_grouped = {}
    for file in path.glob("*.parquet"):
        stem = file.stem

        # Find which framework this file belongs to
        framework_found = None
        operation_key = None

        for framework in frameworks:
            suffix = f"_{framework}"
            if stem.endswith(suffix):
                framework_found = framework
                operation_key = stem.removesuffix(suffix)
                break

        if framework_found is None:
            raise Exception(f"Unknown or missing framework in file: {file.name}")

        files_grouped.setdefault(operation_key, {})[framework_found] = file

    return files_grouped


def compare_dataframes(df1, df2, label1, label2):
    """
    Compares two DataFrames using pandas.testing.assert_frame_equal.

    Returns:
        bool: True if DataFrames are equal, False otherwise.
    """
    print(f"Comparing {label1} vs {label2}:")

    try:
        pd_testing.assert_frame_equal(
            df1,
            df2,
            check_dtype=False,  # Allow dtype differences like int64 vs uint32
            check_index_type=True,
            check_column_type=True,
            check_frame_type=True,
            check_names=True,
            rtol=1e-10,  # Relative tolerance for numerical comparisons
            atol=1e-12,  # Absolute tolerance for numerical comparisons
            check_exact=False,  # Allow numerical tolerance
        )
        print(f"✅ DataFrames are equal between {label1} and {label2}")
        print()
        return True

    except AssertionError as e:
        print(f"⚠️ DataFrames differ between {label1} and {label2}:")
        print(f"  {str(e)}")

        # Show detailed comparison for debugging if DataFrames have same structure
        try:
            if df1.shape == df2.shape and list(df1.columns) == list(df2.columns):
                diff = df1.compare(df2, align_axis=1)
                if not diff.empty:
                    print("\nDetailed comparison (first 5 rows):")
                    print(diff.head().to_markdown())
        except Exception:
            pass  # Skip detailed comparison if it fails

        print()
        return False

    except Exception as e:
        print(f"⚠️ Error during comparison: {e}")
        print()
        return False


# --- Main Execution ---
if not folder_path.exists():
    raise Exception(f"Error: Directory not found at '{folder_path}'")

# Group files by the test operation
grouped_files = group_files_by_operation(folder_path)

# Iterate through each operation and perform comparisons
success = True
for key, fdict in sorted(grouped_files.items()):
    print(f"--- Comparing results for: {key} ---")

    # Ensure the base pandas file exists for comparison
    assert "pandas" in fdict

    df_pandas = pd.read_parquet(fdict["pandas"])

    # Compare pandas with fireducks
    df_fireducks = pd.read_parquet(fdict["fireducks"])
    success &= compare_dataframes(
        df_pandas, df_fireducks, f"{key} (pandas)", f"{key} (fireducks)"
    )

    # Compare pandas with polars
    # Load polars-generated parquet into a pandas DataFrame for comparison
    df_polars = pd.read_parquet(fdict["polars"])
    success &= compare_dataframes(
        df_pandas, df_polars, f"{key} (pandas)", f"{key} (polars)"
    )

if not success:
    raise AssertionError(
        "One or more benchmark result comparisons failed. Check logs above for details."
    )
