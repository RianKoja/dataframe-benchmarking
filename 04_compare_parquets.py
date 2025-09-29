from pathlib import Path

import pandas as pd

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
    Compares two DataFrames and prints differences in markdown format.

    Returns:
        bool: True if DataFrames are equal, False otherwise.
    """
    print(f"Comparing {label1} vs {label2}:")

    # Check that columns match
    if (set1 := set(df1.columns)) != (set2 := set(df2.columns)):
        print("⚠️ Columns differ:")
        print(f"  Only in {label1}: {set1 - set2}")
        print(f"  Only in {label2}: {set2 - set1}")
        print()
        return False

    # Check that length matches
    if len(df1) != len(df2):
        print(f"⚠️ Row count differs: {len(df1)} vs {len(df2)}")
        print()
        return False

    # Check that data types match for common columns
    type_mismatches = []
    for col in df1.columns:
        if df1[col].dtype != df2[col].dtype:
            type_mismatches.append((col, df1[col].dtype, df2[col].dtype))

    if type_mismatches:
        print("⚠️ Data type mismatches:")
        for col, dtype1, dtype2 in type_mismatches:
            print(f"  {col}: {dtype1} vs {dtype2}")
        print()
        return False

    # Check index equality
    if not df1.index.equals(df2.index):
        print("⚠️ Index differs between DataFrames")
        print()
        return False

    # Compare actual values with numerical tolerance for floating point columns
    # The compare method requires dataframes to be sorted for consistent results
    # if the index is not aligned. We assume the index is meaningful and don't sort.
    try:
        # First check if they are exactly equal
        if df1.equals(df2):
            print(f"✅ No differences found between {label1} and {label2}")
            print()
            return True

        # For numerical differences, use tolerance-based comparison
        import numpy as np

        # Check if differences are within numerical tolerance
        numerical_match = True
        significant_diffs = []

        for col in df1.columns:
            col_data1 = df1[col]
            col_data2 = df2[col]

            # For numeric columns, use tolerance-based comparison
            if pd.api.types.is_numeric_dtype(
                col_data1
            ) and pd.api.types.is_numeric_dtype(col_data2):
                # Handle NaN values explicitly
                both_not_nan = ~col_data1.isna() & ~col_data2.isna()

                # Check if NaN patterns match
                nan_mask1 = col_data1.isna()
                nan_mask2 = col_data2.isna()
                if not nan_mask1.equals(nan_mask2):
                    numerical_match = False
                    significant_diffs.append(f"NaN pattern differs in column '{col}'")
                    continue

                # For non-NaN values, check numerical tolerance
                if both_not_nan.any():
                    vals1 = col_data1[both_not_nan].values
                    vals2 = col_data2[both_not_nan].values

                    # Use relative and absolute tolerance
                    if not np.allclose(
                        vals1, vals2, rtol=1e-10, atol=1e-12, equal_nan=True
                    ):
                        max_diff = np.abs(vals1 - vals2).max()
                        # Only consider significant if difference is large
                        if max_diff > 1e-8:
                            numerical_match = False
                            significant_diffs.append(
                                f"Significant numerical differences in column '{col}' "
                                f"(max diff: {max_diff})"
                            )
            else:
                # For non-numeric columns, must be exactly equal
                if not col_data1.equals(col_data2):
                    numerical_match = False
                    significant_diffs.append(
                        f"Non-numeric differences in column '{col}'"
                    )

        if numerical_match:
            if significant_diffs:
                print(
                    f"⚠️ Minor numerical differences found between {label1} and "
                    f"{label2} (within tolerance)"
                )
                for diff in significant_diffs[:3]:  # Show first 3 differences
                    print(f"  {diff}")
                print()
            else:
                print(
                    f"✅ No significant differences found between {label1} and "
                    f"{label2} (within numerical tolerance)"
                )
                print()
            return True
        else:
            print(f"⚠️ Significant differences found between {label1} and {label2}:")
            for diff in significant_diffs[:5]:  # Show first 5 significant differences
                print(f"  {diff}")

            # Show detailed comparison for debugging
            diff = df1.compare(df2, align_axis=1)
            if not diff.empty:
                print("\nDetailed comparison (first 5 rows):")
                print(diff.head().to_markdown())
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
    raise Exception("Some comparison failed check above.")
