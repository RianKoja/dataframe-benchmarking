from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def load_benchmark_results() -> Dict[str, pd.DataFrame]:
    """Load all benchmark result files"""
    results_dir = Path("outputs")

    # Expected result files
    expected_files = [
        "pandas_no_cache_results.parquet",
        "pandas_cache_results.parquet",
        "fireducks_no_cache_results.parquet",
        "fireducks_cache_results.parquet",
        "polars_no_cache_results.parquet",
        "polars_cache_results.parquet",
    ]

    results = {}
    missing_files = []

    for file in expected_files:
        file_path = results_dir / file
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                results[file.replace("_results.parquet", "")] = df
                print(f"✓ Loaded {file}")
            except Exception as e:
                print(f"✗ Error loading {file}: {e}")
                missing_files.append(file)
        else:
            print(f"✗ Missing file: {file}")
            missing_files.append(file)

    if missing_files:
        print(
            f"\nWarning: {len(missing_files)} files are missing. "
            f"Analysis will continue with available data."
        )

    return results


def analyze_result_hashes(successful_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Analyze and compare result hashes across frameworks"""

    # Filter out rows without hashes (failed operations or old runs without
    # hash support)
    hash_df = successful_df[successful_df["result_hash"].notna()].copy()

    if len(hash_df) == 0:
        print(
            "No hash data available for comparison. Results may be from "
            "older benchmark runs."
        )
        return None

    # Create a pivot table of hashes by operation and framework
    hash_pivot = hash_df.pivot_table(
        index="operation",
        columns="framework",
        values="result_hash",
        aggfunc="first",  # Take first hash (should be same for cache/no-cache)
    )

    # Analyze consistency
    hash_analysis = []

    for operation in hash_pivot.index:
        operation_hashes = hash_pivot.loc[operation].dropna()

        if len(operation_hashes) > 1:
            # Check if all hashes are the same
            unique_hashes = operation_hashes.unique()
            is_consistent = len(unique_hashes) == 1

            hash_analysis.append(
                {
                    "operation": operation,
                    "consistent": is_consistent,
                    "num_frameworks": len(operation_hashes),
                    "unique_hashes": len(unique_hashes),
                    "frameworks": list(operation_hashes.index),
                    "hashes": dict(operation_hashes),
                }
            )

    return {
        "hash_pivot": hash_pivot,
        "analysis": hash_analysis,
        "consistency_summary": {
            "total_operations": len(hash_analysis),
            "consistent_operations": sum(1 for a in hash_analysis if a["consistent"]),
            "inconsistent_operations": sum(
                1 for a in hash_analysis if not a["consistent"]
            ),
        },
    }


def create_comparison_tables(
    results: Dict[str, pd.DataFrame],
) -> Optional[Dict[str, Any]]:
    """Create comprehensive comparison tables"""

    # Combine all results
    all_results = []
    for key, df in results.items():
        df_copy = df.copy()

        # Parse framework and cache info from filename
        parts = key.split("_")
        framework = parts[0]
        cache_status = "cache" if "cache" in parts else "no_cache"

        df_copy["framework"] = framework
        df_copy["cache_status"] = cache_status
        all_results.append(df_copy)

    if not all_results:
        print("No results to analyze!")
        return None

    combined_df = pd.concat(all_results, ignore_index=True)

    # Filter only successful operations
    successful_df = combined_df[combined_df["success"]].copy()

    if len(successful_df) == 0:
        print("No successful operations found!")
        return None

    print(
        f"\nAnalyzing {len(successful_df)} successful operations across "
        f"{successful_df['framework'].nunique()} frameworks"
    )

    # Create pivot tables for different views

    # 1. Main comparison table: Operation vs Framework+Cache
    pivot_df = successful_df.pivot_table(
        index="operation",
        columns=["framework", "cache_status"],
        values="execution_time",
        aggfunc="mean",
    )

    # 2. Summary statistics by framework
    summary_stats = (
        successful_df.groupby(["framework", "cache_status"])["execution_time"]
        .agg(["count", "mean", "median", "std", "min", "max", "sum"])
        .round(4)
    )

    # 3. Speed comparison (relative to pandas no cache)
    if ("pandas", "no_cache") in pivot_df.columns:
        baseline = pivot_df[("pandas", "no_cache")]
        speedup_df = pivot_df.div(baseline, axis=0)
        speedup_df.columns = [
            f"{fw}_{cache}_speedup" for fw, cache in speedup_df.columns
        ]
    else:
        speedup_df = None

    # 4. Cache effectiveness (speedup from caching)
    cache_effectiveness = {}
    frameworks = successful_df["framework"].unique()

    for fw in frameworks:
        no_cache_col = (fw, "no_cache")
        cache_col = (fw, "cache")

        if no_cache_col in pivot_df.columns and cache_col in pivot_df.columns:
            effectiveness = pivot_df[no_cache_col] / pivot_df[cache_col]
            cache_effectiveness[f"{fw}_cache_speedup"] = effectiveness

    if cache_effectiveness:
        cache_effectiveness_df = pd.DataFrame(cache_effectiveness)
    else:
        cache_effectiveness_df = None

    # 5. Hash comparison for result verification
    hash_comparison = analyze_result_hashes(successful_df)

    return {
        "main_comparison": pivot_df,
        "summary_stats": summary_stats,
        "speedup_comparison": speedup_df,
        "cache_effectiveness": cache_effectiveness_df,
        "raw_data": successful_df,
        "hash_comparison": hash_comparison,
    }


def print_analysis_report(tables: Dict[str, Any]) -> None:
    """Print comprehensive analysis report"""

    print("\n" + "=" * 80)
    print("DATAFRAME LIBRARY BENCHMARK ANALYSIS REPORT")
    print("=" * 80)

    main_comparison = tables["main_comparison"]
    summary_stats = tables["summary_stats"]
    speedup_comparison = tables["speedup_comparison"]
    cache_effectiveness = tables["cache_effectiveness"]
    raw_data = tables["raw_data"]
    hash_comparison = tables["hash_comparison"]

    # 1. Executive Summary
    print("\n1. EXECUTIVE SUMMARY")
    print("-" * 40)

    frameworks = raw_data["framework"].unique()
    total_operations = len(main_comparison)

    print(f"• Frameworks tested: {', '.join(frameworks)}")
    print(f"• Total operations benchmarked: {total_operations}")
    print("• Cache configurations: with and without cache")

    # Overall performance ranking
    overall_perf = summary_stats.groupby("framework")["mean"].mean().sort_values()
    print("\n• Overall Performance Ranking (avg execution time):")
    for i, (fw, time) in enumerate(overall_perf.items(), 1):
        print(f"  {i}. {fw}: {time:.4f} seconds")

    # 2. Detailed Performance Comparison
    print("\n\n2. DETAILED PERFORMANCE COMPARISON")
    print("-" * 40)
    print("\nExecution times (seconds) by operation and framework:")
    print(main_comparison.round(4).to_string())

    # 3. Summary Statistics
    print("\n\n3. SUMMARY STATISTICS")
    print("-" * 40)
    print(summary_stats.to_string())

    # 4. Result Consistency Analysis (Hash Comparison)
    print("\n\n4. RESULT CONSISTENCY ANALYSIS")
    print("-" * 40)

    if hash_comparison is not None:
        consistency_summary = hash_comparison["consistency_summary"]
        print(f"Total operations compared: {consistency_summary['total_operations']}")
        print(
            f"Consistent results across frameworks: "
            f"{consistency_summary['consistent_operations']}"
        )
        print(
            f"Inconsistent results across frameworks: "
            f"{consistency_summary['inconsistent_operations']}"
        )

        if consistency_summary["inconsistent_operations"] > 0:
            print("\nINCONSISTENT OPERATIONS (frameworks produce different results):")
            print("-" * 60)

            for analysis in hash_comparison["analysis"]:
                if not analysis["consistent"]:
                    print(f"\nOperation: {analysis['operation']}")
                    print(f"  Frameworks tested: {', '.join(analysis['frameworks'])}")
                    print(f"  Unique result hashes: {analysis['unique_hashes']}")
                    print("  Hash values by framework:")
                    for fw, hash_val in analysis["hashes"].items():
                        print(f"    {fw}: {hash_val[:12]}...")
        else:
            print("\n✓ All operations produce consistent results across frameworks!")

        # Display hash comparison table
        print("\n\nHASH COMPARISON TABLE:")
        print(hash_comparison["hash_pivot"].to_string())

    else:
        print("No hash data available for comparison.")
        print("This may be because:")
        print("  - Results are from older benchmark runs without hash support")
        print("  - All operations failed")
        print("  - Result saving failed during benchmark execution")

    # 5. Speed Comparison (if available)
    if speedup_comparison is not None:
        print("\n\n5. RELATIVE PERFORMANCE (vs pandas no cache)")
        print("-" * 40)
        print("Values < 1.0 indicate faster performance than pandas baseline")
        print(speedup_comparison.round(3).to_string())

        # Best and worst performers
        print("\n• Best performing operations by framework:")
        for col in speedup_comparison.columns:
            if col.endswith("_speedup"):
                best_ops = speedup_comparison[col].nsmallest(3)
                fw_name = col.replace("_speedup", "")
                print(f"\n  {fw_name}:")
                for op, speedup in best_ops.items():
                    if not pd.isna(speedup):
                        print(f"    {op}: {speedup:.3f}x")

    # 6. Cache Effectiveness
    if cache_effectiveness is not None:
        print("\n\n6. CACHE EFFECTIVENESS")
        print("-" * 40)
        print("Cache speedup ratios (no_cache_time / cache_time):")
        print(cache_effectiveness.round(3).to_string())

        # Average cache effectiveness by framework
        print("\n• Average cache effectiveness:")
        for col in cache_effectiveness.columns:
            fw_name = col.replace("_cache_speedup", "")
            avg_speedup = cache_effectiveness[col].mean()
            if not pd.isna(avg_speedup):
                print(f"  {fw_name}: {avg_speedup:.3f}x speedup from caching")

    # 7. Operation Analysis
    print("\n\n7. OPERATION-SPECIFIC ANALYSIS")
    print("-" * 40)

    # Most time-consuming operations
    op_times = (
        raw_data.groupby("operation")["execution_time"]
        .mean()
        .sort_values(ascending=False)
    )
    print("\n• Most time-consuming operations:")
    for i, (op, time) in enumerate(op_times.head(5).items(), 1):
        print(f"  {i}. {op}: {time:.4f} seconds")

    # Fastest operations
    print("\n• Fastest operations:")
    for i, (op, time) in enumerate(op_times.tail(5).items(), 1):
        print(f"  {i}. {op}: {time:.4f} seconds")

    # 8. Framework-Specific Insights
    print("\n\n8. FRAMEWORK-SPECIFIC INSIGHTS")
    print("-" * 40)

    for fw in frameworks:
        fw_data = raw_data[raw_data["framework"] == fw]
        failed_ops = fw_data[~fw_data["success"]]["operation"].unique()

        print(f"\n• {fw.upper()}:")
        print(f"  - Successful operations: {len(fw_data[fw_data['success']])}")
        if len(failed_ops) > 0:
            print(f"  - Failed operations: {len(failed_ops)} ({', '.join(failed_ops)})")
        else:
            print("  - Failed operations: 0")

        # Performance characteristics
        fw_success = fw_data[fw_data["success"]]
        if len(fw_success) > 0:
            _mean_t = fw_success["execution_time"].mean()
            print(f"  - Average execution time: {_mean_t:.4f}s")
            print(
                f"  - Median execution time: "
                f"{fw_success['execution_time'].median():.4f}s"
            )
            print(
                f"  - Performance consistency (std/mean): "
                f"{fw_success['execution_time'].std() / _mean_t:.3f}"
            )

    # 9. Recommendations
    print("\n\n9. RECOMMENDATIONS")
    print("-" * 40)

    if len(frameworks) >= 2:
        fastest_fw = overall_perf.index[0]
        print(
            f"• For overall performance: {fastest_fw} shows the best "
            f"average performance"
        )

        if cache_effectiveness is not None:
            best_cache_fw = None
            best_cache_speedup = 0
            for col in cache_effectiveness.columns:
                fw_name = col.replace("_cache_speedup", "")
                avg_speedup = cache_effectiveness[col].mean()
                if not pd.isna(avg_speedup) and avg_speedup > best_cache_speedup:
                    best_cache_speedup = avg_speedup
                    best_cache_fw = fw_name

            if best_cache_fw:
                print(
                    f"• For cache effectiveness: {best_cache_fw} benefits most "
                    f"from caching ({best_cache_speedup:.2f}x speedup)"
                )

        print("• Consider workload characteristics when choosing a framework:")
        print("  - For complex joins: Check join operation performance")
        print("  - For time series: Check rolling and resampling operations")
        print("  - For large datasets: Consider memory efficiency and lazy evaluation")


def generate_summary_statistics_markdown(summary_stats: pd.DataFrame) -> str:
    """Generate markdown formatted summary statistics"""

    markdown = "# Summary Statistics\n\n"
    markdown += "## Framework Performance Overview\n\n"

    # Convert to markdown table
    markdown += summary_stats.to_markdown() + "\n\n"

    # Add some interpretation
    markdown += "## Key Metrics Explanation\n\n"
    markdown += "- **count**: Number of operations benchmarked\n"
    markdown += "- **mean**: Average execution time (seconds)\n"
    markdown += "- **median**: Median execution time (seconds)\n"
    markdown += "- **std**: Standard deviation of execution times\n"
    markdown += "- **min**: Fastest operation time (seconds)\n"
    markdown += "- **max**: Slowest operation time (seconds)\n"
    markdown += "- **sum**: Total execution time for all operations (seconds)\n\n"

    # Add ranking based on mean performance
    mean_times = summary_stats.groupby("framework")["mean"].mean().sort_values()
    markdown += "## Performance Ranking (by average execution time)\n\n"
    for i, (framework, avg_time) in enumerate(mean_times.items(), 1):
        markdown += f"{i}. **{framework}**: {avg_time:.4f} seconds\n"

    return markdown


def save_results_to_files(tables: Dict[str, Any]) -> None:
    """Save analysis results to files"""

    output_dir = Path("outputs")

    # Save main comparison table
    tables["main_comparison"].to_csv(output_dir / "comparison_table.csv")
    print("\n✓ Saved comparison table to outputs/comparison_table.csv")

    # Save summary statistics
    tables["summary_stats"].to_csv(output_dir / "summary_statistics.csv")
    print("✓ Saved summary statistics to outputs/summary_statistics.csv")

    # Save summary statistics as markdown
    summary_markdown = generate_summary_statistics_markdown(tables["summary_stats"])
    with open(output_dir / "summary_statistics.md", "w") as f:
        f.write(summary_markdown)
    print("✓ Saved summary statistics to outputs/summary_statistics.md")

    # Save speedup comparison if available
    if tables["speedup_comparison"] is not None:
        tables["speedup_comparison"].to_csv(output_dir / "speedup_comparison.csv")
        print("✓ Saved speedup comparison to outputs/speedup_comparison.csv")

    # Save cache effectiveness if available
    if tables["cache_effectiveness"] is not None:
        tables["cache_effectiveness"].to_csv(output_dir / "cache_effectiveness.csv")
        print("✓ Saved cache effectiveness to outputs/cache_effectiveness.csv")

    # Save raw processed data
    tables["raw_data"].to_csv(output_dir / "processed_raw_data.csv", index=False)
    print("✓ Saved processed raw data to outputs/processed_raw_data.csv")

    # Save hash comparison if available
    if tables["hash_comparison"] is not None:
        tables["hash_comparison"]["hash_pivot"].to_csv(
            output_dir / "hash_comparison.csv"
        )
        print("✓ Saved hash comparison to outputs/hash_comparison.csv")


def main() -> None:
    """Main analysis function"""

    print("Starting benchmark analysis...")

    # Load results
    results = load_benchmark_results()

    if not results:
        print("No benchmark results found. Please run the benchmarks first.")
        return

    # Create comparison tables
    tables = create_comparison_tables(results)

    if tables is None:
        return

    # Print analysis report
    print_analysis_report(tables)

    # Save results to files
    save_results_to_files(tables)

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("Check the outputs/ directory for detailed CSV files with all results.")


if __name__ == "__main__":
    main()
