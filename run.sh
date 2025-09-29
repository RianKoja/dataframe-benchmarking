
#!/bin/bash
set -euo pipefail

# Initialize timing variables
declare -A execution_times
declare -a execution_order

# Function to run a command and time it
run_timed() {
    local name="$1"
    shift
    local cmd="$*"
    
    echo "Running: $name"
    local start_time=$(date +%s.%N)
    eval "$cmd"
    local exit_code=$?
    local end_time=$(date +%s.%N)
    
    local duration=$(echo "$end_time - $start_time" | bc -l)
    execution_times["$name"]=$(printf "%.2f" "$duration")
    execution_order+=("$name")
    
    if [ $exit_code -ne 0 ]; then
        echo "Error: $name failed (exit code: $exit_code)"
        echo "Benchmark suite cannot continue with incomplete data. Exiting."
        exit 1
    fi
    
    return $exit_code
}

# Function to run a Python command with pyinstrument profiling
run_python_profiled() {
    local name="$1"
    local python_script="$2"
    local args="${3:-}"
    
    # Create pyinstrument output directory
    mkdir -p outputs/pyinstrument
    
    # Create a safe filename for the profiling output
    local safe_name=$(echo "$name" | sed 's/[^a-zA-Z0-9._-]/_/g')
    local profile_output="outputs/pyinstrument/${safe_name}_profile.html"
    
    # Run with pyinstrument profiling
    local full_cmd="pyinstrument -r html -o \"$profile_output\" $python_script $args"
    run_timed "$name" "$full_cmd"
}

# Set up environment
echo "Setting up virtual environment..."
uv venv 
source .venv/bin/activate
uv pip compile requirements.in | uv pip sync -

# Log environment details
{
  echo "# Python $(python --version 2>&1)"
  uv pip freeze
} > artifacts/requirements_$(date +%Y%m%d_%H%M%S).txt

# Ensure code quality
echo "Running code quality checks..."
ruff format .
ruff check . --select E,F,I --fix
uvx ty check .

# Clean up previous results:
rm -rf artifacts/* data/* outputs/* ___pycache__

# Check if input data exists, else create it
if [ ! -d "data" ] || [ -z "$(ls -A data 2>/dev/null)" ]; then
    echo "Data directory is empty or doesn't exist. Creating datasets..."
    run_python_profiled "Data Preparation" "01_prep_data.py" ""
else
    echo "Data directory exists and contains files. Skipping data preparation."
fi

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Clear any existing output files to avoid confusion
echo "Clearing previous benchmark results..."
rm -f outputs/*_results.parquet
rm -rf outputs/results

echo ""
echo "Starting benchmark runs..."
echo "=========================="

# Run all cases with and without cache
rm -rf __pycache__
run_python_profiled "Pandas (no cache)" "02_benchmark.py" "pandas"
run_python_profiled "Pandas (with cache)" "02_benchmark.py" "pandas --cache"
rm -rf __pycache__
run_python_profiled "Fireducks (no cache)" "02_benchmark.py" "fireducks"
run_python_profiled "Fireducks (with cache)" "02_benchmark.py" "fireducks --cache"
rm -rf __pycache__
run_python_profiled "Polars (no cache)" "03_polars.py" ""
run_python_profiled "Polars (with cache)" "03_polars.py" "--cache"

echo ""
echo "All benchmarks completed. Running result comparison..."
echo "====================================================="

# Compare parquet results for accuracy verification
run_python_profiled "Result Comparison" "04_compare_parquets.py" ""

echo ""
echo "Running performance analysis..."
echo "==============================="

# Run analyzer
run_python_profiled "Analysis" "05_comparison.py" ""

echo ""
echo "==============================================="
echo "BENCHMARK SUITE EXECUTION TIME SUMMARY"
echo "==============================================="

# Calculate total time
total_time=0

# Display execution times in a formatted table
printf "%-25s %10s\n" "Component" "Time (s)"
printf "%-25s %10s\n" "-------------------------" "----------"

# Iterate through components in execution order
for key in "${execution_order[@]}"; do
    printf "%-25s %10s\n" "$key" "${execution_times[$key]}"
    total_time=$(echo "$total_time + ${execution_times[$key]}" | bc -l)
done

printf "%-25s %10s\n" "-------------------------" "----------"
printf "%-25s %10.2f\n" "TOTAL TIME" "$total_time"

echo ""
echo "Benchmark suite completed successfully!"
echo "Check the outputs/ directory for detailed results."
echo "Key files:"
echo "  - comparison_table.csv: Main performance comparison"
echo "  - summary_statistics.csv: Statistical summary"
echo "  - speedup_comparison.csv: Relative performance metrics"
echo "  - cache_effectiveness.csv: Cache performance analysis"