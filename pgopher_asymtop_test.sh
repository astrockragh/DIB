#!/bin/bash
# Number of parallel processes to use
NUM_PROCS=64

# Define parameter ranges
TEMP_START=20
TEMP_END=22
TEMP_STEP=2

# Base constants for A and B
AB_START=0.002
AB_END=0.0025
AB_STEP=0.0005

# Base constants for C
C_START=0.03
C_END=0.033
C_STEP=0.003

# Fractions for A and B
FRACTION_AB_START=1.000
FRACTION_AB_END=1.001
FRACTION_AB_STEP=0.001

# Fractions for C
FRACTION_C_START=1.000
FRACTION_C_END=1.001
FRACTION_C_STEP=0.001

EPSILON=0.1

BASE_DIR=~/../../scratch/gpfs/cj1223
OUTPUT_DIR="$BASE_DIR/pgo_outputs"

process_combination() {
    local temp="$1"
    local ab_base="$2"
    local c_base="$3"
    local fraction_AB="$4"
    local fraction_C="$5"

    local A_excited=$(echo "$ab_base * $fraction_AB" | bc -l)
    local B_excited=$(echo "$ab_base * $fraction_AB" | bc -l)
    local C_excited=$(echo "$c_base * $fraction_C" | bc -l)

    local temp_formatted=$(printf "%.1f" "$temp")
    local ab_formatted=$(printf "%.6f" "$ab_base")
    local c_formatted=$(printf "%.6f" "$c_base")
    local fraction_AB_formatted=$(printf "%.4f" "$fraction_AB")
    local fraction_C_formatted=$(printf "%.4f" "$fraction_C")

    local temp_file="$BASE_DIR/temp_T${temp_formatted}_AB${ab_formatted}_C${c_formatted}_FAB${fraction_AB_formatted}_FC${fraction_C_formatted}.pgo"
    local output_file="$OUTPUT_DIR/spec_T${temp_formatted}_AB${ab_formatted}_C${c_formatted}_FAB${fraction_AB_formatted}_FC${fraction_C_formatted}.txt"

    awk -v temp="$temp" \
        -v A_ground="$ab_base" -v B_ground="$ab_base" -v C_ground="$c_base" \
        -v A_excited="$A_excited" -v B_excited="$B_excited" -v C_excited="$C_excited" '

    BEGIN { inside_ground = 0; inside_excited = 0; }

    /<Parameter Name="Temperature" Value="/ {
        sub(/Value="[0-9.eE+-]+"/, "Value=\"" temp "\"")
    }

    /<AsymmetricManifold Name="Ground"/ { inside_ground = 1 }
    /<AsymmetricManifold Name="Excited"/ { inside_excited = 1 }
    /<\/AsymmetricManifold>/ { inside_ground = 0; inside_excited = 0 }

    inside_ground && /<Parameter Name="A" Value="/ { sub(/Value="[0-9.eE+-]+"/, "Value=\"" A_ground "\"") }
    inside_ground && /<Parameter Name="B" Value="/ { sub(/Value="[0-9.eE+-]+"/, "Value=\"" B_ground "\"") }
    inside_ground && /<Parameter Name="C" Value="/ { sub(/Value="[0-9.eE+-]+"/, "Value=\"" C_ground "\"") }

    inside_excited && /<Parameter Name="A" Value="/ { sub(/Value="[0-9.eE+-]+"/, "Value=\"" A_excited "\"") }
    inside_excited && /<Parameter Name="B" Value="/ { sub(/Value="[0-9.eE+-]+"/, "Value=\"" B_excited "\"") }
    inside_excited && /<Parameter Name="C" Value="/ { sub(/Value="[0-9.eE+-]+"/, "Value=\"" C_excited "\"") }

    { print }
    ' asym_top_0.pgo > "$temp_file"

    ./pgo "$temp_file" -o "$output_file" > /dev/null 2>&1
    rm -f "$temp_file"
}

# Clean just in case
rm -rf "$OUTPUT_DIR"

# Export the function so it's available to parallel
export -f process_combination
export BASE_DIR
export OUTPUT_DIR

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run batches
echo "Running batch for asymmetric top with independent bases..."
for temp in $(seq $TEMP_START $TEMP_STEP $TEMP_END); do
    for ab_base in $(seq $AB_START $AB_STEP $AB_END); do
        for c_base in $(seq $C_START $C_STEP $C_END); do
            for fraction_AB in $(seq $FRACTION_AB_START $FRACTION_AB_STEP $FRACTION_AB_END); do
                for fraction_C in $(seq $FRACTION_C_START $FRACTION_C_STEP $FRACTION_C_END); do
                    printf "%s\t%s\t%s\t%s\t%s\n" "$temp" "$ab_base" "$c_base" "$fraction_AB" "$fraction_C"
                done
            done
        done
    done
done | parallel --will-cite --bar -j $NUM_PROCS --colsep '\t' process_combination {1} {2} {3} {4} {5}

# Process first batch with Julia
echo -e "\nProcessing first batch with Julia..."
julia ingest_pgo_asymtop.jl "$OUTPUT_DIR" "${OUTPUT_DIR}_original_test"

# Clean up pgo_outputs directory
echo -e "\nCleaning up pgo_outputs directory..."
rm -rf "$OUTPUT_DIR"

# Create output directory again
mkdir -p "$OUTPUT_DIR"

# Run batches with temperature offset
echo "Running delta temperature run..."
for temp in $(seq $TEMP_START $TEMP_STEP $TEMP_END); do
    adjusted_temp=$(echo "$temp + $EPSILON" | bc)
    for ab_base in $(seq $AB_START $AB_STEP $AB_END); do
        for c_base in $(seq $C_START $C_STEP $C_END); do
            for fraction_AB in $(seq $FRACTION_AB_START $FRACTION_AB_STEP $FRACTION_AB_END); do
                for fraction_C in $(seq $FRACTION_C_START $FRACTION_C_STEP $FRACTION_C_END); do
                    printf "%s\t%s\t%s\t%s\t%s\n" "$adjusted_temp" "$ab_base" "$c_base" "$fraction_AB" "$fraction_C"
                done
            done
        done
    done
done | parallel --will-cite --bar -j $NUM_PROCS --colsep '\t' process_combination {1} {2} {3} {4} {5}

# Process second batch with Julia
echo -e "\nProcessing second batch with Julia..."
julia ingest_pgo_asymtop.jl "$OUTPUT_DIR" "${OUTPUT_DIR}_epsilon_test"

# Clean up pgo_outputs directory
echo -e "\nCleaning up pgo_outputs directory..."
rm -rf "$OUTPUT_DIR"

echo -e "\nScan complete! Results saved in scratch directory."
