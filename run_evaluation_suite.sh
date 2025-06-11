#!/bin/bash
source ~/.bashrc
conda activate seamless

# Default parameters
DATA_DIR="data/fleurs"
BASE_OUTPUT_DIR="evaluation_outputs"
LOG_DIR="evaluation_logs"

# Model configurations
WHISPER_MODEL="large-v2"
WHISPER_BACKEND="faster-whisper"
WHISPER_VAD=true
WHISPER_BUFFER_TRIMMING="segment"
WHISPER_BUFFER_TRIMMING_SEC=15.0

SIMUL_MODEL_PATH="large-v2"
SIMUL_IF_CKPT_PATH="simul_whisper/cif_models/large-v2.pt"
SIMUL_FRAME_THRESHOLD=12
SIMUL_BUFFER_LEN=20
SIMUL_MIN_SEG_LEN=0.0

# Core languages from different families
CORE_LANGUAGES=(
    "eng"  # Western Europe
    "rus"  # Eastern Europe  
    "tur"  # Central Asia
    "swa"  # Sub-Saharan Africa
    "hin"  # South Asia
    "tha"  # South East Asia
    "jpn"  # CJK
)

# Representative languages (all families)
REPRESENTATIVE_LANGUAGES=(
    # Western Europe
    "eng" "spa" "deu" "fra" "ita" "nld"
    # Eastern Europe
    "rus" "pol" "ces" "ukr"
    # Central Asia
    "ara" "tur" "fas" "heb"
    # Sub-Saharan Africa
    "swa" "hau" "yor" "zul" "amh"
    # South Asia
    "hin" "ben" "tam" "pan"
    # South East Asia
    "ind" "tha" "vie" "fil"
    # CJK
    "zho" "jpn" "kor"
)

# Test parameters
WHISPER_CHUNK_SIZES=(1.0)
SIMUL_SEGMENT_LENGTHS=(0.5)

# Create directories
mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$LOG_DIR"

# Function to check if directory is empty
is_dir_empty() {
    local dir="$1"
    if [ -d "$dir" ]; then
        if [ -z "$(ls -A "$dir")" ]; then
            return 0  # Directory is empty
        else
            return 1  # Directory is not empty
        fi
    else
        return 0  # Directory doesn't exist, treat as empty
    fi
}

# Function to run Seamless evaluation
run_seamless_evaluation() {
    local lang=$1
    local output_dir="$BASE_OUTPUT_DIR/seamless/${lang}_fleurs_test"
    local log_file="$LOG_DIR/${lang}_seamless.log"
    
    echo "Starting Seamless evaluation for $lang at $(date)" | tee -a "$log_file"
    
    # Check if output directory exists and is not empty
    if [ -d "$output_dir" ] && ! is_dir_empty "$output_dir"; then
        echo "Skipping $lang - Seamless output directory exists and is not empty" | tee -a "$log_file"
        return 2  # Special return code for skipped
    elif [ -d "$output_dir" ] && is_dir_empty "$output_dir"; then
        echo "Reprocessing $lang - Seamless output directory exists but is empty" | tee -a "$log_file"
        rm -rf "$output_dir"
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    if python evaluate_fleurs_seamless.py \
        --source-lang "$lang" \
        --split "test" \
        --data-dir "$DATA_DIR" \
        --output-dir "$output_dir" 2>&1 | tee -a "$log_file"; then
        
        echo "Successfully completed Seamless evaluation for $lang at $(date)" | tee -a "$log_file"
        return 0
    else
        echo "Failed Seamless evaluation for $lang at $(date)" | tee -a "$log_file"
        return 1
    fi
}

# Function to run Whisper evaluation
run_whisper_evaluation() {
    local lang=$1
    local chunk_size=$2
    local output_dir="$BASE_OUTPUT_DIR/whisper/${lang}_chunk${chunk_size}_fleurs_test"
    local log_file="$LOG_DIR/${lang}_chunk${chunk_size}_whisper.log"
    
    echo "Starting Whisper evaluation for $lang with chunk size $chunk_size at $(date)" | tee -a "$log_file"
    
    # Check if output directory exists and is not empty
    if [ -d "$output_dir" ] && ! is_dir_empty "$output_dir"; then
        echo "Skipping $lang with chunk size $chunk_size - Whisper output directory exists and is not empty" | tee -a "$log_file"
        return 2  # Special return code for skipped
    elif [ -d "$output_dir" ] && is_dir_empty "$output_dir"; then
        echo "Reprocessing $lang with chunk size $chunk_size - Whisper output directory exists but is empty" | tee -a "$log_file"
        rm -rf "$output_dir"
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    if python evaluate_fleurs_whisper.py \
        --source-lang "$lang" \
        --data-dir "$DATA_DIR" \
        --output-dir "$output_dir" \
        --model "$WHISPER_MODEL" \
        --backend "$WHISPER_BACKEND" \
        --min-chunk-size "$chunk_size" \
        --buffer-trimming "$WHISPER_BUFFER_TRIMMING" \
        --buffer-trimming-sec "$WHISPER_BUFFER_TRIMMING_SEC" \
        $([ "$WHISPER_VAD" = false ] && echo "--no-vad") 2>&1 | tee -a "$log_file"; then
        
        echo "Successfully completed Whisper evaluation for $lang with chunk size $chunk_size at $(date)" | tee -a "$log_file"
        return 0
    else
        echo "Failed Whisper evaluation for $lang with chunk size $chunk_size at $(date)" | tee -a "$log_file"
        return 1
    fi
}

# Function to run SimulEval evaluation
run_simul_evaluation() {
    local lang=$1
    local segment_length=$2
    local output_dir="$BASE_OUTPUT_DIR/simul/${lang}_seg${segment_length}_fleurs_test"
    local log_file="$LOG_DIR/${lang}_seg${segment_length}_simul.log"
    
    echo "Starting SimulEval evaluation for $lang with segment length $segment_length at $(date)" | tee -a "$log_file"
    
    # Check if output directory exists and is not empty
    if [ -d "$output_dir" ] && ! is_dir_empty "$output_dir"; then
        echo "Skipping $lang with segment length $segment_length - SimulEval output directory exists and is not empty" | tee -a "$log_file"
        return 2  # Special return code for skipped
    elif [ -d "$output_dir" ] && is_dir_empty "$output_dir"; then
        echo "Reprocessing $lang with segment length $segment_length - SimulEval output directory exists but is empty" | tee -a "$log_file"
        rm -rf "$output_dir"
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    if python evaluate_fleurs_simul.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$output_dir" \
        --source-lang "$lang" \
        --model-path "$SIMUL_MODEL_PATH" \
        --if-ckpt-path "$SIMUL_IF_CKPT_PATH" \
        --segment-length "$segment_length" \
        --frame-threshold "$SIMUL_FRAME_THRESHOLD" \
        --buffer-len "$SIMUL_BUFFER_LEN" \
        --min-seg-len "$SIMUL_MIN_SEG_LEN" \
        --language "$lang" 2>&1 | tee -a "$log_file"; then
        
        echo "Successfully completed SimulEval evaluation for $lang with segment length $segment_length at $(date)" | tee -a "$log_file"
        return 0
    else
        echo "Failed SimulEval evaluation for $lang with segment length $segment_length at $(date)" | tee -a "$log_file"
        return 1
    fi
}

# Parse command line arguments
EVALUATION_TYPE="all"  # Default to running all evaluations
LANGUAGE_SET="core"    # Default to core languages

while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            EVALUATION_TYPE="$2"
            shift 2
            ;;
        --languages)
            LANGUAGE_SET="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            BASE_OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --type TYPE          Evaluation type: seamless, whisper, simul, or all (default: all)"
            echo "  --languages SET      Language set: core or representative (default: core)"
            echo "  --data-dir DIR       Data directory (default: data/fleurs)"
            echo "  --output-dir DIR     Base output directory (default: evaluation_outputs)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Select language set
if [ "$LANGUAGE_SET" = "representative" ]; then
    LANGUAGES=("${REPRESENTATIVE_LANGUAGES[@]}")
else
    LANGUAGES=("${CORE_LANGUAGES[@]}")
fi

# Create summary files
SUMMARY_FILE="$LOG_DIR/evaluation_suite_summary.txt"
echo "Starting evaluation suite at $(date)" > "$SUMMARY_FILE"
echo "Evaluation type: $EVALUATION_TYPE" >> "$SUMMARY_FILE"
echo "Language set: $LANGUAGE_SET (${#LANGUAGES[@]} languages)" >> "$SUMMARY_FILE"
echo "Languages: ${LANGUAGES[*]}" >> "$SUMMARY_FILE"
echo "----------------------------------------" >> "$SUMMARY_FILE"

# Initialize counters
seamless_successful=0
seamless_failed=0
seamless_skipped=0

whisper_successful=0
whisper_failed=0
whisper_skipped=0

simul_successful=0
simul_failed=0
simul_skipped=0

# Run evaluations based on type
for lang in "${LANGUAGES[@]}"; do
    echo "Processing language: $lang"
    echo "Processing language: $lang" >> "$SUMMARY_FILE"
    
    # Run Seamless evaluation
    if [ "$EVALUATION_TYPE" = "all" ] || [ "$EVALUATION_TYPE" = "seamless" ]; then
        echo "Running Seamless evaluation for $lang..."
        case $(run_seamless_evaluation "$lang") in
            0)  # Success
                echo "  SEAMLESS SUCCESS: $lang" >> "$SUMMARY_FILE"
                ((seamless_successful++))
                ;;
            1)  # Failure
                echo "  SEAMLESS FAILED: $lang" >> "$SUMMARY_FILE"
                ((seamless_failed++))
                ;;
            2)  # Skipped
                echo "  SEAMLESS SKIPPED: $lang" >> "$SUMMARY_FILE"
                ((seamless_skipped++))
                ;;
        esac
    fi
    
    # Run Whisper evaluation
    if [ "$EVALUATION_TYPE" = "all" ] || [ "$EVALUATION_TYPE" = "whisper" ]; then
        for chunk_size in "${WHISPER_CHUNK_SIZES[@]}"; do
            echo "Running Whisper evaluation for $lang with chunk size $chunk_size..."
            case $(run_whisper_evaluation "$lang" "$chunk_size") in
                0)  # Success
                    echo "  WHISPER SUCCESS: $lang (chunk: $chunk_size)" >> "$SUMMARY_FILE"
                    ((whisper_successful++))
                    ;;
                1)  # Failure
                    echo "  WHISPER FAILED: $lang (chunk: $chunk_size)" >> "$SUMMARY_FILE"
                    ((whisper_failed++))
                    ;;
                2)  # Skipped
                    echo "  WHISPER SKIPPED: $lang (chunk: $chunk_size)" >> "$SUMMARY_FILE"
                    ((whisper_skipped++))
                    ;;
            esac
        done
    fi
    
    # Run SimulEval evaluation
    if [ "$EVALUATION_TYPE" = "all" ] || [ "$EVALUATION_TYPE" = "simul" ]; then
        for segment_length in "${SIMUL_SEGMENT_LENGTHS[@]}"; do
            echo "Running SimulEval evaluation for $lang with segment length $segment_length..."
            case $(run_simul_evaluation "$lang" "$segment_length") in
                0)  # Success
                    echo "  SIMUL SUCCESS: $lang (segment: $segment_length)" >> "$SUMMARY_FILE"
                    ((simul_successful++))
                    ;;
                1)  # Failure
                    echo "  SIMUL FAILED: $lang (segment: $segment_length)" >> "$SUMMARY_FILE"
                    ((simul_failed++))
                    ;;
                2)  # Skipped
                    echo "  SIMUL SKIPPED: $lang (segment: $segment_length)" >> "$SUMMARY_FILE"
                    ((simul_skipped++))
                    ;;
            esac
        done
    fi
    
    echo "----------------------------------------" >> "$SUMMARY_FILE"
done

# Write final summary
echo "Evaluation suite completed at $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "FINAL RESULTS:" >> "$SUMMARY_FILE"

if [ "$EVALUATION_TYPE" = "all" ] || [ "$EVALUATION_TYPE" = "seamless" ]; then
    echo "Seamless Evaluation:" >> "$SUMMARY_FILE"
    echo "  - Successful: $seamless_successful" >> "$SUMMARY_FILE"
    echo "  - Failed: $seamless_failed" >> "$SUMMARY_FILE"
    echo "  - Skipped: $seamless_skipped" >> "$SUMMARY_FILE"
    echo "  - Total: $((seamless_successful + seamless_failed + seamless_skipped))" >> "$SUMMARY_FILE"
fi

if [ "$EVALUATION_TYPE" = "all" ] || [ "$EVALUATION_TYPE" = "whisper" ]; then
    echo "Whisper Evaluation:" >> "$SUMMARY_FILE"
    echo "  - Successful: $whisper_successful" >> "$SUMMARY_FILE"
    echo "  - Failed: $whisper_failed" >> "$SUMMARY_FILE"
    echo "  - Skipped: $whisper_skipped" >> "$SUMMARY_FILE"
    echo "  - Total: $((whisper_successful + whisper_failed + whisper_skipped))" >> "$SUMMARY_FILE"
fi

if [ "$EVALUATION_TYPE" = "all" ] || [ "$EVALUATION_TYPE" = "simul" ]; then
    echo "SimulEval Evaluation:" >> "$SUMMARY_FILE"
    echo "  - Successful: $simul_successful" >> "$SUMMARY_FILE"
    echo "  - Failed: $simul_failed" >> "$SUMMARY_FILE"
    echo "  - Skipped: $simul_skipped" >> "$SUMMARY_FILE"
    echo "  - Total: $((simul_successful + simul_failed + simul_skipped))" >> "$SUMMARY_FILE"
fi

echo ""
echo "All evaluations completed. Check $SUMMARY_FILE for detailed results."
echo "Individual logs are available in the $LOG_DIR directory."
echo "Output files are saved in the $BASE_OUTPUT_DIR directory." 