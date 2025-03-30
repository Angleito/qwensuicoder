#!/bin/bash

# Code analyzer script using simple_model_demo.py
# Usage: ./code_analyzer.sh <file_path> [analysis_type]

set -e

# Default values
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
MAX_TOKENS=500
ANALYSIS_TYPE=${2:-"basic"}  # Default to basic analysis if not specified

# Check if file exists
if [ -z "$1" ]; then
  echo "Error: Please provide a file path to analyze"
  echo "Usage: ./code_analyzer.sh <file_path> [analysis_type]"
  echo "Analysis types: basic, detailed, bugs, refactor"
  exit 1
fi

FILE_PATH="$1"
if [ ! -f "$FILE_PATH" ]; then
  echo "Error: File '$FILE_PATH' not found"
  exit 1
fi

# Create output directory
mkdir -p code_analysis

# Get file extension and determine language
FILE_EXT="${FILE_PATH##*.}"
LANGUAGE=""
case "$FILE_EXT" in
  py) LANGUAGE="Python" ;;
  js) LANGUAGE="JavaScript" ;;
  ts) LANGUAGE="TypeScript" ;;
  jsx) LANGUAGE="React JSX" ;;
  tsx) LANGUAGE="React TSX" ;;
  cpp|cc) LANGUAGE="C++" ;;
  c) LANGUAGE="C" ;;
  go) LANGUAGE="Go" ;;
  rs) LANGUAGE="Rust" ;;
  java) LANGUAGE="Java" ;;
  rb) LANGUAGE="Ruby" ;;
  php) LANGUAGE="PHP" ;;
  sh) LANGUAGE="Bash" ;;
  *) LANGUAGE="code" ;;
esac

# Get file content with line limit
FILE_CONTENT=$(head -n 300 "$FILE_PATH")

# Create appropriate prompt based on analysis type
PROMPT=""
case "$ANALYSIS_TYPE" in
  basic)
    PROMPT="Analyze this $LANGUAGE code and explain what it does: \`\`\`$FILE_EXT
$FILE_CONTENT
\`\`\`"
    ;;
  detailed)
    PROMPT="Provide a detailed analysis of this $LANGUAGE code including structure, patterns, and key functions: \`\`\`$FILE_EXT
$FILE_CONTENT
\`\`\`"
    ;;
  bugs)
    PROMPT="Identify potential bugs, vulnerabilities or issues in this $LANGUAGE code: \`\`\`$FILE_EXT
$FILE_CONTENT
\`\`\`"
    ;;
  refactor)
    PROMPT="Suggest how this $LANGUAGE code could be refactored for better readability and maintainability: \`\`\`$FILE_EXT
$FILE_CONTENT
\`\`\`"
    ;;
  *)
    echo "Unknown analysis type: $ANALYSIS_TYPE, using 'basic'"
    PROMPT="Analyze this $LANGUAGE code and explain what it does: \`\`\`$FILE_EXT
$FILE_CONTENT
\`\`\`"
    ;;
esac

# Output file name
OUTPUT_FILE="code_analysis/$(basename "$FILE_PATH")_${ANALYSIS_TYPE}_analysis.txt"

echo "Analyzing $FILE_PATH..."
echo "Language: $LANGUAGE"
echo "Analysis type: $ANALYSIS_TYPE"
echo "Output will be saved to: $OUTPUT_FILE"

# Run the model
python simple_model_demo.py --model_name "$MODEL" --prompt "$PROMPT" --max_tokens "$MAX_TOKENS" | tee "$OUTPUT_FILE"

echo "Analysis complete! Results saved to $OUTPUT_FILE" 