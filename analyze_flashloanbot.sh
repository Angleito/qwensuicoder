#!/bin/bash

FLASHLOANBOT_DIR="/home/angle/projects/Flashloanbot"
OUTPUT_DIR="flashloanbot_analysis"
MODEL="qwen-coder-1.5b"
MAX_FILES=10

echo "Starting code analysis of selected Flashloanbot files..."
echo "Using model: $MODEL"
echo "Output will be saved to: $OUTPUT_DIR directory"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create summary file
SUMMARY_FILE="$OUTPUT_DIR/analysis_summary.md"
echo "# Flashloanbot Code Analysis" > "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Analysis performed on $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "## Files Analyzed" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Find important files for analysis
echo "Finding TypeScript and JavaScript files to analyze..."
find "$FLASHLOANBOT_DIR" -type f \( -name "*.ts" -o -name "*.js" \) \
  -not -path "*/node_modules/*" -not -path "*/dist/*" \
  -not -path "*/Flashloanbot-Mock/*" -not -path "*/bluefin-spot-contract-interface/*" \
  | grep -iE '(trade|arbitrage|flashloan|swap|pool|contract|exchange|market|database)' \
  | head -$MAX_FILES > files_to_analyze.txt

# Check if we found files
if [ ! -s files_to_analyze.txt ]; then
  echo "No relevant files found for analysis. Trying with broader criteria..."
  find "$FLASHLOANBOT_DIR" -type f \( -name "*.ts" -o -name "*.js" \) \
    -not -path "*/node_modules/*" -not -path "*/dist/*" \
    -not -path "*/Flashloanbot-Mock/*" -not -path "*/bluefin-spot-contract-interface/*" \
    | head -$MAX_FILES > files_to_analyze.txt
    
  if [ ! -s files_to_analyze.txt ]; then
    echo "No files found for analysis. Please check the Flashloanbot directory structure."
    exit 1
  fi
fi

# Add list of files to the summary
cat files_to_analyze.txt | while read file; do
  filename=$(basename "$file")
  echo "- [$filename](./${filename}_analysis.md)" >> "$SUMMARY_FILE"
done

echo "" >> "$SUMMARY_FILE"
echo "## Summary of Findings" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Each file has been analyzed individually. Click on the file links above to view detailed analysis." >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Now analyze each file individually
echo "Analyzing files individually..."
cat files_to_analyze.txt | while read file; do
  filename=$(basename "$file")
  output_file="$OUTPUT_DIR/${filename}_analysis.md"
  
  echo "Analyzing: $filename"
  ./analyze_single_file.py "$file" --output "$output_file" --model "$MODEL"
  
  # Extract key findings for the summary
  if [ -f "$output_file" ]; then
    echo "### ${filename}" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    # Extract brief description and issues
    awk '/A brief description/,/suggestions/i' "$output_file" | grep -v "suggestions" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
  fi
done

echo "Analysis complete! Results saved to $OUTPUT_DIR directory"
echo "Summary file: $SUMMARY_FILE"

# Cleanup
rm files_to_analyze.txt 