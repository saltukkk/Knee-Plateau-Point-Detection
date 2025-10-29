#!/bin/bash

set -e  # Stop the script if any command fails

run_test() {
    script=$1

    echo -e "\n--- Running tests in $folder ---"

    chmod +x "$script"
    ./"$script"

    cd code
    python run_experiments3.py
    cd ../

    echo "--- Finished tests in $folder ---"
}

# Run all tests
run_test "update_files.sh"

echo -e "\n✅ All tests completed successfully."

