import os
import json
import subprocess
import shutil
import datetime

# ==============================================================================
# 1. DEFINE YOUR EXPERIMENTS HERE
# ==============================================================================
# Each dictionary in this list represents one experiment.
# The 'name' will be used for the output folder.
# The 'params' dictionary contains key-value pairs that will overwrite the
# values in your base 'example_config.json' and 'example_eval_config.json'.
#
# You can change model types, learning rates, loss function parameters, etc.
# Add as many experiment dictionaries as you need.
# ==============================================================================

experiments = [
    # --- Baseline RAUNet ---
    {
        "name": "CRAUNet_baseline_Notch",
        "params": {
            "model_type": "CRAUNet",
            "learning_rate": 1e-4,
            "num_epochs": 50,
            "loss_wing_w": 4.0,
            "loss_wing_epsilon": 1.5
        }
    },
    {
        "name": "CRAUNet_baseline_epoch_100",
        "params": {
            "model_type": "CRAUNet",
            "learning_rate": 1e-4,
            "num_epochs": 100,
            "loss_wing_w": 4.0,
            "loss_wing_epsilon": 1.5
        }
    },
    {
        "name": "CRAUNet_baseline_epoch_300",
        "params": {
            "model_type": "CRAUNet",
            "learning_rate": 1e-4,
            "num_epochs": 300,
            "loss_wing_w": 4.0,
            "loss_wing_epsilon": 1.5
        }
    }
    
]

# ==============================================================================
# 2. SCRIPT CONFIGURATION
# ==============================================================================
# Base configuration files
BASE_TRAIN_CONFIG_PATH = './configs/example_config.json'
BASE_EVAL_CONFIG_PATH = './configs/example_eval_config.json'

# Main directory to store all experiment results
RESULTS_BASE_DIR = 'experiment_results'

# Names of the output files and folders your scripts generate
OUTPUT_FILES = [
    "training_metrics.csv",
    "evaluation_results_generic.csv",
    "evaluation_summary_stats_generic.csv"
]
OUTPUT_DIRS = [
    "predictions_visualization"
]

# ==============================================================================
# 3. EXPERIMENT RUNNER LOGIC
# (You shouldn't need to change anything below this line)
# ==============================================================================

def run_command(command):
    """Executes a shell command and raises an exception if it fails."""
    print(f"\n[RUNNING] {' '.join(command)}")
    subprocess.run(command, check=True)

def main():
    """Main function to run all experiments."""
    print("Starting experiment runner...")
    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

    # Load base configurations once
    try:
        with open(BASE_TRAIN_CONFIG_PATH, 'r') as f:
            base_train_config = json.load(f)
        with open(BASE_EVAL_CONFIG_PATH, 'r') as f:
            base_eval_config = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Base configuration file not found. {e}")
        print("Please ensure this script is in your project's root directory.")
        return

    total_experiments = len(experiments)
    for i, exp in enumerate(experiments):
        experiment_name = f"{exp['name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print("\n" + "="*80)
        print(f"Running Experiment {i+1}/{total_experiments}: {experiment_name}")
        print("="*80)

        # --- Create a directory for the experiment results ---
        experiment_dir = os.path.join(RESULTS_BASE_DIR, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)

        # --- Create temporary config files for this run ---
        temp_train_config = base_train_config.copy()
        temp_eval_config = base_eval_config.copy()

        # Update parameters from the experiment definition
        temp_train_config.update(exp['params'])
        # Ensure model_type is consistent in eval config
        if 'model_type' in exp['params']:
            temp_eval_config['model_type'] = exp['params']['model_type']

        # Overwrite data_filter_labelName to an empty string
        # This is to ensure no specific label filtering is applied
        temp_train_config['data_filter_labelName'] = ''
        temp_eval_config['data_filter_labelName'] = ''

        # Set a unique, organized path for the trained model
        model_filename = f"{exp['params'].get('model_type', 'model')}.pth"
        model_save_path = os.path.join(experiment_dir, model_filename)
        temp_train_config['best_model_path'] = model_save_path
        temp_eval_config['best_model_path'] = model_save_path

        # Write the temporary configs
        temp_train_config_path = os.path.join('configs', f'temp_train_{experiment_name}.json')
        temp_eval_config_path = os.path.join('configs', f'temp_eval_{experiment_name}.json')

        # Refresh the split
        input_file = os.path.join('./data', 'transformation_details.csv')

        output_file1 = os.path.join('../', 'unified_data_with_splits.csv')
        output_file2 = os.path.join('../annotations', 'unified_data_with_splits.csv')
        output_file3 = os.path.join('./', 'unified_data_with_splits.csv')
        output_file4 = os.path.join('./annotations', 'unified_data_with_splits.csv')
        run_command(['python', 'csv_tool.py', input_file, output_file1])
        run_command(['cp', output_file1, output_file2])
        run_command(['cp', output_file1, output_file3])
        run_command(['cp', output_file1, output_file4])

        print("Split is refreshed in unified_data_with_splits.csv file")

        try:
            with open(temp_train_config_path, 'w') as f:
                json.dump(temp_train_config, f, indent=4)
            with open(temp_eval_config_path, 'w') as f:
                json.dump(temp_eval_config, f, indent=4)

            # --- Run the scripts sequentially ---
            print("Step 1: Training model...")
            run_command(['python', 'main.py', '--config', temp_train_config_path])

            print("\nStep 2: Evaluating model...")
            run_command(['python', 'evaluation.py', '--config', temp_eval_config_path])

            print("\nStep 3: Visualizing predictions...")
            run_command(['python', 'visualize_test_predictions.py', '--config', temp_eval_config_path])

            print("\n--- Moving results ---")
            # --- Move output files and directories to the experiment folder ---
            for file in OUTPUT_FILES:
                if os.path.exists(file):
                    shutil.move(file, os.path.join(experiment_dir, file))
                    print(f"Moved: {file}")
            for d in OUTPUT_DIRS:
                if os.path.exists(d):
                    shutil.move(d, os.path.join(experiment_dir, d))
                    print(f"Moved: {d}")

            # The model is already saved in the correct directory, so no move needed.
            print(f"Model saved at: {model_save_path}")
            print(f"\nSUCCESS: Experiment '{experiment_name}' completed successfully.")

        except subprocess.CalledProcessError as e:
            print(f"\nERROR: A script failed to execute for experiment '{experiment_name}'.")
            print(f"Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}.")
            print("Skipping to the next experiment.")
        except Exception as e:
            print(f"\nAn unexpected error occurred during experiment '{experiment_name}': {e}")
            print("Skipping to the next experiment.")

        finally:
            # --- Clean up temporary config files ---
            if os.path.exists(temp_train_config_path):
                os.remove(temp_train_config_path)
            if os.path.exists(temp_eval_config_path):
                os.remove(temp_eval_config_path)
            print("Cleaned up temporary config files.")


    print("\n" + "="*80)
    print("All experiments have finished.")
    print(f"All results are stored in the '{RESULTS_BASE_DIR}' directory.")
    print("="*80)

if __name__ == '__main__':
    main()

