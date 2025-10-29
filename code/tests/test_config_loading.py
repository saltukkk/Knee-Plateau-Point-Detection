# test_config_loading.py
import json

def load_test_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

try:
    train_config = load_test_config('../configs/example_config.json')
    print("Train Config Loaded Successfully:")
    print(f"  Model Type: {train_config.get('model_type')}")
    print(f"  Landmarks: {train_config.get('landmarks')}")
    print(f"  Data Filter: {train_config.get('data_filter_labelName')}")

    eval_config = load_test_config('../configs/example_eval_config.json')
    print("\nEval Config Loaded Successfully:")
    print(f"  Model Type: {eval_config.get('model_type')}")
    print(f"  Landmarks: {eval_config.get('landmarks')}")

except Exception as e:
    print(f"Error loading configuration: {e}")
