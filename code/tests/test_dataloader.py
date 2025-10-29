# test_dataloader.py
import sys
import os

# This adds the parent directory of 'current_script_dir' to Python's search path.
# If your_script.py is in /app/current_script_dir/, this adds /app to sys.path.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root_dir)

from utils.dataloader import preprocess_data, create_dataloaders, Dataset
import torch # Make sure torch is imported

import json

def load_test_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

config = load_test_config('../configs/example_config.json')


print("1. Testing preprocess_data...")
try:
    unified_df = preprocess_data(config)
    print(f"  Unified DataFrame shape: {unified_df.shape}")
    if not unified_df.empty:
        print(f"  Sample row:\n{unified_df.head(1)}")
        # Check if 'adjusted_landmarks' exists and has content
        if 'adjusted_landmarks' in unified_df.columns:
            print(f"  Sample adjusted_landmarks: {unified_df['adjusted_landmarks'].iloc[0]}")
        else:
            print("  'adjusted_landmarks' column not found in unified_df!")
    else:
        print("  Unified DataFrame is empty!")
except Exception as e:
    print(f"  Error in preprocess_data: {e}")
    raise # Stop if this fails

print("\n2. Testing Dataset...")
# Filter for the specific task and a small number of items
label_name_to_filter = config.get('data_filter_labelName')
if label_name_to_filter and 'labelName' in unified_df.columns:
    task_df = unified_df[unified_df['labelName'] == label_name_to_filter].copy()
    print(f"  Filtering unified_df for labelName '{label_name_to_filter}' resulted in {len(task_df)} items.")
else:
    task_df = unified_df.copy()

if task_df.empty:
    print(f"  No data found for labelName '{label_name_to_filter}' or task_df is empty. Cannot test Dataset effectively.")
else:
    # Use a very small subset for testing
    test_subset_df = task_df.head(2) # Take first 2 samples
    if not test_subset_df.empty:
        try:

            print(f"  Testing Dataset with {len(test_subset_df)} items from task_df.")
            print(f"  Task DataFrame shape: {test_subset_df.shape}")
            print(f"  Base image directory: {config['base_image_dir']}")
            print(f"  Landmarks config: {config['landmarks']}")
            custom_dataset = Dataset(
                dataframe=test_subset_df,
                base_image_dir="/app/data/images",
                landmarks_config=config['landmarks']
            )
            if len(custom_dataset) > 0:
                print(f"  Dataset created with {len(custom_dataset)} items.")
                # Test __getitem__ for the first item
                image, coordinates = custom_dataset[0]
                if image is not None and coordinates is not None:
                    print(f"  Sample 0 - Image shape: {image.shape}, Coordinates: {coordinates}")
                    # Check if coordinates are normalized (between 0 and 1 roughly)
                    if torch.any(coordinates < 0) or torch.any(coordinates > 1.5): # Allow some margin for slight over/under
                        print(f"  Warning: Coordinates might not be normalized: {coordinates}")
                else:
                    print(f"  Sample 0 - __getitem__ returned None. Check image paths and landmark parsing for SOP: {test_subset_df.iloc[0]['SOPInstanceUID']}")

                # Test __getitem__ for the second item if available
                if len(custom_dataset) > 1:
                    image, coordinates = custom_dataset[1]
                    if image is not None and coordinates is not None:
                        print(f"  Sample 1 - Image shape: {image.shape}, Coordinates: {coordinates}")
                    else:
                        print(f"  Sample 1 - __getitem__ returned None. Check image paths and landmark parsing for SOP: {test_subset_df.iloc[1]['SOPInstanceUID']}")
            else:
                print("  Dataset created but is empty. Check filtering of unified_df.")
        except Exception as e:
            print(f"  Error creating or getting items from Dataset: {e}")
    else:
        print("  Could not create a non-empty test_subset_df for Dataset testing.")


print("\n3. Testing Dataloaders (create_dataloaders)...")
try:
    # Temporarily reduce batch size for testing if it's large in config
    original_batch_size = config.get('batch_size', 1)
    config_test_dataloader = config.copy() # Create a copy to modify
    config_test_dataloader['batch_size'] = 1 # Use batch_size 1 for easier inspection

    # Use a small, filtered df for testing create_dataloaders to speed it up
    # Ensure 'Split' column exists for this test_df
    if not task_df.empty and 'Split' in task_df.columns:
        # Create a dummy 'Split' column if it's missing for a very small test
        # Or ensure your test data CSV has this column.
        # For now, assuming task_df has 'Split' after filtering.
        small_task_df_for_loader = task_df.head(5).copy() # Use up to 5 items
        # Ensure there's at least one 'Train' item for the loader to be created
        if 'Train' not in small_task_df_for_loader['Split'].unique() and not small_task_df_for_loader.empty:
            small_task_df_for_loader.loc[small_task_df_for_loader.index[0], 'Split'] = 'Train'


        dataloaders, test_df_output = create_dataloaders(small_task_df_for_loader, config_test_dataloader, return_test_df=True, base_image_dir="/app/data/images/")
        if 'Train' in dataloaders and dataloaders['Train'] is not None and len(dataloaders['Train']) > 0:
            print("  Train Dataloader created.")
            
            print(f"    Train DataLoader length: {len(dataloaders['Train'])}")
            print(f"    Train DataLoader batch size: {dataloaders['Train'].batch_size}")
            print(dataloaders['Train'])

            try:
                # Get an iterator for the DataLoader
                train_dataloader_iter = iter(dataloaders['Train'])
                # Fetch the first batch
                images_batch, landmarks_batch = next(train_dataloader_iter)
                
                print(f"    Sample batch fetched successfully.")
                if images_batch is not None and landmarks_batch is not None:
                    print(f"    Sample batch - Images shape: {images_batch.shape}, Landmarks shape: {landmarks_batch.shape}")
                else:
                    # This case should ideally not happen if the path issue (Problem 1) is fixed
                    print("    Sample batch retrieval resulted in None for images or landmarks.")

            except StopIteration:
                print("    Could not retrieve a batch from DataLoader (it might be empty or exhausted).")
            except Exception as e:
                print(f"    Error while trying to fetch a batch: {e}") # Catch other potential errors

        else:
            print("  Train Dataloader is None or empty. Check data splits and filtering.")

        if 'Test' in dataloaders and dataloaders['Test'] is not None and len(dataloaders['Test']) > 0:
             print("  Test Dataloader created.")
             if test_df_output is not None and not test_df_output.empty:
                 print(f"    Test DataFrame returned with shape: {test_df_output.shape}")
             else:
                 print("    Test DataFrame is None or empty from create_dataloaders.")
        else:
            print("  Test Dataloader is None or empty.")

    else:
        print("  Cannot test create_dataloaders effectively: task_df is empty or 'Split' column missing.")

except Exception as e:
    print(f"  Error in create_dataloaders: {e}")
    # raise # Optionally re-raise

print("\nIncremental Dataloader Testing Complete.")
