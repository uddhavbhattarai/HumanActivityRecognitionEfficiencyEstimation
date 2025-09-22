import os
import papermill as pm

data_root = "/media/uddhav/My_Passport/00_STAVROS_LAB/Harvest_2024"
# Path to the single notebook you want to run
notebook_to_run = (
    "/home/uddhav/WorkerActivityRecognitionEfficiencyEstimation/Step3_1_classify_seasonlong_data_pick_nopick.ipynb"
)

FIELD_NAME = "SantaMaria"  # SantaMaria or Salinas

data_folder = os.path.join(data_root, "datasets", FIELD_NAME)

for folder in sorted(os.listdir(data_folder)):
    if folder.startswith("yield") or folder.startswith("Picker"):
        continue
    folder_path = os.path.join(data_folder, folder)

    if os.path.isdir(folder_path):
        try:
            print(f"Running notebook for folder: {folder_path}")
            pm.execute_notebook(
                notebook_to_run,
                notebook_to_run,
                parameters={
                    "data_root": data_root,
                    "date_time": folder,
                    "field": FIELD_NAME,
                },
            )
        except Exception as e:
            print(f"Error occurred while running notebook for {folder_path}: {e}")
