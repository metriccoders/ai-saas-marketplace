import os
import shutil

# Define the base directory where the folders will be created
base_directory = '.'

# Define the paths of the two files to be copied
file1 = 'app.py'
file2 = 'ml_model.py'

# Ensure the base directory exists
if not os.path.exists(base_directory):
    os.makedirs(base_directory)

# Loop to create 500 folders and copy the files
for i in range(1, 501):
    # Create folder name
    folder_name = f'project{i}'

    # Create the full path for the new folder
    folder_path = os.path.join(base_directory, folder_name)

    # Create the folder
    os.makedirs(folder_path, exist_ok=True)

    # Copy the files into the new folder
    shutil.copy(file1, folder_path)
    shutil.copy(file2, folder_path)

print("Folders created and files copied successfully.")
