import os

def get_file_list(directory):
    return set([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

raw_folder = 'resource\\raw'
marked_folder = 'resource\\mask'

raw_files = get_file_list(raw_folder)
marked_files = get_file_list(marked_folder)

only_in_raw = raw_files - marked_files
for file in only_in_raw:
    print(f"'{file}'")

only_in_marked = marked_files - raw_files
for file in only_in_marked:
    print(f"'{file}'")

import os

def get_file_list(directory):
    return set([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

def delete_unmatched_jpg_files(folder, files):
    for file in files:
        if file.lower().endswith('.jpg'):
            os.remove(os.path.join(folder, file))

raw_folder = 'resource\\raw'
marked_folder = 'resource\\mask'

raw_files = get_file_list(raw_folder)
marked_files = get_file_list(marked_folder)

only_in_raw = raw_files - marked_files
delete_unmatched_jpg_files(raw_folder, only_in_raw)

raw_dir = "resource/raw"
marked_dir = "resource/mask"

# Identify images that are not of the size 1000x1000
incorrect_size_images = []