import os
from PIL import Image

folder_path = 'resource/raw'
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg'):
        old_filepath = os.path.join(folder_path, filename)
        new_filename = filename.replace('.jpg', '.png')
        new_filepath = os.path.join(folder_path, new_filename)
        
        img = Image.open(old_filepath)
        img.save(new_filepath)
        os.remove(old_filepath)
        print(f'{old_filepath} has been converted to {new_filepath}')
