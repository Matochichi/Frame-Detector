from PIL import Image
import os

def resize_image_with_padding(img_path, output_path, target_size=1000, directory_path=""):
    img = Image.open(img_path)
    width, height = img.size

    # If the image is from the "marked" directory, convert it to binary
    if "marked" in directory_path:
        img = img.convert("L").point(lambda x: 0 if x < 128 else 255, "1")

    ratio = min(target_size / width, target_size / height)

    new_width = int(width * ratio)
    new_height = int(height * ratio)

    img = img.resize((new_width, new_height), Image.LANCZOS)

    new_img = Image.new("RGB", (target_size, target_size), (255, 255, 255))

    y_offset = (target_size - new_height) // 2
    x_offset = (target_size - new_width) // 2
    new_img.paste(img, (x_offset, y_offset))

    new_img.save(output_path)

def process_images_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(directory_path, filename)
            resize_image_with_padding(file_path, file_path, directory_path=directory_path)


process_images_in_directory("output")

print("Resize completed")

folder_path = 'output'
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg'):
        old_filepath = os.path.join(folder_path, filename)
        new_filename = filename.replace('.jpg', '.png')
        new_filepath = os.path.join(folder_path, new_filename)
        
        img = Image.open(old_filepath)
        img.save(new_filepath)
        os.remove(old_filepath)
        print(f'{old_filepath} has been converted to {new_filepath}')