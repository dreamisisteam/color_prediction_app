import pandas as pd
import numpy as np
import cv2
import os
import json
from tqdm import tqdm

image_size = 240
rect_size = int(image_size * 0.75)

output_images_folder = "color_dataset/images"
output_labels_folder = "color_dataset/labels"

def generate_image_with_background_noise(rgb, noise_level=0.05):

    image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

    noise = np.random.uniform(-noise_level, noise_level, image.shape).astype(np.float32)
    noisy_background = np.clip(image / 255.0 + noise, 0, 1) * 255
    noisy_background = noisy_background.astype(np.uint8)

    x1 = (image_size - rect_size) // 2
    y1 = (image_size - rect_size) // 2
    x2 = x1 + rect_size
    y2 = y1 + rect_size

    final_image = noisy_background.copy()

    bgr = rgb[::-1]
    cv2.rectangle(final_image, (x1, y1), (x2, y2), color=bgr, thickness=-1)

    return final_image

def generate_dataset(excel_path, num_images_per_class=100):

    data = pd.read_excel(excel_path)
    
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        class_id = row['class']
        class_name = row['Name']
        rgb = tuple(map(int, row['Rgb'][1:-1].split(',')))
        
        for i in range(num_images_per_class):
            image = generate_image_with_background_noise(rgb)
            
            image_filename = f"{class_id}_{i}.png"
            image_path = os.path.join(output_images_folder, image_filename)
            cv2.imwrite(image_path, image)
            
            label_data = {
                "class_name": class_name,
                "class_id": int(class_id),
                "rgb": list(rgb)
            }
            label_path = os.path.join(output_labels_folder, f"{class_id}_{i}.json")
            
            with open(label_path, 'w') as json_file:
                json.dump(label_data, json_file, indent=4)

if __name__ == "__main__":
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)

    excel_path = "lego.xlsx"
    generate_dataset(excel_path)
    print(f"Dataset created & located in '{output_images_folder}' and '{output_labels_folder}'")
