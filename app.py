from pathlib import Path
import json
import gradio as gr
from PIL import Image
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from segmentation.opencv_segmentation import extract_object_from_file
from utils.visualize_tools import overlay_mask_with_contour

ROOT_DIR = Path.cwd()
MODELS_DIR = ROOT_DIR / "models"
TEST_DATA_DIR = ROOT_DIR / "data" / "test"
CLASSES_JSON_PATH = ROOT_DIR / "data" / "classes" / "class_to_name.json"

def classify_color(image):
    _image, _, mask, result = extract_object_from_file(image)

    if mask.shape[:2] != _image.shape[:2]:
        raise ValueError("Размеры маски и изображения не совпадают!")

    processed_image = overlay_mask_with_contour(
        _image,
        mask,
        fill_color=(139, 0, 255),
        alpha=0.4,
        contour_color=(139, 0, 255),
        thickness=50
    )

    predicted_class = "Class"
    confidence = 0.95

    model = torch.jit.load(MODELS_DIR / "scripted_model.pt")
    model.load_state_dict(torch.load(MODELS_DIR / "best_model_weights.pth", weights_only=True))

    model = model.to("cpu")
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),

    ])
    input_tensor = preprocess(Image.fromarray(result)).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
    
    with open(CLASSES_JSON_PATH, "r") as f:
        class_to_name = json.load(f)

    predicted_class = class_to_name[str(predicted_class_idx)]
    confidence = probabilities[0, predicted_class_idx].item()
    return processed_image, f"{predicted_class} (Confidence: {confidence:.2f})"


if __name__ == "__main__":
    sample_images = [
        str(TEST_DATA_DIR / "1-1.JPG"),
        str(TEST_DATA_DIR / "22-3.JPG"),
        str(TEST_DATA_DIR / "19-1.JPG"),
        str(TEST_DATA_DIR / "189-3.JPG"),
        str(TEST_DATA_DIR / "161-1.JPG"),
        str(TEST_DATA_DIR / "73-1.JPG"),
        str(TEST_DATA_DIR / "11-1.JPG"),
        str(TEST_DATA_DIR / "294-1.JPG"),
        str(TEST_DATA_DIR / "258-5.JPG") 
    ]

    interface = gr.Interface(
        fn=classify_color,
        inputs=gr.Image(type="pil", label="Загрузите изображение детальки"),
        outputs=[
            gr.Image(type="pil", label="Предобработанное изображение"),
            gr.Textbox(label="Результат классификации")
        ],
        title="Классификация цветов",
        description="Загрузите изображение, чтобы узнать, к какому цвету оно относится. Предобработанное изображение также будет показано.",
        examples=sample_images
    )
    interface.launch()
