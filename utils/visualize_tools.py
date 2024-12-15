import numpy as np
from PIL import Image, ImageFilter, ImageDraw

def overlay_mask_with_contour(image, mask, fill_color=(255, 0, 0), alpha=0.5, contour_color=(255, 0, 0), thickness=3):

    mask = (mask > 0).astype(np.uint8)

    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[:, :] = fill_color

    result = image.copy()
    result[mask == 1] = (result[mask == 1] * (1 - alpha) + color_mask[mask == 1] * alpha).astype(np.uint8)

    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    edges = mask_pil.filter(ImageFilter.FIND_EDGES)

    contour_layer = Image.new("RGB", mask_pil.size, (0, 0, 0))
    draw = ImageDraw.Draw(contour_layer)
    for _ in range(thickness):
        draw.bitmap((0, 0), edges, fill=contour_color)

    result_pil = Image.fromarray(result)
    result_pil.paste(contour_layer, (0, 0), mask=edges)

    return result_pil
