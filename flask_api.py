from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import base64
import requests
import torch
import numpy as np
import gc
from modal_inference_utils import transform_image, get_grounding_output
from modal_inference import Model

app = Flask(__name__)

model = Model()
model.start_runtime()

def get_image_from_request(request):
    if "image_url" in request:
        response = requests.get(request["image_url"])
        if not response.ok:
            raise ValueError(f"[ERROR] Failed to download image from {request['image_url']}")
        return Image.open(BytesIO(response.content)).convert("RGB")
    elif "image_base64" in request:
        img_data = base64.b64decode(request["image_base64"])
        return Image.open(BytesIO(img_data)).convert("RGB")
    else:
        raise ValueError("[ERROR] No image URL or image data provided")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    if "prompt" not in data:
        return jsonify({"error": "No prompt provided."}), 400

    if "image_url" not in data and "image_base64" not in data:
        return jsonify({"error": "No image URL or image base64 data provided."}), 400

    prompt = data.get("prompt", None)
    box_threshold = data.get("box_threshold", 0.3)
    text_threshold = data.get("text_threshold", 0.25)

    # 1. load image and prepare
    image_pil = get_image_from_request(data)
    image = np.array(image_pil)
    transformed_image = transform_image(image_pil)

    # 2. run grounding dino
    boxes_filt, scores, pred_phrases = get_grounding_output(
        model.grounding_model, transformed_image, prompt,
        box_threshold, text_threshold
    )

    H, W = image_pil.size[1], image_pil.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()

    # Run SAM
    model.sam_predictor.set_image(image)
    transformed_boxes = model.sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(model.device)
    masks, _, _ = model.sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False
    )

    b64_masks = []
    for mask in masks:
        mask_np = mask[0].cpu().numpy()
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')
        buffered = BytesIO()
        mask_pil.save(buffered, format="PNG")
        b64_mask = base64.b64encode(buffered.getvalue()).decode("utf-8")
        b64_masks.append(f"data:image/png;base64,{b64_mask}")

    # clear gpu memory
    torch.cuda.empty_cache()
    gc.collect()

    return jsonify({"masks": b64_masks})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
