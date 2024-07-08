import argparse
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

# Import necessary functions and classes from your existing code
from gradio_app import (
    load_model,
    transform_image,
    get_grounding_output,
    build_sam,
    SamPredictor,
    draw_box,
    draw_mask,
)

def main(args):
    # Load models
    groundingdino_model = load_model(args.config_file, args.ckpt_filename, device=args.device)
    sam = build_sam(checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    sam_predictor = SamPredictor(sam)

    # Load and prepare image
    image_pil = Image.open(args.input_image).convert("RGB")
    image = np.array(image_pil)
    transformed_image = transform_image(image_pil)

    # Run Grounding DINO
    boxes_filt, scores, pred_phrases = get_grounding_output(
        groundingdino_model, transformed_image, args.text_prompt, 
        args.box_threshold, args.text_threshold
    )

    # Process boxes
    H, W = image_pil.size[1], image_pil.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()

    # Run SAM
    sam_predictor.set_image(image)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(args.device)
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # Visualize results
    result_image = image_pil.copy()
    draw = ImageDraw.Draw(result_image)

    # Draw boxes and labels
    for box, label in zip(boxes_filt, pred_phrases):
        draw_box(box, draw, label)

    # Draw masks
    mask_image = Image.new('RGBA', image_pil.size, color=(0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)
    for mask in masks:
        draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)

    result_image = result_image.convert('RGBA')
    result_image.alpha_composite(mask_image)

    # Save result
    result_image.save(args.output_image)
    print(f"Result saved to {args.output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded SAM CLI", add_help=True)
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_image", type=str, required=True, help="Path to save output image")
    parser.add_argument("--text_prompt", type=str, required=True, help="Text prompt for grounding")
    parser.add_argument("--config_file", type=str, default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--ckpt_filename", type=str, default="groundingdino_swint_ogc.pth")
    parser.add_argument("--sam_checkpoint", type=str, default="sam_vit_h_4b8939.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    
    args = parser.parse_args()
    main(args)
