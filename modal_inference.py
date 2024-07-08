import modal
from fastapi import HTTPException, status
import requests
import os


GPU_TYPE = os.getenv("GPU_TYPE", "t4")
GPU_COUNT = int(os.getenv("GPU_COUNT", "1"))
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

CONFIG_FILE = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CKPT_REPO_ID = "ShilongLiu/GroundingDINO"
CKPT_FILENAME = "groundingdino_swint_ogc.pth"

SAM_REPO_ID = "ybelkada/segment-anything"
SAM_CKPT = "sam_vit_h_4b8939.pth"

MINUTES = 60

LOG_DIVIDER = "\n==========================\n"


def download_model_to_image():
    from huggingface_hub import snapshot_download
    import transformers

    snapshot_download(repo_id=SAM_REPO_ID)
    transformers.utils.move_cache()


grounded_sam_image = modal.Image.from_registry(
    "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10"
)

grounded_sam_image = grounded_sam_image.apt_install(
    "openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget"
).pip_install(
    "torch",
    "torchvision",
    "onnxruntime-gpu",
    "transformers",
    "diffusers",
    "numpy",
    "opencv-python",
    "pillow",
    "huggingface_hub",
    "pyyaml",
    "setuptools",
    "wheel",
    pre=True,
    extra_index_url="https://download.pytorch.org/whl/cu121",
)

grounded_sam_image = grounded_sam_image \
    .workdir("/root/") \
    .copy_local_dir(".", ".") \
    .run_commands(
        "ls -l",
        "python -m pip install -e segment_anything",
        "pip install --no-build-isolation -e GroundingDINO",
    ) \
    .run_function(download_model_to_image)


app = modal.App("grounded-sam")


@app.cls(
    gpu=GPU_CONFIG,
    timeout=2 * MINUTES,
    container_idle_timeout=1 * MINUTES,
    allow_concurrent_inputs=5,
    image=grounded_sam_image,
)
class Model:
    @modal.enter()
    def start_runtime(self):
        print(f"{LOG_DIVIDER}Starting runtime...{LOG_DIVIDER}")

        from gradio_app import (
            load_model,
            build_sam,
            SamPredictor,
        )
        import torch

        self.device = torch.device("cuda")
        self.grounding_model = load_model(
            CONFIG_FILE, CKPT_FILENAME, device=self.device
        )
        self.sam = build_sam(checkpoint=SAM_CKPT)
        self.sam.to(self.device)
        self.sam_predictor = SamPredictor(self.sam)

        print(f"{LOG_DIVIDER}Runtime started{LOG_DIVIDER}")

    @modal.web_endpoint(method="POST", docs=True)
    def predict(self, request: dict):
        from gradio_app import (
            transform_image,
            get_grounding_output,
            draw_box,
            draw_mask,
        )
        from PIL import Image, ImageDraw
        from io import BytesIO
        import base64
        import numpy as np
        import torch


        if "prompt" not in request:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No prompt provided."
            )

        if "image_url" not in request and "image_base64" not in request:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No image URL or image base64 data provided."
            )

        def get_image_from_request(request: dict) -> Image.Image:
            if "image_url" in request:
                response = requests.get(request["image_url"])
                if not response.ok:
                    raise ValueError(f"[ERROR] Failed to download image from {request['image_url']}")
                return Image.open(BytesIO(response.content)).convert("RGB")
            elif "base64" in request:
                img_data = base64.b64decode(request["base64"])
                return Image.open(BytesIO(img_data)).convert("RGB")
            else:
                raise ValueError("[ERROR] No image URL or image data provided")

        prompt = request.get("prompt", None)
        box_threshold = request.get("box_threshold", 0.3)
        text_threshold = request.get("text_threshold", 0.25)

        # 1. load image and prepare
        image_pil = get_image_from_request(request)
        image = np.array(image_pil)
        transformed_image = transform_image(image_pil)

        # 2. run groudning dino
        boxes_filt, scores, pred_phrases = get_grounding_output(
            self.grounding_model, transformed_image, prompt,
            box_threshold, text_threshold
        )

        H, W = image_pil.size[1], image_pil.size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()

        # Run SAM
        self.sam_predictor.set_image(image)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )

        # Draw masks
        mask_image = Image.new('RGBA', image_pil.size, color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
        for mask in masks:
            draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)

        result_image = result_image.convert('RGBA')
        result_image.alpha_composite(mask_image)

        result_image = result_image.convert("RGB")

        buffered = BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "segmented_image": f"data:image/jpeg;{img_str}"
        }

    @modal.exit()
    def shutdown_runtime(self):
        print(f"{LOG_DIVIDER}Shutting down runtime...{LOG_DIVIDER}")
        self.grounding_model = None
        self.sam = None
        self.sam_predictor = None
        print(f"{LOG_DIVIDER}Runtime shut down{LOG_DIVIDER}")



@app.local_entrypoint()
def main(image_url: str = None, prompt: str = None):
    from PIL import Image
    from io import BytesIO
    import base64

    model = Model()

    response = requests.post(
        model.predict.web_url,
        json={
            "image_url": image_url,
            "prompt": prompt
        }
    )

    data = response.json()
    print(data)

    img_b64 = data["segmented_image"]
    img_b64 = img_b64.split(",")[1]
    img_b64 = base64.b64decode(img_b64)
    img_pil = Image.open(BytesIO(img_b64))
    img_pil.show()


if __name__ == "__main__":
    main()
