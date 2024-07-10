import modal
from fastapi import HTTPException, status
import requests
import os


REQUIRED_MODEL_FILES = ["groundingdino_swint_ogc.pth", "sam_vit_h_4b8939.pth"]
if not all(os.path.exists(file) for file in REQUIRED_MODEL_FILES):
    text = f"[ERROR] Required model files not found: {', '.join(REQUIRED_MODEL_FILES)}"
    raise ValueError(text)


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


grounded_sam_image = modal.Image.from_registry(
    "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10"
)

grounded_sam_image = grounded_sam_image.apt_install(
    "openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget", "clang"
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
    "ninja",
    "cython",
    pre=True,
    extra_index_url="https://download.pytorch.org/whl/cu121",
)

grounded_sam_image = grounded_sam_image \
    .workdir("/root/") \
    .copy_local_dir(".", ".") \
    .env({
        "AM_I_DOCKER": "True",
        "BUILD_WITH_CUDA": "1",
        "CUDA_HOME": "/usr/local/cuda-12.1",
        "TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.6+PTX",
    }) \
    .run_commands(
        "ls -l /usr/local/ | grep -i cuda",
        "env | grep -i cuda",
        "python -m pip install -e segment_anything",
        "python -m pip install --no-build-isolation -e GroundingDINO",
        "cd GroundingDINO && python setup.py build_ext --inplace",
        "ls -l GroundingDINO/groundingdino/"
    )


app = modal.App("grounded-sam")


@app.cls(
    gpu=GPU_CONFIG,
    timeout=2 * MINUTES,
    container_idle_timeout=1 * MINUTES,
    allow_concurrent_inputs=7,
    image=grounded_sam_image,
)
class Model:
    @modal.enter()
    def start_runtime(self):
        print(f"{LOG_DIVIDER}Starting runtime...{LOG_DIVIDER}")

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from modal_inference_utils import load_model
        from segment_anything import build_sam, SamPredictor
        import torch
        import time

        self.device = torch.device("cuda")

        def load_models_concurrently(load_functions_map: dict) -> dict:
            model_id_to_model = {}
            with ThreadPoolExecutor(max_workers=len(load_functions_map)) as executor:
                future_to_model_id = {
                    executor.submit(load_fn): model_id for model_id, load_fn in load_functions_map.items()
                }
                for future in as_completed(future_to_model_id.keys()):
                    model_id_to_model[future_to_model_id[future]] = future.result()

            return model_id_to_model


        def load_sam(ckpt, device):
            sam = build_sam(checkpoint=ckpt)
            sam.to(device)
            return SamPredictor(sam)

        start = time.time()

        components = load_models_concurrently({
            "grounding_model": lambda: load_model(CONFIG_FILE, CKPT_FILENAME, device=self.device),
            "sam": lambda: load_sam(SAM_CKPT, self.device),
        })

        self.grounding_model = components["grounding_model"]
        self.sam_predictor = components["sam"]

        end = time.time()
        print(f"[INFO] Loaded models in {end - start:.2f} seconds")

        print(f"{LOG_DIVIDER}Runtime started{LOG_DIVIDER}")

    @modal.web_endpoint(method="POST", docs=True)
    def predict(self, request: dict):
        from modal_inference_utils import (
            transform_image,
            get_grounding_output,
            draw_box,
            draw_mask
        )
        from PIL import Image, ImageDraw
        from io import BytesIO
        import base64
        import numpy as np
        import torch
        import time
        import gc


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
        return_type = request.get("return_type", "base64")

        # 1. load image and prepare
        image_pil = get_image_from_request(request)
        image = np.array(image_pil)
        transformed_image = transform_image(image_pil)

        # 2. run groudning dino
        start = time.time()
        boxes_filt, scores, pred_phrases = get_grounding_output(
            self.grounding_model, transformed_image, prompt,
            box_threshold, text_threshold
        )
        end = time.time()
        print(f"[INFO] get_grounding_output time: {end - start:.2f} seconds")

        start = time.time()
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
        end = time.time()
        print(f"[INFO] SAM inference time: {end - start:.2f} seconds")

        start = time.time()
        b64_masks = []
        for mask in masks:
            mask_np = mask[0].cpu().numpy()
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')
            buffered = BytesIO()
            mask_pil.save(buffered, format="PNG")
            b64_mask = base64.b64encode(buffered.getvalue()).decode("utf-8")
            b64_masks.append(f"data:image/png;base64,{b64_mask}")
        end = time.time()
        print(f"[INFO] Mask encoding time: {end - start:.2f} seconds")


        start = time.time()
        # clear gpu memory
        torch.cuda.empty_cache()
        gc.collect()
        end = time.time()
        print(f"[INFO] GPU memory cleared in {end - start:.2f} seconds")

        return {
            "masks": b64_masks
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
    img_b64 = data["segmented_image"]
    img_b64 = img_b64.split(",")[1]
    img_b64 = base64.b64decode(img_b64)
    img_pil = Image.open(BytesIO(img_b64))
    img_pil.show()


if __name__ == "__main__":
    main()
