import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, OmDetTurboForObjectDetection

# Load model once at startup (cached in memory)
MODEL_ID = "omlab/omdet-turbo-swin-tiny-hf"

print("Loading OmDet-Turbo model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = OmDetTurboForObjectDetection.from_pretrained(MODEL_ID)
model.eval()
print("OmDet-Turbo loaded successfully!")

# Text classes to detect — covers all banner/signboard types
DETECTION_CLASSES = [
    "banner",
    "hoarding",
    "signboard",
    "billboard",
    "flyer",
    "poster",
    "flex board"
]

def detect_banner(image_bytes: bytes, ref_box: dict) -> dict:
    """
    Detects the banner in the image, ignoring the reference box area.
    Returns: { bbox: {x, y, w, h}, centroid: {cx, cy} }
    """
    # Decode image bytes to numpy
    nparr = np.frombuffer(image_bytes, np.uint8)
    cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if cv_image is None:
        raise ValueError("Could not decode image")

    # Convert BGR to RGB for PIL
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    img_h, img_w = cv_image.shape[:2]

    # Run OmDet-Turbo inference
    inputs = processor(
        images=pil_image,
        text=DETECTION_CLASSES,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_grounded_object_detection(
        outputs,
        classes=DETECTION_CLASSES,
        target_sizes=[(img_h, img_w)],
        score_threshold=0.25,
        nms_threshold=0.3,
    )[0]

    if len(results["scores"]) == 0:
        raise ValueError("No banner detected in image. Try again with better lighting.")

    # Filter out detections that overlap with the reference box
    best_box = None
    best_score = 0.0

    for score, label, box in zip(
        results["scores"],
        results["classes"],
        results["boxes"]
    ):
        x1, y1, x2, y2 = box.tolist()
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Skip if centroid is inside the reference box
        rx, ry, rw, rh = ref_box["x"], ref_box["y"], ref_box["w"], ref_box["h"]
        if rx < cx < rx + rw and ry < cy < ry + rh:
            continue

        if score.item() > best_score:
            best_score = score.item()
            best_box = {
                "bbox": {
                    "x": int(x1),
                    "y": int(y1),
                    "w": int(x2 - x1),
                    "h": int(y2 - y1)
                },
                "centroid": {"cx": cx, "cy": cy},
                "label": label,
                "score": round(best_score, 3)
            }

    if best_box is None:
        raise ValueError("Banner detected but overlaps with reference object. Please reposition.")

    print(f"Detected: {best_box['label']} | score: {best_box['score']} | box: {best_box['bbox']}")
    return best_box
