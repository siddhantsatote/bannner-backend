import os
import re
import json
import base64
import binascii
import urllib.request
import urllib.error
import traceback
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------

HF_API_TOKEN: str | None = os.getenv("HF_API_TOKEN")
HF_API_BASE_URL: str = os.getenv("HF_API_BASE_URL", "https://api-inference.huggingface.co")
HF_OBJECT_MODEL: str = os.getenv("HF_OBJECT_MODEL", "google/owlv2-base-patch16-ensemble")
HF_LLM_MODEL: str | None = os.getenv("HF_LLM_MODEL")

_hf_client: InferenceClient | None = None

# Phone physical dimensions (width_cm x height_cm) — screen face only
PHONE_DIMENSIONS: dict[str, "PhoneDimensions"] = {}


def _build_phone_dimensions() -> dict[str, "PhoneDimensions"]:
    return {
        "iphone-se":       PhoneDimensions(width_cm=5.87,  height_cm=11.37),
        "iphone-12-mini":  PhoneDimensions(width_cm=6.07,  height_cm=13.17),
        "iphone-12":       PhoneDimensions(width_cm=7.15,  height_cm=14.67),
        "iphone-12-pro":   PhoneDimensions(width_cm=7.15,  height_cm=14.67),
        "iphone-12-pro-max": PhoneDimensions(width_cm=7.81, height_cm=16.08),
        "iphone-13-mini":  PhoneDimensions(width_cm=6.07,  height_cm=13.17),
        "iphone-13":       PhoneDimensions(width_cm=7.15,  height_cm=14.67),
        "iphone-13-pro":   PhoneDimensions(width_cm=7.15,  height_cm=14.67),
        "iphone-13-pro-max": PhoneDimensions(width_cm=7.81, height_cm=16.08),
        "iphone-14":       PhoneDimensions(width_cm=7.15,  height_cm=14.67),
        "iphone-14-plus":  PhoneDimensions(width_cm=7.81,  height_cm=16.08),
        "iphone-14-pro":   PhoneDimensions(width_cm=7.15,  height_cm=14.67),
        "iphone-14-pro-max": PhoneDimensions(width_cm=7.81, height_cm=16.08),
        "iphone-15":       PhoneDimensions(width_cm=7.15,  height_cm=14.67),
        "iphone-15-plus":  PhoneDimensions(width_cm=7.81,  height_cm=16.08),
        "iphone-15-pro":   PhoneDimensions(width_cm=7.02,  height_cm=14.95),
        "iphone-15-pro-max": PhoneDimensions(width_cm=7.69, height_cm=16.39),
        "samsung-s23":     PhoneDimensions(width_cm=7.06,  height_cm=14.66),
        "samsung-s23-plus": PhoneDimensions(width_cm=7.60, height_cm=15.75),
        "samsung-s23-ultra": PhoneDimensions(width_cm=7.86, height_cm=16.35),
        "samsung-s24":     PhoneDimensions(width_cm=7.01,  height_cm=14.73),
        "samsung-s24-plus": PhoneDimensions(width_cm=7.58, height_cm=15.84),
        "samsung-s24-ultra": PhoneDimensions(width_cm=7.90, height_cm=16.23),
        "pixel-7":         PhoneDimensions(width_cm=7.25,  height_cm=15.52),
        "pixel-7-pro":     PhoneDimensions(width_cm=7.61,  height_cm=16.28),
        "pixel-8":         PhoneDimensions(width_cm=7.07,  height_cm=15.03),
        "pixel-8-pro":     PhoneDimensions(width_cm=7.59,  height_cm=16.25),
    }


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RefBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class PhoneDimensions(BaseModel):
    width_cm: float
    height_cm: float


class AnalyzeResponse(BaseModel):
    width_cm: float
    height_cm: float
    aspect_ratio: float
    mask_contour: list[list[float]]
    cm_per_pixel: float | None = None
    detected_label: str | None = None
    detected_score: float | None = None


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Banner Buddy API", version="2.0.0")

# Build phone dimensions after models are defined
PHONE_DIMENSIONS = _build_phone_dimensions()

allowed_origins = os.getenv("CORS_ORIGINS", "*")
cors_origins = [origin.strip() for origin in allowed_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _validate_image_data(image_data: str) -> None:
    if "," in image_data:
        _, encoded = image_data.split(",", 1)
    else:
        encoded = image_data

    try:
        base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image_data payload") from exc


def _decode_image_bytes(image_data: str) -> bytes:
    if "," in image_data:
        _, encoded = image_data.split(",", 1)
    else:
        encoded = image_data
    return base64.b64decode(encoded)


def _get_hf_client() -> InferenceClient:
    global _hf_client
    if _hf_client is None:
        _hf_client = InferenceClient(token=HF_API_TOKEN or None)
    return _hf_client


def _get_phone_dimensions(phone_model: str) -> PhoneDimensions:
    if phone_model in PHONE_DIMENSIONS:
        return PHONE_DIMENSIONS[phone_model]

    env_width = os.getenv("PHONE_WIDTH_CM")
    env_height = os.getenv("PHONE_HEIGHT_CM")
    if env_width and env_height:
        return PhoneDimensions(width_cm=float(env_width), height_cm=float(env_height))

    raise HTTPException(
        status_code=400,
        detail=(
            f"Unknown phone model '{phone_model}'. Set PHONE_WIDTH_CM and PHONE_HEIGHT_CM "
            "or add the model to PHONE_DIMENSIONS."
        ),
    )


def _cm_per_pixel_from_phone(phone_box: RefBox, phone_dimensions: PhoneDimensions) -> float:
    if phone_box.w <= 0 or phone_box.h <= 0:
        raise HTTPException(status_code=400, detail="phone_box width and height must be greater than zero")

    cm_per_px_width = phone_dimensions.width_cm / phone_box.w
    cm_per_px_height = phone_dimensions.height_cm / phone_box.h
    return (cm_per_px_width + cm_per_px_height) / 2.0


def _phone_scales(phone_box: RefBox, phone_dimensions: PhoneDimensions) -> tuple[float, float]:
    if phone_box.w <= 0 or phone_box.h <= 0:
        raise HTTPException(status_code=400, detail="phone_box width and height must be greater than zero")

    return (
        phone_dimensions.width_cm / phone_box.w,
        phone_dimensions.height_cm / phone_box.h,
    )


def _measure_box_cm(box: RefBox, cm_per_pixel: float) -> tuple[float, float]:
    return round(box.w * cm_per_pixel, 2), round(box.h * cm_per_pixel, 2)


def _normalize_object_prompt(prompt: str) -> str:
    cleaned = re.sub(r"\s+", " ", prompt).strip().strip('"\'`.,:;!?')
    if not cleaned:
        raise HTTPException(status_code=400, detail="object_prompt must not be empty")

    if not HF_LLM_MODEL:
        return cleaned

    try:
        response = _get_hf_client().text_generation(
            prompt=(
                "Rewrite the object description below as a short, concrete noun phrase for image detection. "
                "Return only the phrase, no punctuation, no explanation.\n\n"
                f"Object description: {cleaned}"
            ),
            model=HF_LLM_MODEL,
            max_new_tokens=24,
            temperature=0.1,
            return_full_text=False,
        )
    except Exception:
        return cleaned

    if isinstance(response, str):
        normalized = re.sub(r"\s+", " ", response).strip().strip('"\'`.,:;!?')
        return normalized or cleaned

    return cleaned


def _hf_detect_object(image_data: str, object_prompt: str) -> tuple[RefBox, str | None, float | None]:
    if not HF_API_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="HF_API_TOKEN is not configured. Add it in Railway to enable AI object detection.",
        )

    prompt = _normalize_object_prompt(object_prompt)

    candidate_labels = [p.strip() for p in re.split(r"[,\n]", prompt) if p.strip()]
    if not candidate_labels:
        raise HTTPException(status_code=400, detail="object_prompt must contain at least one valid label")

    image_bytes = _decode_image_bytes(image_data)
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    if "," in image_data and image_data.startswith("data:"):
        mime_prefix = image_data.split(",", 1)[0]
        image_input = f"{mime_prefix},{encoded_image}"
    else:
        image_input = f"data:image/jpeg;base64,{encoded_image}"

    url = f"{HF_API_BASE_URL}/pipeline/zero-shot-object-detection"
    request_body = json.dumps(
        {
            "model": HF_OBJECT_MODEL,
            "inputs": image_input,
            "parameters": {
                "candidate_labels": candidate_labels,
                "threshold": 0.1,
            },
            "options": {
                "wait_for_model": True,
            },
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=request_body,
        method="POST",
        headers={
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        normalized = body.strip() or exc.reason
        print(f"[HF ERROR] HTTPError: {exc.code} {normalized}")
        raise HTTPException(
            status_code=502,
            detail={
                "error": f"Hugging Face detection failed: {normalized}",
                "model": HF_OBJECT_MODEL,
                "prompt": object_prompt,
            },
        ) from exc
    except urllib.error.URLError as exc:
        print(f"[HF ERROR] URLError: {exc.reason}")
        raise HTTPException(status_code=502, detail={
            "error": f"Hugging Face detection unreachable: {exc.reason}",
            "model": HF_OBJECT_MODEL,
            "prompt": object_prompt,
        }) from exc
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[HF ERROR] Exception: {tb}")
        raise HTTPException(status_code=502, detail={
            "error": f"Unexpected backend error: {str(exc)}",
            "model": HF_OBJECT_MODEL,
            "prompt": object_prompt,
        }) from exc

    if isinstance(payload, dict) and payload.get("error"):
        print(f"[HF ERROR] API error: {payload['error']}")
        raise HTTPException(status_code=502, detail={
            "error": f"Hugging Face error: {payload['error']}",
            "model": HF_OBJECT_MODEL,
            "prompt": object_prompt,
        })

    detections = payload if isinstance(payload, list) else []

    if not detections:
        print("[HF ERROR] No object detected in the image")
        raise HTTPException(status_code=404, detail={
            "error": "No object detected in the image",
            "model": HF_OBJECT_MODEL,
            "prompt": object_prompt,
        })

    def score_of(item: Any) -> float:
        if isinstance(item, dict):
            return float(item.get("score", 0.0) or 0.0)
        return float(getattr(item, "score", 0.0) or 0.0)

    best = max(detections, key=score_of)

    if isinstance(best, dict):
        box = best.get("box", {}) or {}
        xmin = float(box.get("xmin", box.get("x", 0.0)) or 0.0)
        ymin = float(box.get("ymin", box.get("y", 0.0)) or 0.0)
        xmax = float(box.get("xmax", box.get("width", xmin)) or xmin)
        ymax = float(box.get("ymax", box.get("height", ymin)) or ymin)
        label = best.get("label")
        score = float(best.get("score", 0.0) or 0.0)
    else:
        box = getattr(best, "box", None)
        xmin = float(getattr(box, "xmin", 0.0) or 0.0)
        ymin = float(getattr(box, "ymin", 0.0) or 0.0)
        xmax = float(getattr(box, "xmax", 0.0) or 0.0)
        ymax = float(getattr(box, "ymax", 0.0) or 0.0)
        label = getattr(best, "label", None)
        score = float(getattr(best, "score", 0.0) or 0.0)

    return (
        RefBox(x=xmin, y=ymin, w=max(xmax - xmin, 1.0), h=max(ymax - ymin, 1.0)),
        str(label) if label is not None else None,
        score,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    image: UploadFile = File(...),
    ref_box: str = Form(...),
    ref_size_cm: float = Form(...),
    object_prompt: str = Form(default="banner, hoarding, signboard, billboard, poster, flex board"),
):
    try:
        image_bytes = await image.read()
        ref_box_dict = json.loads(ref_box)

        # Encode image as base64 data URL for HF API
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        content_type = image.content_type or "image/jpeg"
        image_data = f"data:{content_type};base64,{encoded}"

        # Detect banner via HuggingFace zero-shot object detection
        detected_box, detected_label, detected_score = _hf_detect_object(image_data, object_prompt)

        bbox_x = detected_box.x
        bbox_y = detected_box.y
        bbox_w = detected_box.w
        bbox_h = detected_box.h

        # Build mask contour from bounding box
        mask_contour = [
            [bbox_x, bbox_y],
            [bbox_x + bbox_w, bbox_y],
            [bbox_x + bbox_w, bbox_y + bbox_h],
            [bbox_x, bbox_y + bbox_h],
        ]

        # Use reference box to compute cm-per-pixel scale
        ref = RefBox(**ref_box_dict)
        # ref_size_cm is the known physical size of the reference object (e.g. phone width in cm)
        cm_per_pixel = ref_size_cm / ref.w if ref.w > 0 else None

        if cm_per_pixel:
            width_cm = round(bbox_w * cm_per_pixel, 2)
            height_cm = round(bbox_h * cm_per_pixel, 2)
        else:
            width_cm = round(bbox_w, 2)
            height_cm = round(bbox_h, 2)

        aspect_ratio = round(width_cm / max(height_cm, 1e-6), 2)

        return AnalyzeResponse(
            width_cm=width_cm,
            height_cm=height_cm,
            aspect_ratio=aspect_ratio,
            mask_contour=mask_contour,
            cm_per_pixel=cm_per_pixel,
            detected_label=detected_label,
            detected_score=detected_score,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
