from __future__ import annotations

import base64
import binascii
import json
import os
import re
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field


class RefBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class PhoneDimensions(BaseModel):
    width_cm: float
    height_cm: float


class AnalyzeRequest(BaseModel):
    image_data: str = Field(..., description="Base64 data URL for the captured image")
    phone_box: RefBox | None = None
    object_box: RefBox | None = None
    phone_model: str = "oneplus_nord_ce_5"
    ref_box: RefBox | None = Field(default=None, description="Manual phone reference box when YOLO is skipped")
    ref_size_cm: float | None = Field(default=None, gt=0)
    object_prompt: str | None = Field(default=None, description="Text prompt describing the object to detect")


class AnalyzeResponse(BaseModel):
    width_cm: float
    height_cm: float
    aspect_ratio: float
    mask_contour: list[list[float]]
    cm_per_pixel: float | None = None
    detected_label: str | None = None
    detected_score: float | None = None


PHONE_DIMENSIONS: dict[str, PhoneDimensions] = {
    # Replace this with the exact device dimensions you want to support.
    # Keeping it configurable avoids hard-coding a potentially wrong model size.
    "oneplus_nord_ce_5": PhoneDimensions(width_cm=7.602, height_cm=16.358),
}

HF_OBJECT_MODEL = os.getenv("HF_OBJECT_MODEL", "google/owlv2-base-patch16-ensemble")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "").strip()

_hf_client: InferenceClient | None = None


app = FastAPI(title="Banner Buddy API", version="1.0.0")

allowed_origins = os.getenv("CORS_ORIGINS", "*")
cors_origins = [origin.strip() for origin in allowed_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    image_bytes = _decode_image_bytes(image_data)

    prompt = _normalize_object_prompt(object_prompt)

    try:
        detections = _get_hf_client().zero_shot_object_detection(
            image_bytes,
            candidate_labels=[prompt],
            model=HF_OBJECT_MODEL,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Hugging Face detection failed: {exc}") from exc

    if not detections:
        raise HTTPException(status_code=404, detail="No object detected in the image")

    ranked = sorted(detections, key=lambda det: float(getattr(det, "score", 0.0) or 0.0), reverse=True)
    best = ranked[0]
    box = getattr(best, "box", None)
    xmin = float(getattr(box, "xmin", 0.0) or 0.0)
    ymin = float(getattr(box, "ymin", 0.0) or 0.0)
    xmax = float(getattr(box, "xmax", 0.0) or 0.0)
    ymax = float(getattr(box, "ymax", 0.0) or 0.0)

    return (
        RefBox(x=xmin, y=ymin, w=max(xmax - xmin, 1.0), h=max(ymax - ymin, 1.0)),
        str(getattr(best, "label", None)) if getattr(best, "label", None) is not None else None,
        float(getattr(best, "score", 0.0) or 0.0),
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    _validate_image_data(payload.image_data)

    # Manual phone reference mode: if the frontend only sends ref_box + ref_size_cm,
    # treat that box as the phone box and use the known phone width for scaling.
    if payload.ref_box and payload.ref_size_cm and not payload.phone_box:
        payload.phone_box = payload.ref_box

    if payload.phone_box and payload.object_box:
        phone_dimensions = _get_phone_dimensions(payload.phone_model)
        cm_per_pixel_x, cm_per_pixel_y = _phone_scales(payload.phone_box, phone_dimensions)
        width_cm = round(payload.object_box.w * cm_per_pixel_x, 2)
        height_cm = round(payload.object_box.h * cm_per_pixel_y, 2)
        aspect_ratio = round(width_cm / max(height_cm, 1e-6), 2)

        contour: list[list[float]] = [
            [payload.object_box.x, payload.object_box.y],
            [payload.object_box.x + payload.object_box.w, payload.object_box.y],
            [payload.object_box.x + payload.object_box.w, payload.object_box.y + payload.object_box.h],
            [payload.object_box.x, payload.object_box.y + payload.object_box.h],
        ]

        return AnalyzeResponse(
            width_cm=width_cm,
            height_cm=height_cm,
            aspect_ratio=aspect_ratio,
            mask_contour=contour,
            cm_per_pixel=round((cm_per_pixel_x + cm_per_pixel_y) / 2.0, 6),
        )

    if payload.phone_box and not payload.object_box:
        if not payload.object_prompt or not payload.object_prompt.strip():
            raise HTTPException(status_code=400, detail="object_prompt is required when using phone-based measurement")

        phone_dimensions = _get_phone_dimensions(payload.phone_model)
        cm_per_pixel_x, cm_per_pixel_y = _phone_scales(payload.phone_box, phone_dimensions)
        detected_box, detected_label, detected_score = _hf_detect_object(payload.image_data, payload.object_prompt)
        width_cm = round(detected_box.w * cm_per_pixel_x, 2)
        height_cm = round(detected_box.h * cm_per_pixel_y, 2)
        aspect_ratio = round(width_cm / max(height_cm, 1e-6), 2)

        contour: list[list[float]] = [
            [detected_box.x, detected_box.y],
            [detected_box.x + detected_box.w, detected_box.y],
            [detected_box.x + detected_box.w, detected_box.y + detected_box.h],
            [detected_box.x, detected_box.y + detected_box.h],
        ]

        return AnalyzeResponse(
            width_cm=width_cm,
            height_cm=height_cm,
            aspect_ratio=aspect_ratio,
            mask_contour=contour,
            cm_per_pixel=round((cm_per_pixel_x + cm_per_pixel_y) / 2.0, 6),
            detected_label=detected_label,
            detected_score=detected_score,
        )

    if payload.ref_box and payload.ref_size_cm:
        width_cm = round(payload.ref_size_cm * max(payload.ref_box.w / max(payload.ref_box.h, 1.0), 1.0), 2)
        height_cm = round(payload.ref_size_cm, 2)
        aspect_ratio = round(width_cm / max(height_cm, 1e-6), 2)

        contour: list[list[float]] = [
            [payload.ref_box.x, payload.ref_box.y],
            [payload.ref_box.x + payload.ref_box.w, payload.ref_box.y],
            [payload.ref_box.x + payload.ref_box.w, payload.ref_box.y + payload.ref_box.h],
            [payload.ref_box.x, payload.ref_box.y + payload.ref_box.h],
        ]

        return AnalyzeResponse(
            width_cm=width_cm,
            height_cm=height_cm,
            aspect_ratio=aspect_ratio,
            mask_contour=contour,
            cm_per_pixel=None,
        )

    raise HTTPException(
        status_code=400,
        detail=(
            "Provide phone_box and object_prompt for phone-based measurement, "
            "or ref_box and ref_size_cm for the fallback reference workflow."
        ),
    )
