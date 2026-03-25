from __future__ import annotations

import base64
import binascii
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


class RefBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class AnalyzeRequest(BaseModel):
    image_data: str = Field(..., description="Base64 data URL for the captured image")
    ref_box: RefBox
    ref_size_cm: float = Field(..., gt=0)


class AnalyzeResponse(BaseModel):
    width_cm: float
    height_cm: float
    aspect_ratio: float
    mask_contour: list[list[float]]


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


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    _validate_image_data(payload.image_data)

    # Replace this placeholder with your real YOLO/SAM pipeline.
    # This keeps the contract working for the frontend and Railway deployment.
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
    )
