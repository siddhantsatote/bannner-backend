
import json
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from detector import detect_banner



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





app = FastAPI(title="Banner Buddy API", version="2.0.0")

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

    prompt = _normalize_object_prompt(object_prompt)

    # Support comma-separated multi-label prompt for broader zero-shot detection.
    candidate_labels = [p.strip() for p in re.split(r"[,\n]", prompt) if p.strip()]
    if not candidate_labels:
        raise HTTPException(status_code=400, detail="object_prompt must contain at least one valid label")

    image_bytes = _decode_image_bytes(image_data)
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    # Keep original MIME if included (typically data:image/jpeg/png;base64,...)
    if "," in image_data and image_data.startswith("data:"):
        mime_prefix = image_data.split(",", 1)[0]
        image_input = f"{mime_prefix},{encoded_image}"
    else:
        image_input = f"data:image/jpeg;base64,{encoded_image}"

    # Use HF inference pipeline endpoint for zero-shot object detection.
    # https://huggingface.co/docs/api-inference/using_the_api/pipelines
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


    import traceback
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        normalized = body.strip() or exc.reason
        print(f"[HF ERROR] HTTPError: {exc.code} {normalized}")
        if exc.code in (404, 422):
            raise HTTPException(
                status_code=502,
                detail={
                    "error": f"Hugging Face detection failed: {normalized}",
                    "model": HF_OBJECT_MODEL,
                    "prompt": object_prompt,
                    "hf_url": url,
                    "hf_body": request_body.decode("utf-8"),
                },
            ) from exc
        raise HTTPException(
            status_code=502,
            detail={
                "error": f"Hugging Face detection failed: {normalized}",
                "model": HF_OBJECT_MODEL,
                "prompt": object_prompt,
                "hf_url": url,
                "hf_body": request_body.decode("utf-8"),
            },
        ) from exc
    except urllib.error.URLError as exc:
        print(f"[HF ERROR] URLError: {exc.reason}")
        raise HTTPException(status_code=502, detail={
            "error": f"Hugging Face detection unreachable: {exc.reason}",
            "model": HF_OBJECT_MODEL,
            "prompt": object_prompt,
            "hf_url": url,
            "hf_body": request_body.decode("utf-8"),
        }) from exc
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[HF ERROR] Exception: {tb}")
        raise HTTPException(status_code=502, detail={
            "error": f"Unexpected backend error: {str(exc)}",
            "traceback": tb,
            "model": HF_OBJECT_MODEL,
            "prompt": object_prompt,
            "hf_url": url,
            "hf_body": request_body.decode("utf-8"),
        }) from exc

    if isinstance(payload, dict) and payload.get("error"):
        print(f"[HF ERROR] API error: {payload['error']}")
        raise HTTPException(status_code=502, detail={
            "error": f"Hugging Face error: {payload['error']}",
            "model": HF_OBJECT_MODEL,
            "prompt": object_prompt,
            "hf_url": url,
            "hf_body": request_body.decode("utf-8"),
        })

    detections = payload if isinstance(payload, list) else []

    if not detections:
        print("[HF ERROR] No object detected in the image")
        raise HTTPException(status_code=404, detail={
            "error": "No object detected in the image",
            "model": HF_OBJECT_MODEL,
            "prompt": object_prompt,
            "hf_url": url,
            "hf_body": request_body.decode("utf-8"),
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



# New analyze endpoint for local model inference
@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    image: UploadFile = File(...),
    ref_box: str = Form(...),
    ref_size_cm: float = Form(...)
):
    try:
        image_bytes = await image.read()
        ref_box_dict = json.loads(ref_box)

        # Step 1 — OmDet-Turbo detects banner
        detection = detect_banner(image_bytes, ref_box_dict)

        # Step 2 — (Optional) Add segmentation, PCA, and real-world size logic here
        # For now, just return the detected bounding box and label
        bbox = detection["bbox"]
        centroid = detection["centroid"]
        label = detection["label"]
        score = detection["score"]

        # Dummy values for mask_contour and cm_per_pixel (replace with real logic as needed)
        mask_contour = [
            [bbox["x"], bbox["y"]],
            [bbox["x"] + bbox["w"], bbox["y"]],
            [bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]],
            [bbox["x"], bbox["y"] + bbox["h"]],
        ]
        width_cm = ref_size_cm  # Placeholder: replace with real calculation
        height_cm = ref_size_cm  # Placeholder: replace with real calculation
        aspect_ratio = round(width_cm / max(height_cm, 1e-6), 2)

        return AnalyzeResponse(
            width_cm=width_cm,
            height_cm=height_cm,
            aspect_ratio=aspect_ratio,
            mask_contour=mask_contour,
            cm_per_pixel=None,
            detected_label=label,
            detected_score=score,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
