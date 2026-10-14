import asyncio
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


app = FastAPI()

PROJECT_FOLDER = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_FOLDER / "models" / "face_landmarker.task"

if not MODEL_PATH.is_file():
    raise RuntimeError("Missing models/face_landmarker.task")


@app.get("/")
def home():
    return FileResponse(PROJECT_FOLDER / "index.html")


def create_tracker():
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )

    return vision.FaceLandmarker.create_from_options(options)


def apply_anime_filter(frame, landmarks):
    height, width = frame.shape[:2]

    # Convert facial landmarks into pixel coordinates.
    points = np.array(
        [
            (int(point.x * width), int(point.y * height))
            for point in landmarks
        ],
        dtype=np.int32,
    )

    # Create a mask covering the face.
    mask = np.zeros((height, width), dtype=np.uint8)
    face_outline = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, face_outline, 255)

    # Soften the mask boundary for a gradual transition.
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]

    # Smooth skin texture while preserving important edges.
    smooth = cv2.bilateralFilter(frame, 9, 65, 65)
    smooth_float = smooth.astype(np.float32)

    # Slightly simplify the colors for illustrated shading.
    simplified = np.round(smooth_float / 16.0) * 16.0
    styled = 0.65 * smooth_float + 0.35 * simplified

    # Detect and gently darken outlines.
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 140)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)

    ink = (edges.astype(np.float32) / 255.0)[..., None]
    styled *= 1.0 - 0.35 * ink

    # Preserve some original detail.
    original = frame.astype(np.float32)
    styled = 0.8 * styled + 0.2 * original

    # Apply the effect within the softened face mask.
    result = original * (1.0 - alpha) + styled * alpha

    return np.clip(result, 0, 255).astype(np.uint8)


def enlarge_eyes(frame, landmarks, strength=0.18):
    height, width = frame.shape[:2]
    output = frame.copy()

    # Keep the distortion within a gentle range.
    strength = float(np.clip(strength, 0.0, 0.35))

    # Inner and outer corners of each eye.
    eye_pairs = [
        (33, 133),
        (362, 263),
    ]

    for first_index, second_index in eye_pairs:
        first = landmarks[first_index]
        second = landmarks[second_index]

        # Convert landmarks into pixel coordinates.
        first_x = first.x * width
        first_y = first.y * height
        second_x = second.x * width
        second_y = second.y * height

        center_x = (first_x + second_x) / 2
        center_y = (first_y + second_y) / 2

        eye_width = np.hypot(
            second_x - first_x,
            second_y - first_y,
        )

        # Skip very small eyes when the face is far away.
        if eye_width < 6:
            continue

        radius = eye_width * 0.9

        # Select a small image region around this eye.
        x_start = max(0, int(np.floor(center_x - radius)))
        y_start = max(0, int(np.floor(center_y - radius)))
        x_end = min(width, int(np.ceil(center_x + radius)) + 1)
        y_end = min(height, int(np.ceil(center_y + radius)) + 1)

        if x_end <= x_start or y_end <= y_start:
            continue

        region = output[y_start:y_end, x_start:x_end].copy()

        # Eye center relative to this cropped region.
        local_x = center_x - x_start
        local_y = center_y - y_start

        # Build coordinates for every pixel in the region.
        grid_y, grid_x = np.mgrid[
            0:region.shape[0],
            0:region.shape[1],
        ].astype(np.float32)

        dx = grid_x - local_x
        dy = grid_y - local_y

        # Normalized distance from the eye center.
        distance_squared = (dx * dx + dy * dy) / (radius * radius)

        # Strongest at the center, fading smoothly to zero.
        influence = np.clip(1.0 - distance_squared, 0.0, 1.0)
        scale = 1.0 - strength * influence * influence

        # Read pixels closer to the center to create enlargement.
        map_x = (local_x + dx * scale).astype(np.float32)
        map_y = (local_y + dy * scale).astype(np.float32)

        enlarged = cv2.remap(
            region,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        output[y_start:y_end, x_start:x_end] = enlarged

    return output

def process_frame(image_bytes, tracker, timestamp_ms, anime_enabled):
    # Decode the JPEG received from the browser.
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Could not read the camera frame.")

    if anime_enabled:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = tracker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            # First apply the illustrated shading.
            frame = apply_anime_filter(frame, landmarks)

            # Then gently enlarge both eyes.
            frame = enlarge_eyes(
                frame,
                landmarks,
                strength=0.18,
            )

    # If anime mode is off, or no face is found,
    # return the original frame.
    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )

    if not success:
        raise ValueError("Could not encode the processed frame.")

    return encoded.tobytes()


@app.websocket("/ws")
async def camera_socket(websocket: WebSocket):
    await websocket.accept()

    tracker = await asyncio.to_thread(create_tracker)
    last_timestamp = -1

    try:
        while True:
            packet = await websocket.receive_bytes()

            # First byte is the mode; remaining bytes are the JPEG.
            if len(packet) < 2 or packet[0] not in (0, 1):
                await websocket.close(code=1003)
                return

            anime_enabled = packet[0] == 1
            image_bytes = packet[1:]

            timestamp_ms = max(
                time.monotonic_ns() // 1_000_000,
                last_timestamp + 1,
            )
            last_timestamp = timestamp_ms

            result = await asyncio.to_thread(
                process_frame,
                image_bytes,
                tracker,
                timestamp_ms,
                anime_enabled,
            )

            await websocket.send_bytes(result)

    except WebSocketDisconnect:
        pass

    finally:
        await asyncio.to_thread(tracker.close)