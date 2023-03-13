import os
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uuid
import asyncio
import logging
from logging.handlers import RotatingFileHandler
import cv2
import mediapipe as mp
from transformer import build_transformer_pose_model
import numpy as np

import settings as s

from extract_joints import mediapipe_detection, extract_keypoints, save_keypoints

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

app = FastAPI()

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

log_file = os.path.join(os.getcwd(), "./logs/websocket.log")
handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=10)
handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


transformer_model = build_transformer_pose_model()
transformer_model.load_weights('./models/tsl_dspl_trans.h5')

actions = np.array(['hi', 'love', 'depression'])


@app.websocket("/videofeed")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    frames = []
    while True:
        try:
            frame = await websocket.receive_text()
            if frame != '':
                frame = convert_to_array(frame)
                frames.append(frame)
            
            if len(frames) == 30:
                print(predict(frames))
                frames = []
        except WebSocketDisconnect as e:
            logger.error(f'WebSocket disconnected: {e}')
            break
        except asyncio.CancelledError:
            logger.error("WebSocket Task cancelled")
            break


def convert_to_array(frame):
    frame_bytes = base64.b64decode(frame)
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame_array = cv2.imdecode(frame_array, flags=cv2.IMREAD_COLOR)
    return frame_array


def save_frame(frame):
    decoded = base64.b64decode(frame)
    filename = f"{str(uuid.uuid4())}.jpg"
    filepath = os.path.join("frames", filename)
    with open(filepath, "wb") as f:
        f.write(decoded)


def predict(frames):
    sequence = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for frame in frames:
            _, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

    res = transformer_model.predict(np.expand_dims(sequence, axis=0))[0]
    return actions[np.argmax(res)]


