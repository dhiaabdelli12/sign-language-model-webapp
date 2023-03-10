import os
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uuid
import asyncio
import logging
from logging.handlers import RotatingFileHandler

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

@app.websocket("/videofeed")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    while True:
        try:
            frame = await websocket.receive_text()
            logger.info("Received frame")
            save_frame(frame)
        except WebSocketDisconnect as e:
            logger.error(f'WebSocket disconnected: {e}')
            break
        except asyncio.CancelledError:
            logger.error("WebSocket Task cancelled")
            break

def save_frame(frame):
    decoded = base64.b64decode(frame)
    filename = f"{str(uuid.uuid4())}.jpg"
    filepath = os.path.join("frames", filename)
    with open(filepath, "wb") as f:
        f.write(decoded)
