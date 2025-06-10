# main.py
import asyncio
import logging
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import socketio
import uvicorn
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRelay

# --- Basic Setup ---
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI application
app = FastAPI()

# Socket.IO server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio)
app.mount("/socket.io", socket_app)

# Jinja2 for HTML templating
templates = Jinja2Templates(directory="templates")

# --- Data Structures ---
# Store peer connections and other session data
pcs = set()
sid_to_pc = {}

# --- FastAPI Routes ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

# --- Socket.IO Events for Signaling ---
@sio.event
async def connect(sid, environ):
    """Handle a new client connection."""
    logger.info(f"Client connected: {sid}")
@sio.event
async def disconnect(sid):
    """Handle a client disconnection."""
    logger.info(f"Client disconnected: {sid}")
    # Clean up peer connection associated with the disconnected client
    pc = sid_to_pc.pop(sid, None)
    if pc:
        await pc.close()
        pcs.discard(pc)
    #     pcs.discard(pc)


@sio.event
async def offer(sid, data):
    """Handle an offer from a peer."""
    logger.info(f"Received offer from {sid}")
    
    offer_desc = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
    pc = RTCPeerConnection()
    pc = RTCPeerConnection()
    # Store the mapping from sid to pc for cleanup later
    sid_to_pc[sid] = pc
    pcs.add(pc)
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state is {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track {track.kind} received")
        # In a real app, you would handle the incoming track.
        # For a simple echo, you could add it back to the connection.
        # For now, we just log it.

    await pc.setRemoteDescription(offer_desc)
    answer_desc = await pc.createAnswer()
    await pc.setLocalDescription(answer_desc)

    await sio.emit('answer', {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }, room=sid)
    logger.info(f"Sent answer to {sid}")


@sio.event
async def message(sid, data):
    """Handle incoming text messages and broadcast them."""
    logger.info(f"Message from {sid}: {data}")
    await sio.emit('message', {'sid': sid, 'text': data})


# --- Server Startup ---
if __name__ == "__main__":
    # Note: Running this directly is for debugging.
    # Production deployment should use: uvicorn main:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
