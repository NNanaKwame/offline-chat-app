import asyncio
import logging
import uuid
import os
import shutil
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import socketio
import uvicorn
import socket

# --- Basic Setup ---
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI application
app = FastAPI()

# Get local IP
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)

# Decide cert and key based on IP
if local_ip.startswith("127."):
    host = "127.0.0.1"
    cert_file = "localhost.pem"
    key_file = "localhost-key.pem"
else:
    host = local_ip
    cert_file = f"{host}.pem"
    key_file = f"{host}-key.pem"

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Socket.IO server
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio)
app.mount("/socket.io", socket_app)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="templates")

# --- Data Structures ---
connected_users = {}  # Track connected users with their info
call_sessions = {}  # Track active call sessions
uploaded_files = {}  # Track uploaded files with metadata

# --- Constants ---
MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024  # 1GB
LARGE_FILE_THRESHOLD = 400 * 1024 * 1024  # 400MB
FILE_LIFETIME_HOURS = 1  # Lifetime for files larger than the threshold
CLEANUP_INTERVAL_SECONDS = 60 * 10  # Check for files to delete every 10 minutes

# This dictionary is still useful for categorizing files on the frontend.
ALLOWED_EXTENSIONS = {
    "image": [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"],
    "document": [".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt"],
    "video": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
    "audio": [".mp3", ".wav", ".ogg", ".m4a", ".flac"],
    "archive": [".zip", ".rar", ".7z", ".tar", ".gz"],
}


def get_file_type(filename):
    """Determine file type based on extension."""
    ext = os.path.splitext(filename.lower())[1]
    for file_type, extensions in ALLOWED_EXTENSIONS.items():
        if ext in extensions:
            return file_type
    return "other"


# --- NEW: Background Cleanup Task ---
async def cleanup_large_files():
    """Periodically checks for and deletes large files older than the defined lifetime."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        logger.info("Running scheduled cleanup of large files...")

        # Make a copy of keys to avoid runtime errors during dictionary modification
        file_ids_to_check = list(uploaded_files.keys())
        now = datetime.now(timezone.utc)

        for file_id in file_ids_to_check:
            if file_id in uploaded_files:
                file_info = uploaded_files[file_id]
                is_large_file = file_info.get("size", 0) > LARGE_FILE_THRESHOLD

                if is_large_file:
                    upload_time = datetime.fromisoformat(file_info["upload_time"])
                    if (now - upload_time) > timedelta(hours=FILE_LIFETIME_HOURS):
                        try:
                            file_path = os.path.join(
                                UPLOAD_DIR, file_info["unique_name"]
                            )
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                logger.info(
                                    f"Deleted expired large file: {file_info['original_name']} ({file_id})"
                                )

                            # Remove from metadata dictionary
                            del uploaded_files[file_id]

                        except Exception as e:
                            logger.error(f"Error deleting file {file_id}: {e}")


# --- FastAPI Routes ---
@app.on_event("startup")
async def startup_event():
    """Start the background task when the server starts."""
    asyncio.create_task(cleanup_large_files())


# --- FastAPI Routes ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main landing page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    """Serve the chat page."""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/call", response_class=HTMLResponse)
async def call(request: Request):
    """Serve the video call page."""
    return templates.TemplateResponse("call.html", {"request": request})


# --- REFACTOR: Rewritten to stream large files and allow all file types ---
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), uploader_id: str = Form(...)):
    """Handles file uploads, now with warnings for large and potentially malicious files."""
    file_id = str(uuid.uuid4())
    filename = file.filename or ""
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{os.path.splitext(filename)[1]}")
    file_size = 0
    try:
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {MAX_FILE_SIZE / (1024*1024*1024):.0f}GB limit.",
                    )
                buffer.write(chunk)
    except Exception as e:
        logger.error(f"File upload failed for {filename}. Reason: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail="Error uploading file.")
    finally:
        await file.close()

    file_metadata = {
        "id": file_id,
        "original_name": filename,
        "unique_name": os.path.basename(file_path),
        "size": file_size,
        "type": get_file_type(filename),
        "mime_type": file.content_type,
        "uploader_id": uploader_id,
        "upload_time": datetime.now(timezone.utc).isoformat(),
        "url": f"/uploads/{os.path.basename(file_path)}",
    }
    uploaded_files[file_id] = file_metadata

    uploader_username = connected_users.get(uploader_id, {}).get(
        "username", f"User-{uploader_id[:8]}"
    )
    message_data = {
        "sid": uploader_id,
        "username": uploader_username,
        "type": "file",
        "file": file_metadata,
        "timestamp": asyncio.get_event_loop().time(),
    }
    await sio.emit("message", message_data)

    # --- NEW: Add Warnings ---
    # 1. Warn ALL users about downloading files
    warning_payload = {
        "message": f"'{uploader_username}' uploaded '{filename}'. Always be cautious with downloaded files.",
        "type": "warning",
    }
    await sio.emit("toast_notification", warning_payload)

    # 2. If the file is large, send a specific notice to the uploader
    if file_size > LARGE_FILE_THRESHOLD:
        notice_payload = {
            "message": f"Your file '{filename}' is over 400MB and will be deleted in 1 hour.",
            "type": "info",
        }
        await sio.emit("toast_notification", notice_payload, room=uploader_id)

    logger.info(f"File uploaded: {filename} by {uploader_username}")
    return {"success": True, "file": file_metadata}


@app.get("/download/{file_id}")
async def download_file(file_id: str):
    """Handle file downloads."""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")

    file_metadata = uploaded_files[file_id]
    file_path = os.path.join(UPLOAD_DIR, file_metadata["unique_name"])

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(
        path=file_path,
        filename=file_metadata["original_name"],
        media_type=file_metadata["mime_type"],
    )


# --- Socket.IO Events (Now with improved error handling) ---
@sio.event
async def connect(sid, environ):
    """Handle a new client connection."""
    logger.info(f"Client connected: {sid}")
    connected_users[sid] = {
        "id": sid,
        "username": f"User-{sid[:8]}",
        "connected_at": asyncio.get_event_loop().time(),
        "status": "online",
    }
    await sio.emit("user_list", list(connected_users.values()))


@sio.event
async def disconnect(sid):
    """Handle a client disconnection, ensuring call cleanup."""
    logger.info(f"Client disconnected: {sid}")

    # Gracefully end any active call the user was in
    if sid in call_sessions:
        partner_sid = call_sessions[sid]
        logger.info(f"User {sid} disconnected during a call with {partner_sid}.")
        
        # Notify the partner that the call has ended
        if partner_sid in connected_users:
            await sio.emit("call_ended", {"reason": "Your partner disconnected."}, room=partner_sid)
            connected_users[partner_sid]["status"] = "online" # Reset partner's status

        # Clean up session data for both users
        call_sessions.pop(partner_sid, None)
        call_sessions.pop(sid, None)

    # Remove user from connected list and update everyone
    connected_users.pop(sid, None)
    await sio.emit("user_list", list(connected_users.values()))


@sio.event
async def set_username(sid, username):
    """Allow users to set their username, with basic validation."""
    if not isinstance(username, str) or not (2 <= len(username) <= 20):
        logger.warning(f"User {sid} sent invalid username: {username}")
        return
        
    if sid in connected_users:
        connected_users[sid]["username"] = username
        await sio.emit("user_list", list(connected_users.values()))
        logger.info(f"User {sid} set username to {username}")


@sio.event
async def message(sid, data):
    """Handle incoming text messages and broadcast them."""
    if not isinstance(data, dict) or data.get("type") != "text":
        logger.warning(f"Received malformed message data from {sid}: {data}")
        return
        
    username = connected_users.get(sid, {}).get("username", f"User-{sid[:8]}")
    message_content = data.get("content", "")
    
    logger.info(f"Message from {username} ({sid}): {message_content}")
    message_data = {
        "sid": sid,
        "username": username,
        "type": "text",
        "text": message_content,
        "timestamp": asyncio.get_event_loop().time(),
    }
    # await sio.emit("message", message_data, skip_sid=sid)
    await sio.emit("message", message_data)


@sio.event
async def call_request(sid, data):
    """Handle a user's request to call another user."""
    if not isinstance(data, dict) or "target_user_id" not in data:
        logger.error(f"Malformed call_request from {sid}: {data}")
        return

    target_sid = data.get("target_user_id")
    caller_username = connected_users.get(sid, {}).get("username", f"User-{sid[:8]}")

    if not target_sid or target_sid not in connected_users:
        await sio.emit("call_error", {"message": "The user you are trying to call is offline."}, room=sid)
        return

    if connected_users[target_sid].get("status") != "online":
        status = connected_users[target_sid].get('status', 'unavailable')
        await sio.emit("call_error", {"message": f"User is currently busy ({status})."}, room=sid)
        return

    # Set statuses to 'Ringing' and notify clients
    connected_users[sid]["status"] = "Ringing"
    connected_users[target_sid]["status"] = "Ringing"
    await sio.emit("user_list", list(connected_users.values()))
    
    # Emit the incoming call event to the target user
    await sio.emit(
        "incoming_call",
        {"caller_id": sid, "caller_username": caller_username},
        room=target_sid,
    )


@sio.event
async def call_response(sid, data):
    """Handle the response to a call invitation (accept/reject)."""
    if not isinstance(data, dict) or "caller_id" not in data or "accepted" not in data:
        logger.error(f"Malformed call_response from {sid}: {data}")
        return
        
    caller_id = data.get("caller_id")
    accepted = data.get("accepted", False)

    if caller_id not in connected_users:
        logger.warning(f"{sid} responded to a call from a non-existent user {caller_id}")
        return

    if accepted:
        logger.info(f"Call accepted between {caller_id} and {sid}")
        # Establish the call session for both users
        call_sessions[sid] = caller_id
        call_sessions[caller_id] = sid
        connected_users[sid]["status"] = "in_call"
        connected_users[caller_id]["status"] = "in_call"
        
        # Notify the original caller to initiate the WebRTC connection
        await sio.emit("call_accepted", {"responder_id": sid}, room=caller_id)
    else:
        logger.info(f"Call rejected by {sid} for caller {caller_id}")
        # Reset status for both users if the call is rejected
        connected_users[caller_id]["status"] = "online"
        connected_users[sid]["status"] = "online"
        await sio.emit("call_rejected", {"responder_username": connected_users[sid]['username']}, room=caller_id)
    
    # Update user lists for all clients
    await sio.emit("user_list", list(connected_users.values()))


@sio.event
async def end_call(sid):
    """Handle call ending initiated by one of the participants."""
    if sid in call_sessions:
        partner_sid = call_sessions[sid]
        logger.info(f"Call ended by {sid}. Notifying partner {partner_sid}")

        # Notify the other user that the call has ended
        await sio.emit("call_ended", {"reason": "The call was ended by your partner."}, room=partner_sid)

        # Reset statuses
        if sid in connected_users:
            connected_users[sid]["status"] = "online"
        if partner_sid in connected_users:
            connected_users[partner_sid]["status"] = "online"

        # Clean up session for both users
        call_sessions.pop(sid, None)
        call_sessions.pop(partner_sid, None)
        
        # Update everyone's user list
        await sio.emit("user_list", list(connected_users.values()))


# --- WebRTC Signaling Events ---

async def forward_webrtc_event(sid, event_name, data):
    """Generic handler to forward WebRTC data to the call partner."""
    if sid not in call_sessions:
        logger.warning(f"{event_name} received from {sid} who is not in a call.")
        return
        
    partner_sid = call_sessions[sid]
    logger.info(f"Forwarding {event_name} from {sid} to {partner_sid}")
    await sio.emit(event_name, data, room=partner_sid)

@sio.event
async def webrtc_offer(sid, data):
    await forward_webrtc_event(sid, "webrtc_offer", data)

@sio.event
async def webrtc_answer(sid, data):
    await forward_webrtc_event(sid, "webrtc_answer", data)

@sio.event
async def webrtc_ice_candidate(sid, data):
    await forward_webrtc_event(sid, "webrtc_ice_candidate", data)


# --- Server Startup ---
if __name__ == "__main__":
    if os.path.exists(cert_file) and os.path.exists(key_file):
        uvicorn.run(
            app, host=host, port=8000, ssl_certfile=cert_file, ssl_keyfile=key_file
        )
    else:
        logger.error(
            f"Missing SSL files: {cert_file}, {key_file}. Please generate them using openssl."
        )

# --- Server Startup --- (Modified for ngrok Development)
# if __name__ == "__main__":
#     # We will run on plain HTTP and let ngrok handle the HTTPS layer.
#     # This is a more stable setup for local development.
#     uvicorn.run(
#         app, host="127.0.0.1", port=8000
#     )