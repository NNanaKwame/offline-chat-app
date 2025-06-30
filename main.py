import asyncio
import logging
import uuid
import os
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import socketio
import uvicorn
import socket
from urllib.parse import parse_qs
import sqlite3
from passlib.context import CryptContext
from pydantic import BaseModel


# --- Pydantic Models for Request Bodies ---
class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str

class MessageDelete(BaseModel):
    message_id: str
    chat_id: str
    delete_for_everyone: bool = False  # True = delete for everyone, False = delete for me only


# --- Password Hashing Setup ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Database Setup ---
DB_NAME = "uniconnect.db"


def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    # Create users table with username as the PRIMARY KEY
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    # We can add the messages table here as well
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY NOT NULL,
            chat_id TEXT NOT NULL,
            sender_username TEXT NOT NULL,
            text_content TEXT,
            file_url TEXT,
            file_mime_type TEXT,
            timestamp TEXT NOT NULL,
            is_sent INTEGER NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()


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


# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up application...")
    init_db()
    cleanup_task = asyncio.create_task(cleanup_large_files())

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down application...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        logger.info("Cleanup task cancelled successfully")


# --- Basic Setup ---
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI application with lifespan handler
app = FastAPI(lifespan=lifespan)

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
sid_to_username = {}
username_to_sid = {}  # Maps session IDs to usernames for easy lookup

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


@app.post("/register")
async def register_user(user: UserCreate):
    conn = get_db_connection()
    try:
        # Check if username or email already exists
        existing_user = conn.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            (user.username, user.email),
        ).fetchone()
        if existing_user:
            raise HTTPException(
                status_code=400, detail="Username or email already registered."
            )

        hashed_password = pwd_context.hash(user.password)
        conn.execute(
            "INSERT INTO users (username, email, hashed_password) VALUES (?, ?, ?)",
            (user.username, user.email, hashed_password),
        )
        conn.commit()
        return {"message": "User registered successfully"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username or email already exists.")
    finally:
        conn.close()


@app.post("/login")
async def login_user(user: UserLogin):
    conn = get_db_connection()
    db_user = conn.execute(
        "SELECT * FROM users WHERE username = ?", (user.username,)
    ).fetchone()
    conn.close()

    if not db_user:
        raise HTTPException(status_code=404, detail="User not found.")

    if not pwd_context.verify(user.password, db_user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect password.")

    # In a real app, you would generate and return a JWT token here.
    # For now, we'll just confirm success.
    return {"message": "Login successful", "username": user.username}


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    uploader_id: str = Form(...),
    recipient_username: str = Form(...)
):
    # Add debugging logs
    logger.info(f"=== UPLOAD DEBUG ===")
    logger.info(f"Uploader ID received: {uploader_id}")
    logger.info(f"Recipient username: {recipient_username}")
    logger.info(f"Current sid_to_username mapping: {sid_to_username}")
    logger.info(f"Current connected_users: {list(connected_users.keys())}")
    
    sender_username = sid_to_username.get(uploader_id)
    logger.info(f"Sender username found: {sender_username}")
    
    if not sender_username:
        logger.error(f"Authentication failed - no username found for SID: {uploader_id}")
        logger.error(f"Available SIDs: {list(sid_to_username.keys())}")
        raise HTTPException(status_code=403, detail="Uploader not authenticated.")

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

    message_data = {
        "sid": uploader_id,
        "username": sender_username,
        "type": "file",
        "file": file_metadata,
        "timestamp": asyncio.get_event_loop().time(),
        "recipient_username": recipient_username,
    }
    
    # Send to recipient if they're online
    recipient_sid = username_to_sid.get(recipient_username)
    if recipient_sid:
        await sio.emit("message", message_data, room=recipient_sid)
        logger.info(f"Sent file '{file.filename}' from {sender_username} to {recipient_username}")
    
    # ALWAYS send back to sender so they can see their own message
    await sio.emit("message", message_data, room=uploader_id)
    logger.info(f"Sent file confirmation to sender {sender_username}")

    logger.info(f"File uploaded: {filename} by {sender_username}")
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


@app.post("/delete-message")
async def delete_message(delete_request: MessageDelete):
    try:
        conn = get_db_connection()
        
        if delete_request.delete_for_everyone:
            # Delete the message completely from database
            conn.execute(
                "DELETE FROM messages WHERE id = ? AND chat_id = ?",
                (delete_request.message_id, delete_request.chat_id)
            )
        else:
            # For "delete for me" functionality, you might want to add a deleted_by column
            # For now, we'll just delete it completely
            # In production, you'd add a deleted_by field to track who deleted it
            conn.execute(
                "DELETE FROM messages WHERE id = ? AND chat_id = ?",
                (delete_request.message_id, delete_request.chat_id)
            )
        
        conn.commit()
        conn.close()
        
        return {"success": True, "message": "Message deleted successfully"}
    
    except Exception as e:
        logger.error(f"Error deleting message: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete message")

# --- Socket.IO Events (Now with improved error handling) ---
@sio.event
async def connect(sid, environ, auth):
    """Handle a client connection after they have logged in."""
    try:
        logger.info(f"=== CONNECT ATTEMPT ===")
        logger.info(f"SID: {sid}")
        logger.info(f"Auth: {auth}")

        username = None

        # Method 1: Try auth parameter (Socket.IO v4+ way)
        if auth and isinstance(auth, dict):
            username = auth.get("username")
            logger.info(f"Username from auth: {username}")

        # Method 2: Try query string from environ
        if not username:
            query_string = environ.get("QUERY_STRING", b"")
            if isinstance(query_string, bytes):
                query_string = query_string.decode()
            elif query_string is None:
                query_string = ""

            logger.info(f"Query string: {query_string}")

            if query_string:
                params = parse_qs(query_string)
                username = params.get("username", [None])[0]
                if not username:
                    username = params.get("userId", [None])[0]
                logger.info(f"Username from query: {username}")

        if not username:
            logger.error("No username found in any method")
            return False

        logger.info(f"Final username: {username}")

        # Add user to our connected list
        connected_users[username] = {
            "username": username,
            "sid": sid,
            "status": "online",
        }
        
        # Map both directions correctly
        sid_to_username[sid] = username
        username_to_sid[username] = sid
        
        logger.info(f"Added user mapping: SID {sid} -> Username {username}")
        logger.info(f"Current connected users: {list(connected_users.keys())}")

        # Broadcast the updated list of online users to ALL connected clients
        online_users_list = [
            {"username": u["username"], "status": u["status"]}
            for u in connected_users.values()
        ]
        
        logger.info(f"Broadcasting user list to all clients: {online_users_list}")
        await sio.emit("user_list", online_users_list)  # Broadcast to all
        
        logger.info(f"=== END CONNECT ===")
        return True

    except Exception as e:
        logger.error(f"Connection from {sid} failed with exception: {str(e)}")
        return False


# Also update the disconnect handler to broadcast user list
@sio.event
async def disconnect(sid):
    """Handle a client disconnection, ensuring call cleanup."""
    
    # Get username before cleanup
    userId = sid_to_username.get(sid)
    logger.info(f"Client disconnected: {sid} (userId: {userId})")

    # Gracefully end any active call the user was in
    if sid in call_sessions:
        partner_sid = call_sessions[sid]
        logger.info(f"User {sid} disconnected during a call with {partner_sid}.")

        # Notify the partner that the call has ended
        if partner_sid in connected_users:
            await sio.emit(
                "call_ended", {"reason": "Your partner disconnected."}, room=partner_sid
            )
            # Reset partner's status
            partner_username = sid_to_username.get(partner_sid)
            if partner_username and partner_username in connected_users:
                connected_users[partner_username]["status"] = "online"

        # Clean up session data for both users
        call_sessions.pop(partner_sid, None)
        call_sessions.pop(sid, None)

    # Clean up user mappings
    if userId:
        connected_users.pop(userId, None)
        sid_to_username.pop(sid, None)
        username_to_sid.pop(userId, None)

    # Broadcast updated user list to all remaining clients
    online_users_list = [
        {"username": u["username"], "status": u["status"]}
        for u in connected_users.values()
    ]
    
    logger.info(f"Broadcasting updated user list after disconnect: {online_users_list}")
    await sio.emit("user_list", online_users_list)  # Broadcast to all


@sio.event
async def set_username(sid, username):
    """Allow users to set their username, linked to their persistent ID."""
    userId = sid_to_username.get(sid)
    if not userId or userId not in connected_users:
        logger.warning(f"set_username from unknown session: {sid}")
        return

    if isinstance(username, str) and 2 <= len(username) <= 20:
        connected_users[userId]["username"] = username
        await sio.emit("user_list", list(connected_users.values()))
        logger.info(f"User {userId} set username to {username}")


@sio.event
async def message(sid, data):
    """Handle incoming messages and send to a specific recipient."""
    sender_username = sid_to_username.get(sid)
    if not sender_username:
        return

    # We expect the frontend to tell us who the recipient is
    recipient_username = data.get("recipient_username")
    if not recipient_username:
        return  # Or handle error

    recipient_sid = username_to_sid.get(recipient_username)

    message_data = {
        "sid": sid,
        "username": sender_username,
        "type": data.get("type", "text"),
        "text": data.get("content"),
        "file": data.get("file"),
        "timestamp": asyncio.get_event_loop().time(),
    }

    # If the recipient is online, send it to them directly
    if recipient_sid:
        await sio.emit("message", message_data, room=recipient_sid)

    # Always send the message back to the sender so they see it
    await sio.emit("message", message_data, room=sid)


@sio.event
async def call_request(sid, data):
    """Handle a user's request to call another user."""
    if not isinstance(data, dict) or "target_user_id" not in data:
        logger.error(f"Malformed call_request from {sid}: {data}")
        return

    target_sid = data.get("target_user_id")
    caller_username = connected_users.get(sid, {}).get("username", f"User-{sid[:8]}")

    if not target_sid or target_sid not in connected_users:
        await sio.emit(
            "call_error",
            {"message": "The user you are trying to call is offline."},
            room=sid,
        )
        return

    if connected_users[target_sid].get("status") != "online":
        status = connected_users[target_sid].get("status", "unavailable")
        await sio.emit(
            "call_error", {"message": f"User is currently busy ({status})."}, room=sid
        )
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
        logger.warning(
            f"{sid} responded to a call from a non-existent user {caller_id}"
        )
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
        await sio.emit(
            "call_rejected",
            {"responder_username": connected_users[sid]["username"]},
            room=caller_id,
        )

    # Update user lists for all clients
    await sio.emit("user_list", list(connected_users.values()))


@sio.event
async def end_call(sid):
    """Handle call ending initiated by one of the participants."""
    if sid in call_sessions:
        partner_sid = call_sessions[sid]
        logger.info(f"Call ended by {sid}. Notifying partner {partner_sid}")

        # Notify the other user that the call has ended
        await sio.emit(
            "call_ended",
            {"reason": "The call was ended by your partner."},
            room=partner_sid,
        )

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


@sio.event
async def delete_message_event(sid, data):
    """Handle real-time message deletion."""
    sender_username = sid_to_username.get(sid)
    if not sender_username:
        return

    message_id = data.get("message_id")
    chat_id = data.get("chat_id")
    delete_for_everyone = data.get("delete_for_everyone", False)
    recipient_username = data.get("recipient_username")

    if not message_id or not chat_id:
        return

    # Delete from database
    try:
        conn = get_db_connection()
        
        if delete_for_everyone:
            conn.execute(
                "DELETE FROM messages WHERE id = ? AND chat_id = ?",
                (message_id, chat_id)
            )
        else:
            # For now, just delete completely
            conn.execute(
                "DELETE FROM messages WHERE id = ? AND chat_id = ?", 
                (message_id, chat_id)
            )
        
        conn.commit()
        conn.close()

        # Emit deletion event to both users
        deletion_data = {
            "message_id": message_id,
            "chat_id": chat_id,
            "delete_for_everyone": delete_for_everyone,
            "deleted_by": sender_username
        }

        # Send to sender
        await sio.emit("message_deleted", deletion_data, room=sid)

        # Send to recipient if online
        if recipient_username:
            recipient_sid = username_to_sid.get(recipient_username)
            if recipient_sid:
                await sio.emit("message_deleted", deletion_data, room=recipient_sid)

        logger.info(f"Message {message_id} deleted by {sender_username}")

    except Exception as e:
        logger.error(f"Error in delete_message_event: {e}")
        await sio.emit("delete_error", {"error": "Failed to delete message"}, room=sid)

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


# Typing Starts
@sio.event
async def typing_start(sid, data):
    """Broadcast when a user starts typing to a specific recipient."""
    sender_username = sid_to_username.get(sid)
    recipient_username = data.get("recipient_username")

    if not sender_username or not recipient_username:
        logger.warning(f"typing_start: Missing sender ({sender_username}) or recipient ({recipient_username})")
        return

    recipient_sid = username_to_sid.get(recipient_username)
    if recipient_sid:
        logger.info(f"User {sender_username} started typing to {recipient_username}.")
        # Emit only to the recipient's room (their socket ID)
        await sio.emit(
            "typing_start",
            {"username": sender_username},
            room=recipient_sid,
        )
    else:
        logger.info(f"Recipient {recipient_username} is not online")


# Replace the existing typing_stop event  
@sio.event
async def typing_stop(sid, data):
    """Broadcast when a user stops typing to a specific recipient."""
    sender_username = sid_to_username.get(sid)
    recipient_username = data.get("recipient_username")

    if not sender_username or not recipient_username:
        logger.warning(f"typing_stop: Missing sender ({sender_username}) or recipient ({recipient_username})")
        return

    recipient_sid = username_to_sid.get(recipient_username)
    if recipient_sid:
        logger.info(f"User {sender_username} stopped typing to {recipient_username}.")
        await sio.emit(
            "typing_stop",
            {"username": sender_username},
            room=recipient_sid,
        )
    else:
        logger.info(f"Recipient {recipient_username} is not online")


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
