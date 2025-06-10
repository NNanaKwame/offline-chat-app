import asyncio
import logging
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import socketio
import uvicorn

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

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Data Structures ---
connected_users = {}  # Track connected users with their info
call_sessions = {}  # Track active call sessions

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

# --- Socket.IO Events ---
@sio.event
async def connect(sid, environ):
    """Handle a new client connection."""
    logger.info(f"Client connected: {sid}")
    # Add user to connected users list
    connected_users[sid] = {
        'id': sid,
        'username': f"User-{sid[:8]}",
        'connected_at': asyncio.get_event_loop().time(),
        'status': 'online'
    }
    # Broadcast updated user list to all clients
    await sio.emit('user_list', list(connected_users.values()))

@sio.event
async def disconnect(sid):
    """Handle a client disconnection."""
    logger.info(f"Client disconnected: {sid}")
    
    # Clean up call sessions
    if sid in call_sessions:
        call_partner = call_sessions[sid]
        if call_partner in call_sessions:
            await sio.emit('call_ended', {'reason': 'partner_disconnected'}, room=call_partner)
            call_sessions.pop(call_partner, None)
        call_sessions.pop(sid, None)
    
    # Remove from connected users
    connected_users.pop(sid, None)
    
    # Broadcast updated user list
    await sio.emit('user_list', list(connected_users.values()))

@sio.event
async def set_username(sid, username):
    """Allow users to set their username."""
    if sid in connected_users:
        connected_users[sid]['username'] = username
        await sio.emit('user_list', list(connected_users.values()))
        logger.info(f"User {sid} set username to {username}")

@sio.event
async def update_status(sid, status):
    """Update user status (online, busy, in_call)."""
    if sid in connected_users:
        connected_users[sid]['status'] = status
        await sio.emit('user_list', list(connected_users.values()))

# --- Chat Events ---
@sio.event
async def message(sid, data):
    """Handle incoming text messages and broadcast them."""
    username = connected_users.get(sid, {}).get('username', f'User-{sid[:8]}')
    logger.info(f"Message from {username} ({sid}): {data}")
    
    message_data = {
        'sid': sid,
        'username': username,
        'text': data,
        'timestamp': asyncio.get_event_loop().time()
    }
    
    await sio.emit('message', message_data)

# --- Call Management Events ---
@sio.event
async def call_request(sid, data):
    """Handle call requests between users."""
    target_sid = data.get('target_user_id')
    caller_username = connected_users.get(sid, {}).get('username', f'User-{sid[:8]}')
    
    if target_sid in connected_users:
        target_status = connected_users[target_sid].get('status', 'online')
        
        if target_status == 'in_call':
            await sio.emit('call_error', {
                'message': 'User is currently in another call'
            }, room=sid)
            return
            
        # Update caller status
        connected_users[sid]['status'] = 'calling'
        await sio.emit('user_list', list(connected_users.values()))
        
        await sio.emit('incoming_call', {
            'caller_id': sid,
            'caller_username': caller_username
        }, room=target_sid)
        logger.info(f"Call request from {sid} to {target_sid}")
    else:
        await sio.emit('call_error', {
            'message': 'User not found or offline'
        }, room=sid)

@sio.event
async def call_response(sid, data):
    """Handle call response (accept/reject)."""
    caller_id = data.get('caller_id')
    accepted = data.get('accepted', False)
    
    if caller_id in connected_users:
        if accepted:
            # Establish call session
            call_sessions[sid] = caller_id
            call_sessions[caller_id] = sid
            
            # Update both users' status
            connected_users[sid]['status'] = 'in_call'
            connected_users[caller_id]['status'] = 'in_call'
            await sio.emit('user_list', list(connected_users.values()))
            
            # Notify both parties that call is accepted
            await sio.emit('call_accepted', {
                'responder_id': sid,
                'caller_id': caller_id
            }, room=caller_id)
            
        else:
            # Reset caller status
            connected_users[caller_id]['status'] = 'online'
            await sio.emit('user_list', list(connected_users.values()))
            
        await sio.emit('call_response', {
            'responder_id': sid,
            'accepted': accepted
        }, room=caller_id)
        logger.info(f"Call response from {sid} to {caller_id}: {'accepted' if accepted else 'rejected'}")

@sio.event
async def end_call(sid):
    """Handle call ending."""
    if sid in call_sessions:
        partner_sid = call_sessions[sid]
        
        # Clean up call session
        call_sessions.pop(sid, None)
        call_sessions.pop(partner_sid, None)
        
        # Update user statuses
        if sid in connected_users:
            connected_users[sid]['status'] = 'online'
        if partner_sid in connected_users:
            connected_users[partner_sid]['status'] = 'online'
            
        # Notify partner
        await sio.emit('call_ended', {'reason': 'ended_by_partner'}, room=partner_sid)
        
        # Broadcast updated user list
        await sio.emit('user_list', list(connected_users.values()))
        
        logger.info(f"Call ended between {sid} and {partner_sid}")

# --- WebRTC Signaling Events ---
@sio.event
async def webrtc_offer(sid, data):
    """Handle WebRTC offer and forward to call partner."""
    if sid in call_sessions:
        partner_sid = call_sessions[sid]
        await sio.emit('webrtc_offer', {
            'from': sid,
            'offer': data
        }, room=partner_sid)
        logger.info(f"Forwarded offer from {sid} to {partner_sid}")

@sio.event
async def webrtc_answer(sid, data):
    """Handle WebRTC answer and forward to call partner."""
    if sid in call_sessions:
        partner_sid = call_sessions[sid]
        await sio.emit('webrtc_answer', {
            'from': sid,
            'answer': data
        }, room=partner_sid)
        logger.info(f"Forwarded answer from {sid} to {partner_sid}")

@sio.event
async def webrtc_ice_candidate(sid, data):
    """Handle ICE candidates and forward to call partner."""
    if sid in call_sessions:
        partner_sid = call_sessions[sid]
        await sio.emit('webrtc_ice_candidate', {
            'from': sid,
            'candidate': data
        }, room=partner_sid)
        logger.info(f"Forwarded ICE candidate from {sid} to {partner_sid}")

# --- Server Startup ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)