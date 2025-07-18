import asyncio
import logging
import uuid
import os
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Form, Query
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
from typing import List, Optional


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
    delete_for_everyone: bool = False


class GroupCreate(BaseModel):
    name: str
    description: Optional[str] = None
    members: List[str]


class GroupUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class AddMemberRequest(BaseModel):
    group_id: int
    username: str


class RemoveMemberRequest(BaseModel):
    group_id: int
    username: str


class GroupInfoResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    creator_username: str
    created_at: str
    memberCount: int
    isAdmin: bool
    members: List[dict]


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
    
    # Users table
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
    
    # Messages table for direct chats
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY NOT NULL,
            chat_id TEXT NOT NULL,
            sender_username TEXT NOT NULL,
            text_content TEXT,
            file_url TEXT,
            file_mime_type TEXT,
            file_size INTEGER DEFAULT 0,
            file_id TEXT,
            timestamp TEXT NOT NULL,
            is_sent INTEGER NOT NULL
        )
        """
    )
    
    # Groups table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            creator_username TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (creator_username) REFERENCES users (username)
        )
        """
    )
    
    # Group Members table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS group_members (
            group_id INTEGER NOT NULL,
            username TEXT NOT NULL,
            role TEXT DEFAULT 'member',
            joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (group_id, username),
            FOREIGN KEY (group_id) REFERENCES groups (id) ON DELETE CASCADE,
            FOREIGN KEY (username) REFERENCES users (username)
        )
        """
    )
    
    # Group Messages table - FIXED with all required columns
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS group_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id INTEGER NOT NULL,
            sender_username TEXT NOT NULL,
            content TEXT,
            message_type TEXT DEFAULT 'text',
            file_url TEXT,
            file_mime_type TEXT,
            file_size INTEGER DEFAULT 0,
            file_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (group_id) REFERENCES groups (id) ON DELETE CASCADE,
            FOREIGN KEY (sender_username) REFERENCES users (username)
        )
        """
    )
    
    conn.commit()
    conn.close()


# --- Background Cleanup Task ---
async def cleanup_large_files():
    """Periodically checks for and deletes large files older than the defined lifetime."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        logger.info("Running scheduled cleanup of large files...")

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
                            file_path = os.path.join(UPLOAD_DIR, file_info["unique_name"])
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                logger.info(f"Deleted expired large file: {file_info['original_name']} ({file_id})")
                            del uploaded_files[file_id]
                        except Exception as e:
                            logger.error(f"Error deleting file {file_id}: {e}")


# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up application...")
    init_db()  # This will now include migration
    cleanup_task = asyncio.create_task(cleanup_large_files())
    yield
    logger.info("Shutting down application...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        logger.info("Cleanup task cancelled successfully")


# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(lifespan=lifespan)

hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)

if local_ip.startswith("127."):
    host = "127.0.0.1"
    cert_file = "localhost.pem"
    key_file = "localhost-key.pem"
else:
    host = local_ip
    cert_file = f"{host}.pem"
    key_file = f"{host}-key.pem"

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Socket.IO server
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio)
app.mount("/socket.io", socket_app)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="templates")

# --- Data Structures ---
connected_users = {}
call_sessions = {}
uploaded_files = {}
sid_to_username = {}
username_to_sid = {}

# Broadcast control
_broadcasting_user_list = False
_broadcast_lock = asyncio.Lock()

# --- Constants ---
MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024  # 1GB
LARGE_FILE_THRESHOLD = 400 * 1024 * 1024  # 400MB
FILE_LIFETIME_HOURS = 1
CLEANUP_INTERVAL_SECONDS = 60 * 10

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


# --- Helper Functions ---
async def broadcast_user_list():
    """Broadcast the current user list to all connected clients."""
    global _broadcasting_user_list
    
    async with _broadcast_lock:
        if _broadcasting_user_list:
            logger.debug("Broadcast already in progress, skipping")
            return
            
        _broadcasting_user_list = True
        
        try:
            online_users_list = [
                {"username": user["username"], "status": user["status"]}
                for user in connected_users.values()
            ]
            
            logger.info(f"Broadcasting user list: {len(online_users_list)} users online")
            await sio.emit("user_list", online_users_list)
            
        except Exception as e:
            logger.error(f"Error broadcasting user list: {e}")
        finally:
            _broadcasting_user_list = False


# --- API Routes ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/call", response_class=HTMLResponse)
async def call(request: Request):
    return templates.TemplateResponse("call.html", {"request": request})


@app.post("/register")
async def register_user(user: UserCreate):
    conn = get_db_connection()
    try:
        existing_user = conn.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            (user.username, user.email),
        ).fetchone()
        if existing_user:
            raise HTTPException(status_code=400, detail="Username or email already registered.")

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

    return {"message": "Login successful", "username": user.username}


# --- GROUP ENDPOINTS ---
@app.post("/groups", response_model=dict)
async def create_group_endpoint(group_data: GroupCreate):
    """Create a new group with the specified members."""
    conn = get_db_connection()
    try:
        if not group_data.name or not group_data.name.strip():
            raise HTTPException(status_code=400, detail="Group name is required.")
        
        if not group_data.members or len(group_data.members) == 0:
            raise HTTPException(status_code=400, detail="Group must have at least one member.")
        
        creator_username = group_data.members[0]
        
        # Validate creator exists
        creator_exists = conn.execute(
            "SELECT username FROM users WHERE username = ?", (creator_username,)
        ).fetchone()
        if not creator_exists:
            raise HTTPException(status_code=403, detail=f"Creator username '{creator_username}' is not registered.")

        # Check if group name already exists
        existing_group = conn.execute(
            "SELECT id FROM groups WHERE name = ?", (group_data.name.strip(),)
        ).fetchone()
        if existing_group:
            raise HTTPException(status_code=400, detail="Group name already exists.")

        # Create group
        cursor = conn.execute(
            "INSERT INTO groups (name, description, creator_username) VALUES (?, ?, ?)",
            (group_data.name.strip(), group_data.description or "", creator_username),
        )
        group_id = cursor.lastrowid

        if not group_id:
            raise HTTPException(status_code=500, detail="Failed to create group.")

        # Add members to group
        added_members = []
        for member_username in group_data.members:
            if not member_username or not member_username.strip():
                continue
                
            user_exists = conn.execute(
                "SELECT username FROM users WHERE username = ?", (member_username.strip(),)
            ).fetchone()
            
            if not user_exists:
                logger.warning(f"User '{member_username}' not found, skipping.")
                continue

            role = 'admin' if member_username == creator_username else 'member'
            
            try:
                conn.execute(
                    "INSERT INTO group_members (group_id, username, role) VALUES (?, ?, ?)",
                    (group_id, member_username.strip(), role),
                )
                added_members.append(member_username.strip())
                
                # Add to Socket.IO room if online
                member_sid = username_to_sid.get(member_username.strip())
                if member_sid:
                    await sio.enter_room(member_sid, str(group_id))
                    logger.info(f"Added {member_username} to group {group_id} room")
                    
            except sqlite3.IntegrityError:
                logger.warning(f"User '{member_username}' already in group {group_id}")
                continue

        conn.commit()
        
        if not added_members:
            conn.execute("DELETE FROM groups WHERE id = ?", (group_id,))
            conn.commit()
            raise HTTPException(status_code=400, detail="No valid members found. Group creation failed.")

        group_info = {
            "id": group_id,
            "name": group_data.name.strip(),
            "description": group_data.description or "",
            "creator_username": creator_username,
            "members": added_members,
            "member_count": len(added_members)
        }

        # Notify members via Socket.IO
        notification_data = {
            "type": "group_created",
            "group": group_info,
            "message": f"You've been added to group '{group_data.name}'"
        }
        
        for member_username in added_members:
            member_sid = username_to_sid.get(member_username)
            if member_sid:
                await sio.emit("group_notification", notification_data, room=member_sid)

        logger.info(f"Group '{group_data.name}' ({group_id}) created by {creator_username} with {len(added_members)} members")

        return {"message": "Group created successfully", "group_id": group_id, "group": group_info}

    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating group: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create group: {str(e)}")
    finally:
        conn.close()


@app.get("/my-groups", response_model=List[GroupInfoResponse])
async def get_my_groups(username: str = Query(...)):
    """Get all groups that a user is a member of."""
    conn = get_db_connection()
    try:
        groups_data = []
        
        member_groups = conn.execute(
            """
            SELECT 
                g.id, 
                g.name, 
                g.description, 
                g.creator_username, 
                g.created_at,
                (SELECT COUNT(*) FROM group_members gm2 WHERE gm2.group_id = g.id) AS memberCount,
                (SELECT gm_role.role FROM group_members gm_role WHERE gm_role.group_id = g.id AND gm_role.username = ?) AS userRole
            FROM groups g
            JOIN group_members gm ON g.id = gm.group_id
            WHERE gm.username = ?
            ORDER BY g.created_at DESC;
            """, (username, username)
        ).fetchall()

        for group_row in member_groups:
            members_info = conn.execute(
                "SELECT username, role FROM group_members WHERE group_id = ?",
                (group_row["id"],)
            ).fetchall()
            
            groups_data.append(GroupInfoResponse(
                id=group_row["id"],
                name=group_row["name"],
                description=group_row["description"],
                creator_username=group_row["creator_username"],
                created_at=group_row["created_at"],
                memberCount=group_row["memberCount"],
                isAdmin=(group_row["userRole"] == 'admin'),
                members=[{"username": m["username"], "role": m["role"]} for m in members_info]
            ))
        
        return groups_data
    except Exception as e:
        logger.error(f"Error fetching groups for user {username}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch groups: {e}")
    finally:
        conn.close()


@app.get("/groups/{group_id}/messages")
async def get_group_messages(group_id: int, username: str = Query(...), limit: int = Query(50, le=100)):
    """Get recent messages from a group."""
    conn = get_db_connection()
    try:
        # Check if user is a member
        is_member = conn.execute(
            "SELECT 1 FROM group_members WHERE group_id = ? AND username = ?",
            (group_id, username)
        ).fetchone()
        
        if not is_member:
            raise HTTPException(status_code=403, detail="Not a member of this group")
        
        # Get recent messages
        messages = conn.execute(
            """
            SELECT sender_username, content, message_type, file_url, file_mime_type, 
                   file_size, file_id, created_at
            FROM group_messages 
            WHERE group_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
            """,
            (group_id, limit)
        ).fetchall()
        
        return [dict(msg) for msg in reversed(messages)]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting group messages: {e}")
        raise HTTPException(status_code=500, detail="Failed to get messages")
    finally:
        conn.close()

@app.get("/groups/{group_id}")
async def get_group_details(group_id: int, username: str = Query(...)):
    """Get details of a specific group."""
    conn = get_db_connection()
    try:
        # Check if user is a member of the group
        is_member = conn.execute(
            "SELECT role FROM group_members WHERE group_id = ? AND username = ?",
            (group_id, username)
        ).fetchone()
        
        if not is_member:
            raise HTTPException(status_code=404, detail="Group not found or you do not have access")
        
        # Get group details
        group_data = conn.execute(
            """
            SELECT id, name, description, creator_username, created_at,
                   (SELECT COUNT(*) FROM group_members WHERE group_id = ?) as member_count
            FROM groups 
            WHERE id = ?
            """,
            (group_id, group_id)
        ).fetchone()
        
        if not group_data:
            raise HTTPException(status_code=404, detail="Group not found")
        
        # Get all members
        members = conn.execute(
            "SELECT username, role, joined_at FROM group_members WHERE group_id = ? ORDER BY joined_at",
            (group_id,)
        ).fetchall()
        
        # Format response
        group_info = {
            "id": group_data["id"],
            "name": group_data["name"],
            "description": group_data["description"],
            "creator_username": group_data["creator_username"],
            "created_at": group_data["created_at"],
            "memberCount": group_data["member_count"],
            "isAdmin": (is_member["role"] == 'admin'),
            "members": [
                {
                    "username": member["username"],
                    "role": member["role"],
                    "joined_at": member["joined_at"]
                }
                for member in members
            ]
        }
        
        logger.info(f"Retrieved group {group_id} details for user {username}")
        return group_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting group {group_id} details for user {username}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get group details: {str(e)}")
    finally:
        conn.close()



@app.delete("/groups/{group_id}")
async def delete_group_endpoint(group_id: int, current_user: str = Query(...)):
    """Delete a group. Only the creator/admin can delete a group."""
    conn = get_db_connection()
    try:
        # Check if group exists
        group_info = conn.execute(
            "SELECT id, name, creator_username FROM groups WHERE id = ?",
            (group_id,)
        ).fetchone()
        
        if not group_info:
            raise HTTPException(status_code=404, detail="Group not found")
        
        # Check if current user is the creator or an admin
        user_role = conn.execute(
            "SELECT role FROM group_members WHERE group_id = ? AND username = ?",
            (group_id, current_user)
        ).fetchone()
        
        if not user_role:
            raise HTTPException(status_code=403, detail="You are not a member of this group")
        
        # Check if user has permission to delete (creator or admin)
        is_creator = group_info["creator_username"] == current_user
        is_admin = user_role["role"] == "admin"
        
        if not (is_creator or is_admin):
            raise HTTPException(status_code=403, detail="Only the group creator or admin can delete the group")
        
        # Get all group members before deletion (for notifications)
        group_members = conn.execute(
            "SELECT username FROM group_members WHERE group_id = ?",
            (group_id,)
        ).fetchall()
        
        # Delete the group (CASCADE will handle group_members and group_messages)
        conn.execute("DELETE FROM groups WHERE id = ?", (group_id,))
        conn.commit()
        
        logger.info(f"Group {group_id} ({group_info['name']}) deleted by {current_user}")
        
        # Notify all group members via Socket.IO
        notification_data = {
            "type": "group_deleted",
            "group_id": group_id,
            "group_name": group_info["name"],
            "deleted_by": current_user,
            "message": f"Group '{group_info['name']}' has been deleted by {current_user}"
        }
        
        for member in group_members:
            member_sid = username_to_sid.get(member["username"])
            if member_sid:
                # Remove from Socket.IO room
                await sio.leave_room(member_sid, str(group_id))
                # Send notification
                await sio.emit("group_notification", notification_data, room=member_sid)
        
        return {"message": "Group deleted successfully", "group_id": group_id}
        
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        logger.error(f"Error deleting group {group_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete group: {str(e)}")
    finally:
        conn.close()


@app.put("/groups/{group_id}")
async def update_group_endpoint(group_id: int, group_update: GroupUpdate, current_user: str = Query(...)):
    """Update group information. Only admins can update group info."""
    conn = get_db_connection()
    try:
        # Check if group exists
        group_info = conn.execute(
            "SELECT id, name, description, creator_username FROM groups WHERE id = ?",
            (group_id,)
        ).fetchone()
        
        if not group_info:
            raise HTTPException(status_code=404, detail="Group not found")
        
        # Check if current user is admin
        user_role = conn.execute(
            "SELECT role FROM group_members WHERE group_id = ? AND username = ?",
            (group_id, current_user)
        ).fetchone()
        
        if not user_role:
            raise HTTPException(status_code=403, detail="You are not a member of this group")
        
        if user_role["role"] != "admin":
            raise HTTPException(status_code=403, detail="Only admins can update group information")
        
        # Prepare update data
        updates = []
        params = []
        
        if group_update.name is not None and group_update.name.strip():
            # Check if new name already exists (for other groups)
            existing_name = conn.execute(
                "SELECT id FROM groups WHERE name = ? AND id != ?",
                (group_update.name.strip(), group_id)
            ).fetchone()
            
            if existing_name:
                raise HTTPException(status_code=400, detail="Group name already exists")
            
            updates.append("name = ?")
            params.append(group_update.name.strip())
        
        if group_update.description is not None:
            updates.append("description = ?")
            params.append(group_update.description.strip())
        
        if not updates:
            raise HTTPException(status_code=400, detail="No valid updates provided")
        
        # Perform update
        params.append(group_id)
        update_query = f"UPDATE groups SET {', '.join(updates)} WHERE id = ?"
        conn.execute(update_query, params)
        conn.commit()
        
        # Get updated group info
        updated_group = conn.execute(
            "SELECT id, name, description, creator_username FROM groups WHERE id = ?",
            (group_id,)
        ).fetchone()
        
        logger.info(f"Group {group_id} updated by {current_user}")
        
        # Notify all group members
        group_members = conn.execute(
            "SELECT username FROM group_members WHERE group_id = ?",
            (group_id,)
        ).fetchall()
        
        notification_data = {
            "type": "group_updated",
            "group_id": group_id,
            "group_name": updated_group["name"],
            "updated_by": current_user,
            "message": f"Group '{updated_group['name']}' has been updated"
        }
        
        for member in group_members:
            member_sid = username_to_sid.get(member["username"])
            if member_sid:
                await sio.emit("group_notification", notification_data, room=member_sid)
        
        return {
            "message": "Group updated successfully",
            "group": {
                "id": updated_group["id"],
                "name": updated_group["name"],
                "description": updated_group["description"],
                "creator_username": updated_group["creator_username"]
            }
        }
        
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        logger.error(f"Error updating group {group_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update group: {str(e)}")
    finally:
        conn.close()


@app.delete("/groups/{group_id}/members/{username}")
async def remove_member_endpoint(group_id: int, username: str, current_user: str = Query(...)):
    """Remove a member from a group or leave a group."""
    conn = get_db_connection()
    try:
        # Check if group exists
        group_info = conn.execute(
            "SELECT id, name, creator_username FROM groups WHERE id = ?",
            (group_id,)
        ).fetchone()
        
        if not group_info:
            raise HTTPException(status_code=404, detail="Group not found")
        
        # Check if target user is a member
        target_member = conn.execute(
            "SELECT username, role FROM group_members WHERE group_id = ? AND username = ?",
            (group_id, username)
        ).fetchone()
        
        if not target_member:
            raise HTTPException(status_code=404, detail="User is not a member of this group")
        
        # Check permissions
        if current_user == username:
            # User is leaving the group themselves
            pass
        else:
            # Someone else is trying to remove the user
            current_user_role = conn.execute(
                "SELECT role FROM group_members WHERE group_id = ? AND username = ?",
                (group_id, current_user)
            ).fetchone()
            
            if not current_user_role:
                raise HTTPException(status_code=403, detail="You are not a member of this group")
            
            if current_user_role["role"] != "admin":
                raise HTTPException(status_code=403, detail="Only admins can remove members")
            
            # Don't allow removing the creator unless they're leaving themselves
            if target_member["username"] == group_info["creator_username"]:
                raise HTTPException(status_code=403, detail="Cannot remove the group creator")
        
        # Remove the member
        conn.execute(
            "DELETE FROM group_members WHERE group_id = ? AND username = ?",
            (group_id, username)
        )
        
        # Check if group is empty now
        remaining_members = conn.execute(
            "SELECT COUNT(*) as count FROM group_members WHERE group_id = ?",
            (group_id,)
        ).fetchone()
        
        if remaining_members["count"] == 0:
            # Delete empty group
            conn.execute("DELETE FROM groups WHERE id = ?", (group_id,))
            logger.info(f"Empty group {group_id} deleted")
        
        conn.commit()
        
        # Remove from Socket.IO room
        target_sid = username_to_sid.get(username)
        if target_sid:
            await sio.leave_room(target_sid, str(group_id))
        
        # Notify remaining members
        if remaining_members["count"] > 0:
            remaining_member_list = conn.execute(
                "SELECT username FROM group_members WHERE group_id = ?",
                (group_id,)
            ).fetchall()
            
            action = "left" if current_user == username else "was removed from"
            notification_data = {
                "type": "member_removed",
                "group_id": group_id,
                "group_name": group_info["name"],
                "username": username,
                "removed_by": current_user,
                "message": f"{username} {action} the group"
            }
            
            for member in remaining_member_list:
                member_sid = username_to_sid.get(member["username"])
                if member_sid:
                    await sio.emit("group_notification", notification_data, room=member_sid)
        
        # Notify the removed/leaving user
        if target_sid:
            leave_notification = {
                "type": "left_group",
                "group_id": group_id,
                "group_name": group_info["name"],
                "message": f"You have {'left' if current_user == username else 'been removed from'} the group '{group_info['name']}'"
            }
            await sio.emit("group_notification", leave_notification, room=target_sid)
        
        logger.info(f"User {username} {'left' if current_user == username else 'removed from'} group {group_id} by {current_user}")
        
        return {"message": f"{'Left' if current_user == username else 'Member removed from'} group successfully"}
        
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        logger.error(f"Error removing member {username} from group {group_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove member: {str(e)}")
    finally:
        conn.close()

# --- FILE UPLOAD ---
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    uploader_id: str = Form(...),
    recipient_username: Optional[str] = Form(None),
    group_id: Optional[str] = Form(None)
):
    logger.info(f"=== UPLOAD DEBUG ===")
    logger.info(f"Uploader ID: {uploader_id}")
    logger.info(f"Recipient: {recipient_username}")
    logger.info(f"Group ID: {group_id}")
    
    sender_username = sid_to_username.get(uploader_id)
    if not sender_username:
        logger.error(f"Authentication failed - no username found for SID: {uploader_id}")
        raise HTTPException(status_code=403, detail="Uploader not authenticated.")

    if not recipient_username and not group_id:
        raise HTTPException(status_code=400, detail="Either recipient_username or group_id must be provided.")
    
    if recipient_username and group_id:
        raise HTTPException(status_code=400, detail="Cannot send to both a recipient and a group simultaneously.")

    # Handle file upload
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
                    raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE / (1024*1024*1024):.0f}GB limit.")
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "recipient_username": recipient_username,
        "group_id": group_id
    }
    
    if group_id:
        # Save to group messages database
        conn = get_db_connection()
        try:
            conn.execute(
                """
                INSERT INTO group_messages (group_id, sender_username, content, message_type, 
                                          file_url, file_mime_type, file_size, file_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (group_id, sender_username, filename, "file", 
                 file_metadata["url"], file_metadata["mime_type"], file_size, file_id)
            )
            conn.commit()
        finally:
            conn.close()
            
        await sio.emit("message", message_data, room=group_id)
        logger.info(f"Sent file '{filename}' from {sender_username} to group {group_id}")
    elif recipient_username:
        recipient_sid = username_to_sid.get(recipient_username)
        if recipient_sid:
            await sio.emit("message", message_data, room=recipient_sid)
            logger.info(f"Sent file '{filename}' from {sender_username} to {recipient_username}")
        
        await sio.emit("message", message_data, room=uploader_id)
        logger.info(f"Sent file confirmation to sender {sender_username}")

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


# --- SOCKET.IO EVENTS ---
@sio.event
async def connect(sid, environ, auth):
    """Handle client connection."""
    try:
        logger.info(f"=== CONNECTION ATTEMPT ===")
        logger.info(f"SID: {sid}")

        username = None

        if auth and isinstance(auth, dict):
            username = auth.get("username")

        if not username:
            query_string = environ.get("QUERY_STRING", b"")
            if isinstance(query_string, bytes):
                query_string = query_string.decode()
            elif query_string is None:
                query_string = ""

            if query_string:
                params = parse_qs(query_string)
                username = params.get("username", [None])[0]
                if not username:
                    username = params.get("userId", [None])[0]

        if not username:
            logger.error("No username found in connection attempt")
            return False

        # Clean up existing connection if any
        if username in username_to_sid:
            old_sid = username_to_sid[username]
            logger.info(f"User {username} reconnecting, removing old connection {old_sid}")
            
            if old_sid in connected_users:
                del connected_users[old_sid]
            if old_sid in sid_to_username:
                del sid_to_username[old_sid]

        # Add user to connected list
        connected_users[sid] = {
            "username": username,
            "sid": sid,
            "status": "online",
        }
        
        sid_to_username[sid] = username
        username_to_sid[username] = sid
        
        logger.info(f"User {username} connected with SID {sid}")

        # Join group rooms
        conn = get_db_connection()
        try:
            member_groups = conn.execute(
                "SELECT group_id FROM group_members WHERE username = ?", (username,)
            ).fetchall()
            
            for row in member_groups:
                group_id_str = str(row["group_id"])
                await sio.enter_room(sid, group_id_str)
                logger.info(f"User {username} joined group room {group_id_str}")
                
        except Exception as e:
            logger.error(f"Error joining user {username} to groups: {e}")
        finally:
            conn.close()

        # Notify other users
        await sio.emit("user_joined", {"username": username, "status": "online"}, skip_sid=sid)
        await broadcast_user_list()
        
        return True

    except Exception as e:
        logger.error(f"Connection failed for {sid}: {str(e)}")
        return False


@sio.event
async def disconnect(sid):
    """Handle client disconnection."""
    username = sid_to_username.get(sid)
    logger.info(f"User disconnecting: {username} (SID: {sid})")

    try:
        # Handle call cleanup
        if sid in call_sessions:
            partner_sid = call_sessions[sid]
            await sio.emit("call_ended", {"reason": "Partner disconnected"}, room=partner_sid)
            
            if partner_sid in connected_users:
                connected_users[partner_sid]["status"] = "online"
            
            call_sessions.pop(partner_sid, None)
            call_sessions.pop(sid, None)

        # Clean up user data
        if username:
            await sio.emit("user_left", {"username": username})
            connected_users.pop(sid, None)
            sid_to_username.pop(sid, None)
            username_to_sid.pop(username, None)

        await broadcast_user_list()
        
    except Exception as e:
        logger.error(f"Error during disconnect cleanup: {e}")


@sio.event
async def request_user_list(sid):
    """Handle explicit requests for user list."""
    logger.info(f"User list requested by {sid}")
    await asyncio.sleep(0.1)
    await broadcast_user_list()


@sio.event
async def message(sid, data):
    """Handle incoming messages - UNIFIED for both direct and group messages."""
    sender_username = sid_to_username.get(sid)
    if not sender_username:
        logger.warning(f"Message from unknown SID: {sid}")
        return

    recipient_username = data.get("recipient_username")
    group_id = data.get("group_id")

    if not recipient_username and not group_id:
        logger.warning(f"Message from {sender_username} has no recipient or group")
        return

    if recipient_username and group_id:
        logger.warning(f"Message from {sender_username} has both recipient and group")
        return

    message_data = {
        "sid": sid,
        "username": sender_username,
        "type": data.get("type", "text"),
        "text": data.get("content"),
        "file": data.get("file"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "recipient_username": recipient_username,
        "group_id": group_id
    }

    try:
        if group_id:
            # Group message handling
            conn = get_db_connection()
            try:
                # Validate membership
                is_member = conn.execute(
                    "SELECT 1 FROM group_members WHERE group_id = ? AND username = ?",
                    (group_id, sender_username)
                ).fetchone()
                
                if not is_member:
                    logger.warning(f"User {sender_username} not a member of group {group_id}")
                    await sio.emit("error", {"message": "Not a member of this group"}, room=sid)
                    return
                
                # Save text message to database
                if data.get("type") == "text" and data.get("content"):
                    conn.execute(
                        """
                        INSERT INTO group_messages (group_id, sender_username, content, message_type)
                        VALUES (?, ?, ?, ?)
                        """,
                        (group_id, sender_username, data.get("content"), "text")
                    )
                    conn.commit()
                
                # Send to group room
                await sio.emit("message", message_data, room=str(group_id))
                logger.info(f"Sent group message from {sender_username} to group {group_id}")
                
            finally:
                conn.close()
            
        elif recipient_username:
            # Direct message handling
            recipient_sid = username_to_sid.get(recipient_username)
            
            # Send to recipient if online
            if recipient_sid:
                await sio.emit("message", message_data, room=recipient_sid)
                logger.info(f"Sent message from {sender_username} to {recipient_username}")
            
            # Send confirmation to sender
            await sio.emit("message", message_data, room=sid)
            
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        await sio.emit("error", {"message": "Failed to send message"}, room=sid)


# --- TYPING EVENTS - UNIFIED HANDLING ---
@sio.event
async def typing_start(sid, data):
    """Handle typing start for both direct messages and groups."""
    sender_username = sid_to_username.get(sid)
    if not sender_username:
        logger.warning(f"typing_start: No username for SID: {sid}")
        return

    recipient_username = data.get("recipient_username")
    group_id = data.get("group_id")

    if recipient_username:
        # Direct message typing
        recipient_sid = username_to_sid.get(recipient_username)
        if recipient_sid:
            await sio.emit("typing_start", {"username": sender_username}, room=recipient_sid)
            logger.info(f"User {sender_username} started typing to {recipient_username}")
    elif group_id:
        # Group typing - validate membership first
        conn = get_db_connection()
        try:
            is_member = conn.execute(
                "SELECT 1 FROM group_members WHERE group_id = ? AND username = ?",
                (group_id, sender_username)
            ).fetchone()
            
            if is_member:
                await sio.emit(
                    "typing_start",
                    {"username": sender_username, "group_id": group_id},
                    room=str(group_id),
                    skip_sid=sid
                )
                logger.info(f"User {sender_username} started typing in group {group_id}")
        finally:
            conn.close()


@sio.event
async def typing_stop(sid, data):
    """Handle typing stop for both direct messages and groups."""
    sender_username = sid_to_username.get(sid)
    if not sender_username:
        logger.warning(f"typing_stop: No username for SID: {sid}")
        return

    recipient_username = data.get("recipient_username")
    group_id = data.get("group_id")

    if recipient_username:
        # Direct message typing
        recipient_sid = username_to_sid.get(recipient_username)
        if recipient_sid:
            await sio.emit("typing_stop", {"username": sender_username}, room=recipient_sid)
            logger.info(f"User {sender_username} stopped typing to {recipient_username}")
    elif group_id:
        # Group typing - validate membership first
        conn = get_db_connection()
        try:
            is_member = conn.execute(
                "SELECT 1 FROM group_members WHERE group_id = ? AND username = ?",
                (group_id, sender_username)
            ).fetchone()
            
            if is_member:
                await sio.emit(
                    "typing_stop",
                    {"username": sender_username, "group_id": group_id},
                    room=str(group_id),
                    skip_sid=sid
                )
                logger.info(f"User {sender_username} stopped typing in group {group_id}")
        finally:
            conn.close()


# --- GROUP SOCKET EVENTS ---
@sio.event
async def join_group_room(sid, data):
    """Join a group room for real-time messaging."""
    group_id = data.get("group_id")
    username = sid_to_username.get(sid)

    if not group_id or not username:
        logger.warning(f"join_group_room: Missing group_id or username for SID: {sid}")
        return

    conn = get_db_connection()
    try:
        # Verify membership
        is_member = conn.execute(
            "SELECT 1 FROM group_members WHERE group_id = ? AND username = ?",
            (group_id, username)
        ).fetchone()

        if is_member:
            await sio.enter_room(sid, str(group_id))
            logger.info(f"User {username} joined group room {group_id}")
            await sio.emit("group_room_joined", {"group_id": group_id, "success": True}, room=sid)
        else:
            logger.warning(f"User {username} attempted to join non-member group room {group_id}")
            await sio.emit("group_room_joined", {"group_id": group_id, "success": False, "error": "Not a member"}, room=sid)
    except Exception as e:
        logger.error(f"Error joining group room {group_id} for user {username}: {e}")
        await sio.emit("group_room_joined", {"group_id": group_id, "success": False, "error": str(e)}, room=sid)
    finally:
        conn.close()


@sio.event
async def leave_group_room(sid, data):
    """Leave a group room."""
    group_id = data.get("group_id")
    username = sid_to_username.get(sid)

    if not group_id or not username:
        logger.warning(f"leave_group_room: Missing group_id or username for SID: {sid}")
        return

    try:
        await sio.leave_room(sid, str(group_id))
        logger.info(f"User {username} left group room {group_id}")
        await sio.emit("group_room_left", {"group_id": group_id, "success": True}, room=sid)
    except Exception as e:
        logger.error(f"Error leaving group room {group_id} for user {username}: {e}")
        await sio.emit("group_room_left", {"group_id": group_id, "success": False, "error": str(e)}, room=sid)


# --- MESSAGE DELETION ---
@sio.event
async def delete_message_event(sid, data):
    """Handle real-time message deletion for both direct and group messages."""
    sender_username = sid_to_username.get(sid)
    if not sender_username:
        await sio.emit("delete_error", {"error": "User not authenticated"}, room=sid)
        return

    message_id = data.get("message_id")
    chat_id = data.get("chat_id")
    delete_for_everyone = data.get("delete_for_everyone", False)
    recipient_username = data.get("recipient_username")
    group_id = data.get("group_id")

    if not message_id:
        await sio.emit("delete_error", {"error": "Message ID required"}, room=sid)
        return

    try:
        conn = get_db_connection()
        
        if group_id:
            # FIXED: Handle group message deletion properly
            if delete_for_everyone:
                # For delete for everyone, check if user is admin or message sender
                message_check = conn.execute(
                    "SELECT sender_username FROM group_messages WHERE id = ? AND group_id = ?",
                    (message_id, group_id)
                ).fetchone()
                
                admin_check = conn.execute(
                    "SELECT role FROM group_members WHERE group_id = ? AND username = ?",
                    (group_id, sender_username)
                ).fetchone()
                
                is_admin = admin_check and admin_check[0] == 'admin'
                is_sender = message_check and message_check[0] == sender_username
                
                if not (is_admin or is_sender):
                    await sio.emit("delete_error", {"error": "You can only delete your own messages or admin can delete any message"}, room=sid)
                    return
                
                # Delete the message from database
                conn.execute(
                    "DELETE FROM group_messages WHERE id = ? AND group_id = ?",
                    (message_id, group_id)
                )
            else:
                # For delete for me only, just mark as deleted for this user
                # We could implement a separate table for user-specific deletions
                # For now, we'll only allow delete for everyone in groups for simplicity
                await sio.emit("delete_error", {"error": "Only 'delete for everyone' is supported in groups"}, room=sid)
                return
        else:
            # Handle direct message deletion (existing logic)
            if delete_for_everyone:
                # Check if user is the sender
                message_check = conn.execute(
                    "SELECT sender_username FROM messages WHERE id = ? AND chat_id = ?",
                    (message_id, chat_id)
                ).fetchone()
                
                if not message_check or message_check[0] != sender_username:
                    await sio.emit("delete_error", {"error": "You can only delete your own messages"}, room=sid)
                    return
            
            # Delete from regular messages table
            conn.execute(
                "DELETE FROM messages WHERE id = ? AND chat_id = ?",
                (message_id, chat_id)
            )
        
        conn.commit()
        conn.close()

        deletion_data = {
            "message_id": message_id,
            "chat_id": chat_id,
            "delete_for_everyone": delete_for_everyone,
            "deleted_by": sender_username,
            "group_id": group_id
        }

        if group_id:
            # Group message deletion - emit to all group members
            logger.info(f"Emitting group message deletion to room {group_id}")
            await sio.emit("message_deleted", deletion_data, room=str(group_id))
        elif recipient_username:
            # Direct message deletion
            await sio.emit("message_deleted", deletion_data, room=sid)
            recipient_sid = username_to_sid.get(recipient_username)
            if recipient_sid and delete_for_everyone:
                await sio.emit("message_deleted", deletion_data, room=recipient_sid)
        else:
            await sio.emit("message_deleted", deletion_data, room=sid)

        logger.info(f"Message {message_id} deleted by {sender_username} (delete_for_everyone: {delete_for_everyone})")

    except Exception as e:
        logger.error(f"Error in delete_message_event: {e}")
        await sio.emit("delete_error", {"error": "Failed to delete message"}, room=sid)


# --- CALL HANDLING ---
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
        status = connected_users[target_sid].get("status", "unavailable")
        await sio.emit("call_error", {"message": f"User is currently busy ({status})."}, room=sid)
        return

    # Set statuses to 'Ringing'
    connected_users[sid]["status"] = "ringing"
    connected_users[target_sid]["status"] = "ringing"
    
    await broadcast_user_list()

    # Emit the incoming call event to the target user
    await sio.emit("incoming_call", {"caller_id": sid, "caller_username": caller_username}, room=target_sid)


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
        call_sessions[sid] = caller_id
        call_sessions[caller_id] = sid
        connected_users[sid]["status"] = "in_call"
        connected_users[caller_id]["status"] = "in_call"

        await sio.emit("call_accepted", {"responder_id": sid}, room=caller_id)
    else:
        logger.info(f"Call rejected by {sid} for caller {caller_id}")
        connected_users[caller_id]["status"] = "online"
        connected_users[sid]["status"] = "online"
        await sio.emit("call_rejected", {"responder_username": connected_users[sid]["username"]}, room=caller_id)

    await broadcast_user_list()


@sio.event
async def end_call(sid):
    """Handle call ending initiated by one of the participants."""
    if sid in call_sessions:
        partner_sid = call_sessions[sid]
        logger.info(f"Call ended by {sid}. Notifying partner {partner_sid}")

        await sio.emit("call_ended", {"reason": "The call was ended by your partner."}, room=partner_sid)

        # Reset statuses
        if sid in connected_users:
            connected_users[sid]["status"] = "online"
        if partner_sid in connected_users:
            connected_users[partner_sid]["status"] = "online"

        # Clean up session for both users
        call_sessions.pop(sid, None)
        call_sessions.pop(partner_sid, None)

        await broadcast_user_list()


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
        uvicorn.run(app, host=host, port=8000, ssl_certfile=cert_file, ssl_keyfile=key_file)
    else:
        logger.error(f"Missing SSL files: {cert_file}, {key_file}. Please generate them using openssl.")
        # Fallback to HTTP for development
        uvicorn.run(app, host="127.0.0.1", port=8000)