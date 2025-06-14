/**
 * call.js
 * * This script handles all client-side logic for the video call page,
 * including WebRTC connections, screen sharing, and error handling.
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const localVideo = document.getElementById('localVideo');
    const remoteVideo = document.getElementById('remoteVideo');
    const userListElement = document.getElementById('user-list');
    const hangUpBtn = document.getElementById('hangUpBtn');
    const muteBtn = document.getElementById('muteBtn');
    const videoBtn = document.getElementById('videoBtn');
    const screenShareBtn = document.getElementById('screenShareBtn');
    const callPlaceholder = document.getElementById('call-placeholder');

    // Modal elements
    const incomingCallModal = new bootstrap.Modal(document.getElementById('incomingCallModal'));
    const callerUsernameElement = document.getElementById('callerUsername');
    const acceptCallBtn = document.getElementById('acceptCallBtn');
    const rejectCallBtn = document.getElementById('rejectCallBtn');

    // Toast element for notifications
    const toastContainer = document.querySelector('.toast-container');

    // --- State Management ---
    let localStream;
    let screenStream;
    let peerConnection;
    let mySid;
    let currentCallPartner = null;
    let callerId = null;
    let isSharingScreen = false;

    const socket = io();

    const configuration = {
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    };

    /**
     * Shows a toast notification.
     * @param {string} message The message to display.
     * @param {string} type 'success', 'danger', or 'info'.
     */
    function showToast(message, type = 'info') {
        const toastEl = document.createElement('div');
        toastEl.className = `toast align-items-center text-white bg-${type} border-0`;
        toastEl.setAttribute('role', 'alert');
        toastEl.setAttribute('aria-live', 'assertive');
        toastEl.setAttribute('aria-atomic', 'true');

        toastEl.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        `;
        toastContainer.appendChild(toastEl);
        const toast = new bootstrap.Toast(toastEl, { delay: 5000 });
        toast.show();
        toastEl.addEventListener('hidden.bs.toast', () => toastEl.remove());
    }

    /**
     * Initializes the local camera and microphone.
     */
    async function initLocalMedia() {
        try {
            localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
            localVideo.srcObject = localStream;
        } catch (error) {
            console.error("Error accessing media devices.", error);
            showToast("Could not access camera/microphone. Please check permissions.", "danger");
        }
    }

    /**
     * Toggles screen sharing.
     */
    async function toggleScreenShare() {
        if (!peerConnection) {
            showToast("You must be in a call to share your screen.", "info");
            return;
        }

        if (isSharingScreen) {
            // Stop screen sharing and revert to camera
            await switchTrack(localStream.getVideoTracks()[0]);
            screenStream.getTracks().forEach(track => track.stop());
            screenStream = null;
            isSharingScreen = false;
            screenShareBtn.classList.remove('active-control');
            screenShareBtn.innerHTML = '<i class="fas fa-desktop"></i> Share Screen';
        } else {
            // Start screen sharing
            try {
                screenStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
                const screenTrack = screenStream.getVideoTracks()[0];
                await switchTrack(screenTrack);

                // Listen for when the user stops sharing via browser controls
                screenTrack.onended = () => {
                    if (isSharingScreen) toggleScreenShare();
                };
                
                isSharingScreen = true;
                screenShareBtn.classList.add('active-control');
                screenShareBtn.innerHTML = '<i class="fas fa-stop-circle"></i> Stop Sharing';
            } catch (error) {
                console.error("Error starting screen share.", error);
                showToast("Could not start screen sharing.", "danger");
            }
        }
    }
    
    /**
     * Switches the video track being sent to the peer.
     * @param {MediaStreamTrack} newTrack The new track to send.
     */
    async function switchTrack(newTrack) {
        const sender = peerConnection.getSenders().find(s => s.track.kind === 'video');
        if (sender) {
            await sender.replaceTrack(newTrack);
        }
    }

    /**
     * Updates the user list UI.
     * @param {Array} users - Array of user objects from the server.
     */
    function updateUserList(users) {
        userListElement.innerHTML = '';
        users.forEach(user => {
            if (user.id === mySid) return;
            const userElement = document.createElement('a');
            userElement.href = '#';
            userElement.className = 'list-group-item list-group-item-action d-flex justify-content-between align-items-center';
            userElement.innerHTML = `
                <div>
                    <span>${user.username}</span>
                    <small class="text-muted ms-2 status-${user.status}">(${user.status})</small>
                </div>
            `;
            const callBtn = document.createElement('button');
            callBtn.className = 'btn btn-sm btn-success';
            callBtn.innerHTML = '<i class="fas fa-phone"></i>';
            callBtn.disabled = user.status !== 'online';
            callBtn.onclick = () => startCall(user.id);
            userElement.appendChild(callBtn);
            userListElement.appendChild(userElement);
        });
    }

    /**
     * Creates and configures the RTCPeerConnection.
     * @param {string} partnerSid - The SID of the call partner.
     */
    function createPeerConnection(partnerSid) {
        try {
            peerConnection = new RTCPeerConnection(configuration);

            peerConnection.onicecandidate = event => {
                if (event.candidate) {
                    socket.emit('webrtc_ice_candidate', { candidate: event.candidate });
                }
            };

            peerConnection.ontrack = event => {
                remoteVideo.srcObject = event.streams[0];
                callPlaceholder.style.display = 'none';
                remoteVideo.style.display = 'block';
            };

            // NEW: Handle connection state changes for better error recovery
            peerConnection.oniceconnectionstatechange = () => {
                console.log(`ICE Connection State: ${peerConnection.iceConnectionState}`);
                if (peerConnection.iceConnectionState === 'failed' || peerConnection.iceConnectionState === 'disconnected') {
                    showToast("Connection lost. Please try again.", "danger");
                    endCall();
                }
            };
            
            localStream.getTracks().forEach(track => {
                peerConnection.addTrack(track, localStream);
            });
            
            updateCallControls(true);
            currentCallPartner = partnerSid;
        } catch (error) {
            console.error("Error creating peer connection:", error);
            showToast("Failed to create call connection.", "danger");
            endCall();
        }
    }

    /**
     * Initiates a call to a target user.
     * @param {string} targetUserId - The SID of the user to call.
     */
    function startCall(targetUserId) {
        if (!localStream) {
            showToast("Your camera/microphone are not ready.", "danger");
            return;
        }
        console.log(`Attempting to call user: ${targetUserId}`);
        socket.emit('call_request', { target_user_id: targetUserId });
    }

    /**
     * Ends the current call and resets the state.
     */
    function endCall() {
        if (currentCallPartner) {
            socket.emit('end_call');
        }
        if (isSharingScreen) {
             toggleScreenShare(); // Stop sharing and revert to camera
        }
        if (peerConnection) {
            peerConnection.close();
            peerConnection = null;
        }
        remoteVideo.srcObject = null;
        remoteVideo.style.display = 'none';
        callPlaceholder.style.display = 'block';
        updateCallControls(false);
        currentCallPartner = null;
        callerId = null;
    }
    
    /**
     * Enables or disables call control buttons.
     * @param {boolean} inCall - Whether the user is currently in a call.
     */
    function updateCallControls(inCall) {
        hangUpBtn.disabled = !inCall;
        muteBtn.disabled = !inCall;
        videoBtn.disabled = !inCall;
        screenShareBtn.disabled = !inCall;
    }


    // --- Socket.IO Event Handlers ---
    
    socket.on('connect', () => {
        mySid = socket.id;
        console.log(`Connected to server with SID: ${mySid}`);
        initLocalMedia();
    });

    socket.on('user_list', updateUserList);

    socket.on('incoming_call', (data) => {
        if (currentCallPartner) return; // Ignore if already in a call
        callerId = data.caller_id;
        callerUsernameElement.textContent = data.caller_username;
        incomingCallModal.show();
    });
    
    socket.on('call_accepted', async ({ responder_id }) => {
        console.log("Call accepted by:", responder_id);
        createPeerConnection(responder_id);
        try {
            const offer = await peerConnection.createOffer();
            await peerConnection.setLocalDescription(offer);
            socket.emit('webrtc_offer', { offer: offer });
        } catch (error) {
            console.error("Error creating offer:", error);
            showToast("Failed to initiate call.", "danger");
            endCall();
        }
    });

    socket.on('call_rejected', ({ responder_username }) => {
        showToast(`${responder_username} rejected the call.`, 'info');
        endCall();
    });

    socket.on('call_error', ({ message }) => {
        showToast(message, 'danger');
        endCall();
    });

    socket.on('webrtc_offer', async ({ offer, from_sid }) => {
        if (!peerConnection) createPeerConnection(from_sid);
        try {
            await peerConnection.setRemoteDescription(new RTCSessionDescription(offer));
            const answer = await peerConnection.createAnswer();
            await peerConnection.setLocalDescription(answer);
            socket.emit('webrtc_answer', { answer: answer });
        } catch (error) {
            console.error("Error handling offer:", error);
            showToast("Failed to accept call.", "danger");
            endCall();
        }
    });

    socket.on('webrtc_answer', async ({ answer }) => {
        try {
            await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));
        } catch (error) {
            console.error("Error setting remote answer:", error);
            showToast("Call connection failed.", "danger");
            endCall();
        }
    });

    socket.on('webrtc_ice_candidate', async ({ candidate }) => {
        if (peerConnection) {
            try {
                await peerConnection.addIceCandidate(new RTCIceCandidate(candidate));
            } catch (error) {
                console.error("Error adding ICE candidate", error);
            }
        }
    });

    socket.on('call_ended', ({ reason }) => {
        showToast(reason, 'info');
        endCall();
    });
    
    socket.on('disconnect', () => {
        showToast("You have been disconnected from the server.", "danger");
        endCall();
        // Disable all user interaction
        document.querySelectorAll('button').forEach(btn => btn.disabled = true);
    });

    // --- UI Event Listeners ---

    acceptCallBtn.addEventListener('click', () => {
        if (!localStream) {
            showToast("Your camera and microphone are not ready.", "danger");
            return;
        }
        socket.emit('call_response', { caller_id: callerId, accepted: true });
        incomingCallModal.hide();
    });

    rejectCallBtn.addEventListener('click', () => {
        socket.emit('call_response', { caller_id: callerId, accepted: false });
    });

    hangUpBtn.addEventListener('click', endCall);
    screenShareBtn.addEventListener('click', toggleScreenShare);

    muteBtn.addEventListener('click', () => {
        const audioTrack = localStream.getAudioTracks()[0];
        audioTrack.enabled = !audioTrack.enabled;
        muteBtn.innerHTML = audioTrack.enabled ? '<i class="fas fa-microphone"></i> Mute' : '<i class="fas fa-microphone-slash"></i> Unmute';
        muteBtn.classList.toggle('active-control');
    });

    videoBtn.addEventListener('click', () => {
        const videoTrack = localStream.getVideoTracks()[0];
        videoTrack.enabled = !videoTrack.enabled;
        videoBtn.innerHTML = videoTrack.enabled ? '<i class="fas fa-video"></i> Video' : '<i class="fas fa-video-slash"></i> No Video';
        videoBtn.classList.toggle('active-control');
    });
});