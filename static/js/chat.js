class ChatApp {
    constructor() {
        this.socket = null;
        this.currentUsername = '';
        this.connectedUsers = [];
        this.isConnected = false;
        this.isMobile = window.innerWidth < 768;
        this.sidebarOpen = false;
        this.selectedFile = null;
        this.uploadAbortController = null;
        this.currentFileType = null;

        this.initializeElements();
        this.connectToServer();
        this.bindEvents();
        this.handleResize();
    }

    initializeElements() {
        this.elements = {
            fileUploadBtn: document.getElementById('file-upload-btn'),
            fileTypeModal: document.getElementById('file-type-modal'),
            cancelFileType: document.getElementById('cancel-file-type'),
            previewFiletype: document.getElementById('preview-filetype'),
            connectionStatus: document.getElementById('connection-status'),
            connectionText: document.getElementById('connection-text'),
            userCount: document.getElementById('user-count'),
            usersList: document.getElementById('users-list'),
            messagesContainer: document.getElementById('messages-container'),
            messageForm: document.getElementById('message-form'),
            messageInput: document.getElementById('message-input'),
            sendButton: document.getElementById('send-button'),
            charCounter: document.getElementById('char-counter'),
            usernameBtn: document.getElementById('username-btn'),
            usernameModal: document.getElementById('username-modal'),
            usernameForm: document.getElementById('username-form'),
            usernameInput: document.getElementById('username-input'),
            cancelUsername: document.getElementById('cancel-username'),
            sidebarToggle: document.getElementById('sidebar-toggle'),
            sidebar: document.getElementById('sidebar'),
            sidebarOverlay: document.getElementById('sidebar-overlay'),
            sidebarClose: document.getElementById('sidebar-close'),
            toastContainer: document.getElementById('toast-container'),
            fileInput: document.getElementById('file-input'),
            imagePreview: document.getElementById('image-preview'),
            previewImage: document.getElementById('preview-image'),
            previewFilename: document.getElementById('preview-filename'),
            previewFilesize: document.getElementById('preview-filesize'),
            removePreview: document.getElementById('remove-preview'),
            uploadProgress: document.getElementById('upload-progress'),
            uploadBar: document.getElementById('upload-bar'),
            uploadPercentage: document.getElementById('upload-percentage'),
            cancelUpload: document.getElementById('cancel-upload'),
            imageModal: document.getElementById('image-modal'),
            modalImage: document.getElementById('modal-image'),
            closeImageModal: document.getElementById('close-image-modal')
        };
    }

    connectToServer() {
        this.socket = io.connect(`${window.location.protocol}//${window.location.hostname}:8000`, {
            secure: true
        });

        this.socket.on('connect', () => {
            this.isConnected = true;
            this.updateConnectionStatus('connected');
            this.showToast('Connected to server', 'success');
        });

        this.socket.on('disconnect', () => {
            this.isConnected = false;
            this.updateConnectionStatus('disconnected');
            this.showToast('Disconnected from server', 'error');
        });

        this.socket.on('message', (data) => {
            this.addMessage(data);
            this.vibrate();
        });
        this.socket.on('file_message', (data) => {
            this.addFileMessage(data);
            this.vibrate();
        });
        this.socket.on('username_set', (username) => {
            this.currentUsername = username;
            this.updateConnectionStatus('connected');
            this.showToast(`Username set to ${username}`, 'success');
        });
        
        this.socket.on('user_list', (users) => {
            this.updateUsersList(users);
        });
        this.socket.on('toast_notification', (data) => {
            // The payload from the server contains the message and a type (info, warning, success, error)
            this.showToast(data.message, data.type);
        });
    }

    bindEvents() {
        // Form submission
        this.elements.messageForm.addEventListener('submit', (e) => {
            e.preventDefault();
            if (this.selectedFile) {
                this.uploadFile();
            } else {
                this.sendMessage();
            }
        });


        // File upload button
        this.elements.fileUploadBtn.addEventListener('click', () => {
            this.elements.fileTypeModal.classList.remove('hidden');
            this.elements.fileTypeModal.classList.add('flex');
        });

        // File type selection
        document.querySelectorAll('[data-type]').forEach(button => {
            button.addEventListener('click', (e) => {
                const type = e.currentTarget.getAttribute('data-type');
                this.currentFileType = type;
                this.elements.fileTypeModal.classList.add('hidden');
                this.elements.fileTypeModal.classList.remove('flex');

                // Set appropriate accept attribute based on type
                let accept = '';
                switch (type) {
                    case 'image':
                        accept = 'image/*';
                        break;
                    case 'document':
                        accept = '.pdf,.doc,.docx,.txt,.rtf,.odt';
                        break;
                    case 'video':
                        accept = 'video/*';
                        break;
                    default:
                        accept = '*';
                }

                this.elements.fileInput.setAttribute('accept', accept);
                this.elements.fileInput.click();
            });
        });

        // Cancel file type selection
        this.elements.cancelFileType.addEventListener('click', () => {
            this.elements.fileTypeModal.classList.add('hidden');
            this.elements.fileTypeModal.classList.remove('flex');
        });

        // File input change
        this.elements.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });

        // Message input events
        this.elements.messageInput.addEventListener('input', () => {
            this.updateCharCounter();
            this.updateSendButton();
        });

        this.elements.messageInput.addEventListener('focus', () => {
            this.elements.charCounter.classList.remove('hidden');
        });

        this.elements.messageInput.addEventListener('blur', () => {
            if (this.elements.messageInput.value.length === 0) {
                this.elements.charCounter.classList.add('hidden');
            }
        });

        // File input events
        this.elements.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });

        // Preview removal
        this.elements.removePreview.addEventListener('click', () => {
            this.clearFileSelection();
        });

        // Upload cancellation
        this.elements.cancelUpload.addEventListener('click', () => {
            this.cancelUpload();
        });

        // Image modal events
        this.elements.closeImageModal.addEventListener('click', () => {
            this.closeImageModal();
        });

        this.elements.imageModal.addEventListener('click', (e) => {
            if (e.target === this.elements.imageModal) {
                this.closeImageModal();
            }
        });

        // Username modal events
        this.elements.usernameBtn.addEventListener('click', () => {
            this.showUsernameModal();
        });

        this.elements.usernameForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.setUsername();
        });

        this.elements.cancelUsername.addEventListener('click', () => {
            this.hideUsernameModal();
        });

        // Sidebar events (mobile)
        this.elements.sidebarToggle.addEventListener('click', () => {
            this.toggleSidebar();
        });

        this.elements.sidebarClose.addEventListener('click', () => {
            this.closeSidebar();
        });

        this.elements.sidebarOverlay.addEventListener('click', () => {
            this.closeSidebar();
        });

        // Modal events
        this.elements.usernameModal.addEventListener('click', (e) => {
            if (e.target === this.elements.usernameModal) {
                this.hideUsernameModal();
            }
        });

        // Window events
        window.addEventListener('resize', () => {
            this.handleResize();
        });

        // Drag and drop events
        this.bindDragDropEvents();

        // Touch events for better mobile UX
        this.bindTouchEvents();

        // Keyboard shortcuts
        this.bindKeyboardShortcuts();
    }

    bindDragDropEvents() {
        const dropZones = [this.elements.messagesContainer, this.elements.messageInput];

        dropZones.forEach(zone => {
            zone.addEventListener('dragover', (e) => {
                e.preventDefault();
                zone.classList.add('drag-over');
            });

            zone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                zone.classList.remove('drag-over');
            });

            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.classList.remove('drag-over');

                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    this.handleFileSelect(files[0]);
                }
            });
        });
    }

    bindTouchEvents() {
        // Prevent zoom on double tap for buttons
        const buttons = document.querySelectorAll('button, a');
        buttons.forEach(button => {
            button.addEventListener('touchend', (e) => {
                e.preventDefault();
                button.click();
            }, { passive: false });
        });
    }

    bindKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Escape to close sidebar or modal
            if (e.key === 'Escape') {
                if (!this.elements.imageModal.classList.contains('hidden')) {
                    this.closeImageModal();
                } else if (this.sidebarOpen && this.isMobile) {
                    this.closeSidebar();
                } else if (!this.elements.usernameModal.classList.contains('hidden')) {
                    this.hideUsernameModal();
                }
            }

            // Ctrl/Cmd + Enter to send message
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                if (this.selectedFile) {
                    this.uploadFile();
                } else {
                    this.sendMessage();
                }
            }
        });
    }

    handleResize() {
        const wasMobile = this.isMobile;
        this.isMobile = window.innerWidth < 768;

        if (wasMobile !== this.isMobile) {
            if (!this.isMobile) {
                this.closeSidebar();
            }
        }
    }

    toggleSidebar() {
        if (this.sidebarOpen) {
            this.closeSidebar();
        } else {
            this.openSidebar();
        }
    }

    openSidebar() {
        this.sidebarOpen = true;
        this.elements.sidebar.classList.add('open');
        this.elements.sidebarOverlay.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }

    closeSidebar() {
        this.sidebarOpen = false;
        this.elements.sidebar.classList.remove('open');
        this.elements.sidebarOverlay.classList.add('hidden');
        document.body.style.overflow = '';
    }

    updateConnectionStatus(status) {
        const statusElement = this.elements.connectionStatus;
        const textElement = this.elements.connectionText;

        switch (status) {
            case 'connected':
                statusElement.className = 'w-2 h-2 bg-green-400 rounded-full animate-pulse';
                textElement.textContent = 'Connected';
                textElement.className = 'text-xs sm:text-sm text-green-400 hidden sm:inline';
                break;
            case 'disconnected':
                statusElement.className = 'w-2 h-2 bg-red-400 rounded-full';
                textElement.textContent = 'Disconnected';
                textElement.className = 'text-xs sm:text-sm text-red-400 hidden sm:inline';
                break;
            default:
                statusElement.className = 'w-2 h-2 bg-yellow-400 rounded-full animate-pulse';
                textElement.textContent = 'Connecting...';
                textElement.className = 'text-xs sm:text-sm text-yellow-400 hidden sm:inline';
        }
    }

    updateCharCounter() {
        const length = this.elements.messageInput.value.length;
        this.elements.charCounter.textContent = `${length}/500`;

        if (length > 400) {
            this.elements.charCounter.className = 'absolute right-12 top-1/2 transform -translate-y-1/2 text-xs text-red-400';
        } else if (length > 300) {
            this.elements.charCounter.className = 'absolute right-12 top-1/2 transform -translate-y-1/2 text-xs text-yellow-400';
        } else {
            this.elements.charCounter.className = 'absolute right-12 top-1/2 transform -translate-y-1/2 text-xs text-gray-500';
        }
    }

    updateSendButton() {
        const hasText = this.elements.messageInput.value.trim().length > 0;
        const hasFile = this.selectedFile !== null;
        this.elements.sendButton.disabled = (!hasText && !hasFile) || !this.isConnected;
    }

    updateUsersList(users) {
        this.connectedUsers = users;
        this.elements.userCount.textContent = users.length;

        const usersList = this.elements.usersList;
        usersList.innerHTML = '';

        users.forEach(user => {
            const userElement = this.createUserElement(user);
            usersList.appendChild(userElement);
        });
    }

    createUserElement(user) {
        const div = document.createElement('div');
        div.className = 'flex items-center p-2 sm:p-3 rounded-lg hover:bg-gray-700 active:bg-gray-600 transition-colors cursor-pointer no-select';

        const isCurrentUser = user.id === this.socket?.id;

        div.innerHTML = `
            <div class="w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center text-white font-medium mr-2 sm:mr-3 text-sm sm:text-base">
                ${user.username.charAt(0).toUpperCase()}
            </div>
            <div class="flex-1 min-w-0">
                <div class="flex items-center space-x-2">
                    <span class="font-medium text-white text-sm sm:text-base truncate">${this.escapeHtml(user.username)}</span>
                    ${isCurrentUser ? '<span class="text-xs bg-blue-500 text-white px-1.5 py-0.5 rounded-full flex-shrink-0">You</span>' : ''}
                </div>
                <div class="flex items-center space-x-1 mt-0.5 sm:mt-1">
                    <div class="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-green-400 rounded-full"></div>
                    <span class="text-xs text-gray-400">Online</span>
                </div>
            </div>
        `;

        return div;
    }

    handleFileSelect(file) {
        // Check file size (10MB limit)
        if (file.size > 10 * 1024 * 1024) {
            this.showToast('File too large (max 10MB)', 'error');
            return;
        }

        this.selectedFile = file;

        // Determine file type icon and label
        let fileType = 'other';
        let typeLabel = 'File';
        let iconColor = 'text-gray-400';

        if (file.type.startsWith('image/')) {
            fileType = 'image';
            typeLabel = 'Image';
            iconColor = 'text-blue-400';
        } else if (file.type.startsWith('video/')) {
            fileType = 'video';
            typeLabel = 'Video';
            iconColor = 'text-red-400';
        } else if (file.type.includes('pdf') ||
            file.type.includes('document') ||
            file.type.includes('text')) {
            fileType = 'document';
            typeLabel = 'Document';
            iconColor = 'text-green-400';
        }

        this.currentFileType = fileType;

        // Show preview
        if (fileType === 'image') {
            const reader = new FileReader();
            reader.onload = (e) => {
                this.elements.previewImage.src = e.target.result;
                this.elements.previewFilename.textContent = file.name;
                this.elements.previewFilesize.textContent = this.formatFileSize(file.size);
                this.elements.previewFiletype.textContent = typeLabel;
                this.elements.previewFiletype.className = `text-xs ${iconColor} mt-1`;
                this.elements.imagePreview.classList.remove('hidden');
                this.updateSendButton();
            };
            reader.readAsDataURL(file);
        } else {
            this.elements.previewImage.src = '';
            this.elements.previewFilename.textContent = file.name;
            this.elements.previewFilesize.textContent = this.formatFileSize(file.size);
            this.elements.previewFiletype.textContent = typeLabel;
            this.elements.previewFiletype.className = `text-xs ${iconColor} mt-1`;
            this.elements.imagePreview.classList.remove('hidden');
            this.updateSendButton();
        }
    }

    clearFileSelection() {
        this.selectedFile = null;
        this.elements.imagePreview.classList.add('hidden');
        this.elements.fileInput.value = '';
        this.updateSendButton();
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    uploadFile() {
        if (!this.selectedFile || !this.isConnected) return;

        const formData = new FormData();
        formData.append('file', this.selectedFile);
        formData.append('uploader_id', this.socket.id);

        // Only create one AbortController
        if (!this.uploadAbortController) {
            this.uploadAbortController = new AbortController();
        }

        // Show upload progress
        this.elements.uploadProgress.classList.remove('hidden');
        this.elements.uploadBar.style.width = '0%';
        this.elements.uploadPercentage.textContent = '0%';

        // Use fetch for the actual upload
        fetch('/upload', {
            method: 'POST',
            body: formData,
            signal: this.uploadAbortController.signal
        })
            .then(response => {
                if (!response.ok) {
                    // If the response is not OK, read the error message
                    return response.json().then(err => { throw new Error(err.detail || 'Upload failed') });
                }
                return response.json();
            })
            .then(data => {
                // ---- THIS IS THE NEW LOGIC ----
                // Manually add the message to the sender's UI upon success
                const messageData = {
                    sid: this.socket.id,
                    username: this.currentUsername || `User-${this.socket.id.substring(0, 8)}`,
                    type: 'file',
                    file: data.file, // 'data.file' is the metadata from the server response
                    timestamp: Date.now() / 1000
                };
                this.addMessage(messageData);
                // ---- END OF NEW LOGIC ----

                this.showToast('File uploaded successfully', 'success');
                this.clearFileSelection(); // Clear the file preview
            })
            .catch(error => {
                if (error.name !== 'AbortError') {
                    this.showToast(`File upload failed: ${error.message}`, 'error');
                    console.error('Upload error:', error);
                }
            })
            .finally(() => {
                this.elements.uploadProgress.classList.add('hidden');
                this.uploadAbortController = null;
                this.updateSendButton();
            });
        // Use XMLHttpRequest for progress tracking
        const xhr = new XMLHttpRequest();
        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                this.elements.uploadBar.style.width = `${percent}%`;
                this.elements.uploadPercentage.textContent = `${percent}%`;
            }
        };

        xhr.open('POST', '/upload', true);
        xhr.send(formData);
    }


    cancelUpload() {
        if (this.uploadAbortController) {
            this.uploadAbortController.abort();
            this.showToast('Upload cancelled', 'warning');
            this.clearFileSelection();
            this.elements.uploadProgress.classList.add('hidden');
        }
    }

    addMessage(data) {
        const messageElement = this.createMessageElement(data);
        this.elements.messagesContainer.appendChild(messageElement);
        this.scrollToBottom();
    }

    createMessageElement(data) {
        const messageElement = document.createElement('div');
        const isOwnMessage = data.sid === this.socket?.id;
        const timestamp = new Date(data.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        messageElement.className = `flex ${isOwnMessage ? 'justify-end' : 'justify-start'} animate-in slide-in-from-bottom-2 duration-300`;

        if (data.type === 'file') {
            // File message
            messageElement.innerHTML = `
            <div class="max-w-[280px] sm:max-w-xs lg:max-w-md ${isOwnMessage ? 'bg-blue-600' : 'bg-gray-700'} rounded-2xl px-3 sm:px-4 py-2 shadow-lg">
                ${!isOwnMessage ? `<div class="text-xs text-gray-300 mb-1 font-medium truncate">${this.escapeHtml(data.username)}</div>` : ''}
                <div class="flex items-center space-x-3">
                    <a href="/download/${data.file.id}" download="${this.escapeHtml(data.file.original_name)}" class="w-10 h-10 bg-gray-600 rounded-lg flex items-center justify-center">
                        <svg class="w-5 h-5 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                        </svg>
                    </a>
                    <div class="flex-1 min-w-0">
                        <div class="text-white text-sm truncate">${this.escapeHtml(data.file.original_name)}</div>
                        <div class="text-xs ${isOwnMessage ? 'text-blue-200' : 'text-gray-400'}">${this.formatFileSize(data.file.size)}</div>
                    </div>
                </div>
                <div class="text-xs ${isOwnMessage ? 'text-blue-200' : 'text-gray-400'} mt-1 text-right">
                    ${timestamp}
                </div>
            </div>
        `;
        } else {
            // Text message
            messageElement.innerHTML = `
                <div class="max-w-[280px] sm:max-w-xs lg:max-w-md ${isOwnMessage ? 'bg-blue-600' : 'bg-gray-700'} rounded-2xl px-3 sm:px-4 py-2 shadow-lg">
                    ${!isOwnMessage ? `<div class="text-xs text-gray-300 mb-1 font-medium truncate">${this.escapeHtml(data.username)}</div>` : ''}
                    <div class="text-white break-words text-sm sm:text-base">${this.escapeHtml(data.text)}</div>
                    <div class="text-xs ${isOwnMessage ? 'text-blue-200' : 'text-gray-400'} mt-1 text-right">
                        ${timestamp}
                    </div>
                </div>
            `;
        }

        return messageElement;
    }

    showUsernameModal() {
        this.elements.usernameModal.classList.remove('hidden');
        this.elements.usernameModal.classList.add('flex');
        setTimeout(() => {
            this.elements.usernameInput.focus();
        }, 100);
    }

    hideUsernameModal() {
        this.elements.usernameModal.classList.add('hidden');
        this.elements.usernameModal.classList.remove('flex');
    }

    loadInitialState() {
        // Load username from local storage if available
        const savedUsername = localStorage.getItem('localchat_username');
        if (savedUsername) {
            this.currentUsername = savedUsername;
            this.elements.usernameInput.value = savedUsername;
            this.updateUsernameButtonText();
        } else {
            // If no username is saved, prompt the user to set one
            setTimeout(() => this.showUsernameModal(), 500);
        }
    }

    onConnect() {
        this.isConnected = true;
        this.updateConnectionStatus('connected');
        this.showToast('Connected to server', 'success');
        // If a username is already set, send it to the server upon connection
        if (this.currentUsername) {
            this.socket.emit('set_username', this.currentUsername);
        }
        this.updateSendButton();
    }

    onDisconnect() {
        this.isConnected = false;
        this.updateConnectionStatus('disconnected');
        this.showToast('Disconnected. Trying to reconnect...', 'error');
        this.updateUsersList([]); // Clear user list on disconnect
        this.updateSendButton();
    }

    onMessage(data) {
        // Add message to the UI and provide feedback
        this.addMessage(data);
        this.vibrate();
        // If the message is not from the current user and notifications are allowed, show a notification
        if (data.sid !== this.socket?.id) {
            this.showBrowserNotification(data);
        }
    }

    onConnectError(err) {
        console.error('Connection Error:', err);
        this.showToast('Failed to connect to the server.', 'error');
        this.updateConnectionStatus('disconnected');
    }

    handleFormSubmit() {
        const message = this.elements.messageInput.value.trim();

        if (!this.isConnected) {
            this.showToast('Not connected to the server.', 'error');
            return;
        }

        if (this.selectedFile) {
            this.uploadFile();
        } else if (message) {
            this.sendMessage(message);
        }
    }

    sendMessage() {
        const message = this.elements.messageInput.value.trim();
        if (message && this.isConnected) {
            // First, optimistically add the message to the sender's own UI
            this.addMessage({
                text: message,
                username: this.currentUsername,
                sid: this.socket.id,
                timestamp: Date.now() / 1000,
                type: 'text'
            });

            // Second, send the message to the server for broadcasting to others
            this.socket.emit('message', { type: 'text', content: message });

            // Finally, clear the input and reset UI elements
            this.elements.messageInput.value = '';
            this.updateCharCounter();
            this.updateSendButton();
            this.elements.charCounter.classList.add('hidden');
            this.elements.messageInput.focus();
        }
    }

    setUsername() {
        const username = this.elements.usernameInput.value.trim();
        if (username) {
            if (username.length < 3) {
                this.showToast('Username must be at least 3 characters', 'warning');
                return;
            }
            this.currentUsername = username;
            localStorage.setItem('localchat_username', username); // Save to local storage
            if (this.isConnected) {
                this.socket.emit('set_username', username);
            }
            this.hideUsernameModal();
            this.showToast(`Username set to "${username}"`, 'success');
            this.updateUsernameButtonText();
        }
    }

    updateUsernameButtonText() {
        const usernameBtnSpan = this.elements.usernameBtn.querySelector('span');
        if (this.currentUsername) {
            usernameBtnSpan.textContent = this.currentUsername;
        } else {
            usernameBtnSpan.textContent = 'Set Username';
        }
    }

    showImageModal(imageUrl) {
        this.elements.modalImage.src = imageUrl;
        this.elements.imageModal.classList.remove('hidden');
        this.elements.imageModal.classList.add('flex');
        document.body.style.overflow = 'hidden';
    }

    closeImageModal() {
        this.elements.imageModal.classList.add('hidden');
        this.elements.imageModal.classList.remove('flex');
        document.body.style.overflow = '';
    }

    showToast(message, type = 'info', duration = 3000) {
        const toast = document.createElement('div');
        const colors = {
            'info': 'bg-blue-600',
            'success': 'bg-green-600',
            'error': 'bg-red-600',
            'warning': 'bg-yellow-600'
        };

        toast.className = `${colors[type]} text-white px-4 py-2 rounded-lg shadow-lg transform translate-x-full transition-transform duration-300 text-sm`;
        toast.textContent = message;

        this.elements.toastContainer.appendChild(toast);

        // Slide in
        setTimeout(() => {
            toast.classList.remove('translate-x-full');
        }, 10);

        // Slide out and remove
        setTimeout(() => {
            toast.classList.add('translate-x-full');
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, duration);
    }

    vibrate() {
        if ('vibrate' in navigator) {
            navigator.vibrate(50);
        }
    }

    scrollToBottom() {
        this.elements.messagesContainer.scrollTop = this.elements.messagesContainer.scrollHeight;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the chat app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChatApp();
});