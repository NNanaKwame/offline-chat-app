// Hybrid WebView permissions helper - works with React Native permission bridge
(function() {
    'use strict';
    
    console.log('🔧 Hybrid permission system loading...');
    
    // Store original getUserMedia
    const originalGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    
    // Permission state tracking
    let nativePermissionsGranted = false;
    let pendingRequests = [];
    
    // Override getUserMedia to work with native permission bridge
    navigator.mediaDevices.getUserMedia = function(constraints) {
        console.log('📱 getUserMedia intercepted:', constraints);
        
        if (nativePermissionsGranted) {
            console.log('✅ Using original getUserMedia with native permissions');
            return originalGetUserMedia(constraints);
        }
        
        console.log('🔄 Requesting native permissions...');
        
        // Send permission request to React Native
        if (window.ReactNativeWebView) {
            window.ReactNativeWebView.postMessage(JSON.stringify({
                type: 'request_permissions',
                constraints: constraints
            }));
        }
        
        // Return a promise that will be resolved when permissions are granted
        return new Promise((resolve, reject) => {
            pendingRequests.push({ resolve, reject, constraints });
            
            // Set a timeout to avoid hanging forever
            setTimeout(() => {
                const index = pendingRequests.findIndex(req => req.resolve === resolve);
                if (index !== -1) {
                    pendingRequests.splice(index, 1);
                    reject(new Error('Permission request timeout'));
                }
            }, 10000); // 10 second timeout
        });
    };
    
    // Listen for permission responses from React Native
    window.addEventListener('message', function(event) {
        try {
            const data = JSON.parse(event.data);
            console.log('📨 Received message from React Native:', data);
            
            if (data.type === 'permissions_granted') {
                console.log('✅ Native permissions granted!');
                nativePermissionsGranted = true;
                
                // Retry all pending getUserMedia requests
                const currentRequests = [...pendingRequests];
                pendingRequests = [];
                
                currentRequests.forEach(async ({ resolve, reject, constraints }) => {
                    try {
                        console.log('🔄 Retrying getUserMedia with constraints:', constraints);
                        const stream = await originalGetUserMedia(constraints);
                        console.log('✅ getUserMedia success with stream tracks:', 
                            stream.getTracks().map(t => `${t.kind}: ${t.label}`));
                        resolve(stream);
                    } catch (error) {
                        console.error('❌ getUserMedia failed even with permissions:', error);
                        console.error('Error details:', {
                            name: error.name,
                            message: error.message,
                            stack: error.stack
                        });
                        reject(error);
                    }
                });
                
            } else if (data.type === 'permissions_denied') {
                console.log('❌ Native permissions denied');
                
                // Reject all pending requests
                const currentRequests = [...pendingRequests];
                pendingRequests = [];
                
                currentRequests.forEach(({ reject }) => {
                    reject(new Error('Camera/microphone permissions denied'));
                });
            }
        } catch (error) {
            console.error('Error processing message from React Native:', error);
        }
    });
    
    // Enhanced permission checking
    if (navigator.permissions) {
        const checkPermission = async (name) => {
            try {
                const permission = await navigator.permissions.query({name: name});
                console.log(`${name} permission state:`, permission.state);
                return permission.state;
            } catch (error) {
                console.log(`Could not query ${name} permission:`, error);
                return 'unknown';
            }
        };
        
        // Check initial permission states
        checkPermission('camera');
        checkPermission('microphone');
    }
    
    console.log('🚀 Hybrid permission system ready');
})();