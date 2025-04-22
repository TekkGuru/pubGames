// Global variables
let video;
let backgroundFrame;
let isBackgroundCaptured = false;
let isProcessingActive = false;
let pianoKeys = [];
let synths = {};
let cvReady = false;

// Status elements
const statusText = document.getElementById('statusText');

// Function called when OpenCV.js is loaded
function onOpenCVReady() {
    console.log('OpenCV.js is ready');
    cvReady = true;
    statusText.textContent = 'OpenCV Ready - Setup Camera';
    setupCamera().then(() => {
        statusText.textContent = 'Camera Ready - Capture Background';
        setupPianoKeys();
        setupAudio();
        setupEventListeners();
    }).catch(error => {
        statusText.textContent = 'Camera Error: ' + error.message;
        console.error('Error setting up camera:', error);
    });
}

// Set up the camera
async function setupCamera() {
    try {
        // Get video element
        video = document.getElementById('video');
        
        // Request camera access
        const stream = await navigator.mediaDevices.getUserMedia({
            'video': {
                width: 640,
                height: 480,
                frameRate: { ideal: 30, min: 15 }
            }
        });
        
        // Set video source
        video.srcObject = stream;
        
        // Wait for video to be ready
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                video.play();
                resolve();
            };
        });
    } catch (error) {
        console.error('Error accessing camera:', error);
        throw error;
    }
}

// Define piano keys layout
function setupPianoKeys() {
    // Define 8 colorful keys (one octave)
    const keyColors = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#9400D3', '#FF00FF'];
    const noteNames = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C'];
    const notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'];
    
    // Get the canvas dimensions
    const outputCanvas = document.getElementById('outputCanvas');
    const keyWidth = outputCanvas.width / 8;
    const keyHeight = outputCanvas.height * 0.8;
    const keyY = outputCanvas.height * 0.1;
    
    // Create piano keys across the width of the canvas
    for (let i = 0; i < 8; i++) {
        pianoKeys.push({
            id: i,
            x: i * keyWidth,
            y: keyY,
            width: keyWidth,
            height: keyHeight,
            color: keyColors[i],
            noteName: noteNames[i],
            note: notes[i],
            isActive: false
        });
    }
    
    // Initial draw of keys
    drawPianoKeys();
}

// Set up audio synthesizers using Tone.js
function setupAudio() {
    // Create a synth for each piano key
    pianoKeys.forEach(key => {
        synths[key.id] = new Tone.Synth({
            oscillator: {
                type: 'triangle'
            },
            envelope: {
                attack: 0.005,
                decay: 0.1,
                sustain: 0.3,
                release: 1
            }
        }).toDestination();
    });
    
    // Initialize Tone.js
    Tone.start();
}

// Setup event listeners for buttons
function setupEventListeners() {
    // Capture background button
    document.getElementById('captureBackground').addEventListener('click', () => {
        captureBackground();
    });
    
    // Start piano button
    document.getElementById('startPiano').addEventListener('click', () => {
        if (!isBackgroundCaptured) {
            alert('Please capture background first!');
            return;
        }
        
        if (!isProcessingActive) {
            startProcessing();
            document.getElementById('startPiano').textContent = 'Stop Piano';
            statusText.textContent = 'Piano Active';
        } else {
            stopProcessing();
            document.getElementById('startPiano').textContent = 'Start Piano';
            statusText.textContent = 'Piano Stopped';
        }
    });
}

// Capture the background frame
function captureBackground() {
    if (!cvReady) {
        alert('OpenCV is not ready yet. Please wait.');
        return;
    }
    
    try {
        // Get the video canvas and draw the current frame
        const videoCanvas = document.getElementById('videoCanvas');
        const ctx = videoCanvas.getContext('2d');
        ctx.drawImage(video, 0, 0, videoCanvas.width, videoCanvas.height);
        
        // Capture this frame as background
        backgroundFrame = cv.imread(videoCanvas);
        
        isBackgroundCaptured = true;
        statusText.textContent = 'Background Captured - Ready to Start';
        
        // Show a preview of the background
        cv.imshow('processCanvas', backgroundFrame);
        
        console.log('Background captured');
    } catch (error) {
        console.error('Error capturing background:', error);
        statusText.textContent = 'Error capturing background';
    }
}

// Main processing loop variables
let processingInterval;
let playingKeys = [];

// Start the frame processing
function startProcessing() {
    if (isProcessingActive) return;
    
    isProcessingActive = true;
    
    // Set up processing loop
    processingInterval = setInterval(() => {
        try {
            processCurrentFrame();
        } catch (error) {
            console.error('Error processing frame:', error);
            stopProcessing();
            statusText.textContent = 'Processing Error: ' + error.message;
        }
    }, 60); // ~30 FPS
}

// Stop the frame processing
function stopProcessing() {
    if (processingInterval) {
        clearInterval(processingInterval);
    }

    // Reset all keys and stop all sounds
    pianoKeys.forEach(key => {
        key.isActive = false;
        stopSound(key.id); // Ensure all sounds are stopped
    });

    playingKeys = []; // Clear the playing keys array
    drawPianoKeys(); // Redraw the piano keys to reflect inactive state

    isProcessingActive = false; // Update the processing state
    console.log('Processing stopped'); // Debug log
}

// Process the current video frame
function processCurrentFrame() {
    // Draw the current video frame to canvas
    const videoCanvas = document.getElementById('videoCanvas');
    const ctxVideo = videoCanvas.getContext('2d');
    ctxVideo.drawImage(video, 0, 0, videoCanvas.width, videoCanvas.height);
    try {
        // Process the frame using OpenCV.js
        const currentFrame = cv.imread(videoCanvas);
        
        // Use background subtraction to detect movement
        const movementMask = backgroundSubtraction(currentFrame);
        
        // Show the movement mask on process canvas
        cv.imshow('processCanvas', movementMask);
        
        // Detect blobs (feet) in the movement mask
        const footPositions = detectBlobs(movementMask);
        
        // Update piano key states based on detected feet
        updateKeyStates(footPositions);
        
        // Draw the piano keys with updated states
        drawPianoKeys();
        
        // Clean up OpenCV objects
        currentFrame.delete();
        movementMask.delete();
    } catch (error) {
        console.error('Frame processing error:', error);
    }
}
// Background subtraction algorithm
function backgroundSubtraction(currentFrame) {
    // Convert frames to grayscale for easier processing
    const grayBackground = new cv.Mat();
    const grayCurrentFrame = new cv.Mat();
    cv.cvtColor(backgroundFrame, grayBackground, cv.COLOR_RGBA2GRAY);
    cv.cvtColor(currentFrame, grayCurrentFrame, cv.COLOR_RGBA2GRAY);
    
    // Calculate absolute difference between current frame and background
    const diffFrame = new cv.Mat();
    cv.absdiff(grayBackground, grayCurrentFrame, diffFrame);
    
    // Apply threshold to get binary mask of movement
    const thresholdValue = 100; // Adjust based on lighting conditions
    const mask = new cv.Mat();
    cv.threshold(diffFrame, mask, thresholdValue, 255, cv.THRESH_BINARY);
    
    // Apply morphological operations to remove noise
    const kernel = cv.Mat.ones(5, 5, cv.CV_8U);
    const processedMask = new cv.Mat();
    
    // Dilate to fill in holes
    cv.dilate(mask, processedMask, kernel);
    
    // Clean up
    grayBackground.delete();
    grayCurrentFrame.delete();
    diffFrame.delete();
    mask.delete();
    kernel.delete();
    
    return processedMask;
}

// Blob detection to find feet
function detectBlobs(mask) {
    // Find contours in the mask
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    
    // Array to store foot positions
    const footPositions = [];
    
    // Minimum area to be considered a foot (to filter out noise)
    const minFootArea = 300; // Adjust based on your setup
    
    // Get video canvas dimensions for normalization
    const videoCanvas = document.getElementById('videoCanvas');
    const videoWidth = videoCanvas.width;
    const videoHeight = videoCanvas.height;
    
    // Process each contour
    for (let i = 0; i < contours.size(); i++) {
        const contour = contours.get(i);
        const area = cv.contourArea(contour);
        if (area > minFootArea) {
            const moments = cv.moments(contour);
            if (moments.m00 > 0) {
                const centerX = moments.m10 / moments.m00;
                const centerY = moments.m01 / moments.m00;
                footPositions.push({
                    x: centerX / videoWidth,
                    y: centerY / videoHeight
                });
            }
        }
        contour.delete(); // ✅ This is critical
    }
    
    
    // Clean up
    contours.delete();
    hierarchy.delete();
    
    return footPositions;
}

// Update which piano keys are being pressed
function updateKeyStates(footPositions) {
    // Get output canvas dimensions
    const outputCanvas = document.getElementById('outputCanvas');
    const canvasWidth = outputCanvas.width;
    const canvasHeight = outputCanvas.height;
    
    // Reset all keys to inactive
    pianoKeys.forEach(key => {
        key.isActive = false;
    });
    
    // Check each foot position against piano keys
    footPositions.forEach(foot => {
        // Map normalized foot position to canvas coordinates
        const mappedX = foot.x * canvasWidth;
        const mappedY = foot.y * canvasHeight;
        
        pianoKeys.forEach(key => {
            // Check if foot is inside this key
            if (mappedX >= key.x && mappedX <= key.x + key.width &&
                mappedY >= key.y && mappedY <= key.y + key.height) {
                key.isActive = true;
            }
        });
    });
    
    // Play sounds for newly activated keys and stop sounds for deactivated keys
    pianoKeys.forEach(key => {
        if (key.isActive) {
            playSound(key.id);
        }
    });
}

// Draw the piano keys
function drawPianoKeys() {
    const outputCanvas = document.getElementById('outputCanvas');
    const ctx = outputCanvas.getContext('2d');
    
    // Clear the canvas
    ctx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    
    // Draw each key
    pianoKeys.forEach(key => {
        // Set color based on active state
        ctx.fillStyle = key.color;
        ctx.globalAlpha = key.isActive ? 1.0 : 0.7;
        
        // Draw the key
        ctx.fillRect(key.x, key.y, key.width, key.height);
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.strokeRect(key.x, key.y, key.width, key.height);
        
        // Draw note name
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '24px Arial';
        ctx.textAlign = 'center';
        ctx.globalAlpha = 1.0;
        ctx.fillText(key.noteName, key.x + key.width/2, key.y + key.height/2);
    });
}

// Play sound for a key
function playSound(keyId) {
    const key = pianoKeys.find(k => k.id === keyId);
    if (key) {
        synths[keyId].triggerAttackRelease(key.note, "0.5");
        console.log(`Playing note: ${key.note}`);
    }
}

// Stop sound for a key
function stopSound(keyId) {
    synths[keyId].triggerRelease();
    console.log(`Stopping note: ${key.note}`);
}

// Check if OpenCV is already loaded (in case the onload event already fired)
if (window.cv && typeof cv !== 'undefined') {
    onOpenCVReady();
}