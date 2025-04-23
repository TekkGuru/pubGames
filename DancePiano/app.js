// Global variables
let video;
let backgroundFrame;
let isBackgroundCaptured = false;
let isProcessingActive = false;
let pianoKeys = [];
let synths = {};
let cvReady = false;
let cameraDimX = 1280;
let cameraDimY = 720;
let transformMatrix = null;  // Store globally


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
                width: cameraDimX,
                height: cameraDimY,
                frameRate: { ideal: 30, min: 15 }
            }
        });
        
        // Set video source
        video.srcObject = stream;
        
        // Wait for video to be ready
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                video.play();
            
                const width = video.videoWidth;
                const height = video.videoHeight;
            
                // Match canvas sizes to camera resolution
                ['videoCanvas', 'outputCanvas', 'processCanvas'].forEach(id => {
                    const canvas = document.getElementById(id);
                    canvas.width = width;
                    canvas.height = height;
                });
            
                console.log(`Camera size: ${width}x${height}`);
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
        const videoCanvas = document.getElementById('videoCanvas');
        const ctx = videoCanvas.getContext('2d');
        ctx.drawImage(video, 0, 0, videoCanvas.width, videoCanvas.height);

        backgroundFrame = cv.imread(videoCanvas);
        isBackgroundCaptured = true;
        statusText.textContent = 'Background Captured - Detecting Corners...';

        // 🔍 Detect white stickers as corners
        const corners = findPianoSurfaceCorners(backgroundFrame);
        if (corners.length !== 4) {
            alert(`Detected ${corners.length} corners. Need exactly 4.`);
            return;
        }

        // ✅ Sort corners (rough sorting TL, TR, BR, BL)
        corners.sort((a, b) => a.y - b.y);
        const top = corners.slice(0, 2).sort((a, b) => a.x - b.x);
        const bottom = corners.slice(2, 4).sort((a, b) => a.x - b.x);
        const orderedCorners = [top[0], top[1], bottom[1], bottom[0]];

        // 🔁 Get perspective transform matrix
        transformMatrix = computePerspectiveMatrix(orderedCorners);

        statusText.textContent = 'Corners Detected - Perspective Ready';
        cv.imshow('processCanvas', backgroundFrame);
        console.log('Background captured and transform matrix computed');

    } catch (error) {
        console.error('Error capturing background:', error);
        statusText.textContent = 'Error capturing background';
    }
}

function computePerspectiveMatrix(cornerPoints) {
    const srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
        cornerPoints[0].x, cornerPoints[0].y,
        cornerPoints[1].x, cornerPoints[1].y,
        cornerPoints[2].x, cornerPoints[2].y,
        cornerPoints[3].x, cornerPoints[3].y,
    ]);

    const rectWidth = 1280;
    const rectHeight = 720;
    const dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
        0, 0,
        rectWidth, 0,
        rectWidth, rectHeight,
        0, rectHeight
    ]);

    const transformMatrix = cv.getPerspectiveTransform(srcPts, dstPts);
    srcPts.delete(); dstPts.delete();
    return transformMatrix;
}

// Find white corners in image
function findWhiteCorners(frame) {
    const gray = new cv.Mat();
    cv.cvtColor(frame, gray, cv.COLOR_RGBA2GRAY);

    const thresholded = new cv.Mat();
    cv.threshold(gray, thresholded, 230, 255, cv.THRESH_BINARY);  // Bright white threshold

    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(thresholded, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    const corners = [];

    for (let i = 0; i < contours.size(); i++) {
        const contour = contours.get(i);
        const area = cv.contourArea(contour);
        if (area > 80) {  // Filter small spots
            const moments = cv.moments(contour);
            if (moments.m00 !== 0) {
                corners.push({
                    x: moments.m10 / moments.m00,
                    y: moments.m01 / moments.m00
                });
            }
        }
        contour.delete();
    }

    gray.delete(); thresholded.delete(); contours.delete(); hierarchy.delete();

    return corners;
}

// Find Red and green corners in the image.
function findColoredCorners_HSV(frame) {
    const hsv = new cv.Mat();
    cv.cvtColor(frame, hsv, cv.COLOR_RGBA2RGB);
    cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);

    const findColorCenters = (lowerHSV, upperHSV) => {
        const mask = new cv.Mat();
        cv.inRange(hsv, lowerHSV, upperHSV, mask);

        const contours = new cv.MatVector();
        const hierarchy = new cv.Mat();
        cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        const centers = [];
        for (let i = 0; i < contours.size(); i++) {
            const area = cv.contourArea(contours.get(i));
            if (area > 100) {
                const m = cv.moments(contours.get(i));
                if (m.m00 !== 0) {
                    centers.push({ x: m.m10 / m.m00, y: m.m01 / m.m00 });
                }
            }
        }

        mask.delete(); contours.delete(); hierarchy.delete();
        return centers;
    };

    // HSV ranges for red and green (adjust as needed)
    const redLower = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [0, 100, 100, 0]);
    const redUpper = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [10, 255, 255, 255]);
    const greenLower = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [40, 50, 50, 0]);
    const greenUpper = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [90, 255, 255, 255]);

    const redCorners = findColorCenters(redLower, redUpper);
    const greenCorners = findColorCenters(greenLower, greenUpper);

    redLower.delete(); redUpper.delete(); greenLower.delete(); greenUpper.delete(); hsv.delete();

    if (redCorners.length !== 2 || greenCorners.length !== 2) {
        console.warn(`Expected 2 red and 2 green markers. Found ${redCorners.length} red, ${greenCorners.length} green.`);
        return [];
    }

    // Sort each pair by x to ensure proper left/right
    redCorners.sort((a, b) => a.x - b.x);   // TL, TR
    greenCorners.sort((a, b) => a.x - b.x); // BL, BR

    return [redCorners[0], redCorners[1], greenCorners[1], greenCorners[0]]; // TL, TR, BR, BL
}

//Find piano surface corners
function findPianoSurfaceCorners(frame) {
    const gray = new cv.Mat();
    cv.cvtColor(frame, gray, cv.COLOR_RGBA2GRAY);

    const thresholded = new cv.Mat();
    cv.threshold(gray, thresholded, 200, 255, cv.THRESH_BINARY); // bright white threshold

    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(thresholded, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    let maxArea = 0;
    let maxContour = null;

    for (let i = 0; i < contours.size(); i++) {
        const contour = contours.get(i);
        const area = cv.contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
            maxContour = contour;
        }
    }

    let corners = [];
    if (maxContour) {
        const approx = new cv.Mat();
        const epsilon = 0.02 * cv.arcLength(maxContour, true);
        cv.approxPolyDP(maxContour, approx, epsilon, true);

        if (approx.rows === 4) {
            for (let i = 0; i < 4; i++) {
                const point = approx.data32S.slice(i * 2, i * 2 + 2);
                corners.push({ x: point[0], y: point[1] });
            }
        } else {
            console.warn(`Expected 4 corners, got ${approx.rows}`);
        }

        approx.delete();
    }

    gray.delete(); thresholded.delete(); contours.delete(); hierarchy.delete();

    return corners;
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
    }, 40); // ~30 FPS
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
        
        const warped = new cv.Mat();
        cv.warpPerspective(currentFrame, warped, transformMatrix, new cv.Size(cameraDimX, cameraDimY));

        // Use background subtraction to detect movement
        // Use warped instead of currentFrame
        const movementMask = backgroundSubtraction(warped);
        
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
// Background subtraction algorithm corrected for perspective and corner detection.
function backgroundSubtraction(currentFrame) {
    if (!transformMatrix || !backgroundFrame) {
        console.warn('Transform matrix or background frame missing');
        return new cv.Mat(); // Return empty
    }

    // 1. Warp both frames to top-down view
    const warpedCurrent = new cv.Mat();
    const warpedBackground = new cv.Mat();
    const warpSize = new cv.Size(1280, 720); // Or your piano area resolution

    cv.warpPerspective(currentFrame, warpedCurrent, transformMatrix, warpSize);
    cv.warpPerspective(backgroundFrame, warpedBackground, transformMatrix, warpSize);

    // 2. Convert to grayscale
    const grayCurrent = new cv.Mat();
    const grayBackground = new cv.Mat();
    cv.cvtColor(warpedCurrent, grayCurrent, cv.COLOR_RGBA2GRAY);
    cv.cvtColor(warpedBackground, grayBackground, cv.COLOR_RGBA2GRAY);

    // 3. Compute absolute difference
    const diffFrame = new cv.Mat();
    cv.absdiff(grayCurrent, grayBackground, diffFrame);

    // 4. Apply threshold to highlight motion
    const mask = new cv.Mat();
    const thresholdValue = 40; // Adjust as needed
    cv.threshold(diffFrame, mask, thresholdValue, 255, cv.THRESH_BINARY);

    // 5. Optional: clean up noise with morphology
    const kernel = cv.Mat.ones(5, 5, cv.CV_8U);
    const processedMask = new cv.Mat();
    cv.morphologyEx(mask, processedMask, cv.MORPH_OPEN, kernel);
    cv.morphologyEx(processedMask, processedMask, cv.MORPH_CLOSE, kernel);

    // 🧼 Cleanup
    warpedCurrent.delete(); warpedBackground.delete();
    grayCurrent.delete(); grayBackground.delete();
    diffFrame.delete(); mask.delete(); kernel.delete();

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
    
    const now = Date.now(); // Current time in ms

    // Check each foot position against piano keys
    footPositions.forEach(foot => {
        // Map normalized foot position to canvas coordinates
        const mappedX = foot.x * canvasWidth;
        const mappedY = foot.y * canvasHeight;
        
        pianoKeys.forEach(key => {
            // Check if foot is inside this key
            if (mappedX >= key.x && mappedX <= key.x + key.width &&
                mappedY >= key.y && mappedY <= key.y + key.height) {
                key.lastActiveTime = now;
            }
        });
    });
    
    // Play sounds for newly activated keys and stop sounds for deactivated keys
    // Now decide which keys should play or stop
    pianoKeys.forEach(key => {
        const timeSinceLastTouch = now - key.lastActiveTime;

        if (timeSinceLastTouch < 100) { // Still active within last 300ms
            key.isActive = true;
            if (!key.isPlaying) {
                synths[key.id].triggerAttack(key.note);
                key.isPlaying = true;
                console.log(`Starting note: ${key.note}`);
            }
        } else {
            key.isActive = false;
            if (key.isPlaying) {
                synths[key.id].triggerRelease();
                key.isPlaying = false;
                console.log(`Stopping note: ${key.note}`);
            }
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
    console.log(`Stopping note keyID: ${keyId}`);
}

// Check if OpenCV is already loaded (in case the onload event already fired)
if (window.cv && typeof cv !== 'undefined') {
    onOpenCVReady();
}