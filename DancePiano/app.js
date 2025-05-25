

// Global variables
let video; // Will store the HTMLVideoElement
let backgroundFrame; // Unwarped background frame from camera
let warpedBackgroundFrame; // Warped background frame (perspective corrected)
let isBackgroundCaptured = false;
let isProcessingActive = false;
let pianoKeys = [];
let synths = {};
let cvReady = false;
let cameraDimX = 1280;
let cameraDimY = 720;
let transformMatrix = null;
let projectionMode = true; // Flag for projection mode (not fully used in this logic but kept)

// Status elements
const statusText = document.getElementById('statusText');

// Function called when OpenCV.js is loaded
function onOpenCVReady() {
    console.log('OpenCV.js is ready');
    cvReady = true;
    statusText.textContent = 'OpenCV Ready - Setup Camera';
    setupCamera().then(() => {
        statusText.textContent = 'Camera Ready - Create Piano';
        setupPianoKeys();
        setupAudio();
        setupEventListeners();
        drawPianoKeys(); // Draw initial piano projection
    }).catch(error => {
        statusText.textContent = 'Camera Error: ' + error.message;
        console.error('Error setting up camera or subsequent operations:', error);
    });
}

// Set up the camera
async function setupCamera() {
    try {
        console.log("setupCamera: Entered function");
        const videoElement = document.getElementById('video');
        const videoCanvas = document.getElementById('videoCanvas');
        const processCanvas = document.getElementById('processCanvas');
        const outputCanvas = document.getElementById('outputCanvas');

        console.log("setupCamera: Got video element:", videoElement);
        if (!videoElement || !videoCanvas || !processCanvas || !outputCanvas) {
            throw new Error("One or more canvas/video elements not found in DOM.");
        }

        videoCanvas.width = cameraDimX;
        videoCanvas.height = cameraDimY;
        processCanvas.width = cameraDimX;
        processCanvas.height = cameraDimY;
        outputCanvas.width = cameraDimX;
        outputCanvas.height = cameraDimY;
        console.log("setupCamera: Canvas dimensions set.");

        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: cameraDimX },
                height: { ideal: cameraDimY }
            },
            audio: false
        });
        console.log("setupCamera: Got user media stream:", stream);

        videoElement.srcObject = stream;
        video = videoElement; // Assign to global variable
        console.log("setupCamera: Assigned stream to video element. Waiting for onloadedmetadata...");

        return new Promise((resolve, reject) => {
            const timeoutDuration = 10000; // 10 seconds timeout for metadata
            const metadataTimeout = setTimeout(() => {
                console.warn(`setupCamera: onloadedmetadata timed out after ${timeoutDuration}ms.`);
                reject(new Error("Camera metadata failed to load in time. Check camera connection and permissions."));
            }, timeoutDuration);

            video.onloadedmetadata = () => {
                clearTimeout(metadataTimeout); // Clear the timeout
                console.log("setupCamera: onloadedmetadata event fired.");
                video.play().then(() => {
                    console.log("setupCamera: Video playing.");
                    // Start continuous drawing of video to videoCanvas
                    setInterval(() => {
                        const ctx = videoCanvas.getContext('2d');
                        // Ensure video is playing and has data before drawing
                        if (video && video.readyState >= video.HAVE_CURRENT_DATA && !video.paused && !video.ended) {
                             ctx.drawImage(video, 0, 0, videoCanvas.width, videoCanvas.height);
                        }
                    }, 1000 / 30); // 30 FPS
                    console.log("setupCamera: Resolving promise.");
                    resolve(video);
                }).catch(playError => {
                    clearTimeout(metadataTimeout);
                    console.error("setupCamera: Error playing video:", playError);
                    reject(new Error('Failed to play video: ' + playError.message));
                });
            };

            video.onerror = (e) => { // Add generic error handler for video element
                clearTimeout(metadataTimeout);
                console.error("setupCamera: Video element error:", e);
                reject(new Error("Video element encountered an error. Check browser console for details."));
            };
        });
    } catch (error) { // Catches getUserMedia errors or other synchronous errors in setupCamera
        console.error('setupCamera: Error during initial setup:', error);
        // Attempt to update statusText, ensure it exists
        if (statusText) {
             statusText.textContent = 'Camera Init Error: ' + error.message;
        }
        throw error; // Re-throw to be caught by onOpenCVReady's catch block
    }
}

// Define piano keys layout
function setupPianoKeys() {
    // New color palette: Yellows, Oranges, Greens
    const keyColors = [
        '#FFD700', // Gold (Yellow)
        '#FFA500', // Orange
        '#32CD32', // LimeGreen
        '#FFD700', // Gold (Yellow)
        '#FFA500', // Orange
        '#32CD32', // LimeGreen
        '#FFD700', // Gold (Yellow)
        '#FFA500'  // Orange
    ];
    const noteNames = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C'];
    const notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'];
    
    const outputCanvas = document.getElementById('outputCanvas');
    if (!outputCanvas) {
        console.error("setupPianoKeys: outputCanvas not found!");
        return;
    }
    const keyWidth = outputCanvas.width / 8;
    const keyHeight = outputCanvas.height * 0.8; // 80% of canvas height for keys
    const keyY = outputCanvas.height * 0.1; // 10% margin from top

    pianoKeys = []; // Clear previous keys if any
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
            isActive: false,
            isPlaying: false,
            lastActiveTime: 0,
            hsv: getHSVRangeForColor(keyColors[i]) // Initial HSV guess
        });
    }
    drawPianoKeys();
    console.log("setupPianoKeys: Piano keys configured with new color palette.");
}

// Helper function to convert hex color to approximate HSV range
function getHSVRangeForColor(hexColor) {
    const r = parseInt(hexColor.slice(1, 3), 16) / 255;
    const g = parseInt(hexColor.slice(3, 5), 16) / 255;
    const b = parseInt(hexColor.slice(5, 7), 16) / 255;
    
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const delta = max - min;
    let h, s, v = max;

    if (delta === 0) h = 0;
    else if (max === r) h = ((g - b) / delta) % 6;
    else if (max === g) h = (b - r) / delta + 2;
    else h = (r - g) / delta + 4;
    
    h = Math.round(h * 60);
    if (h < 0) h += 360;
    
    s = max === 0 ? 0 : delta / max;
    
    // Adjust HSV range detection parameters: more sensitive Saturation and Value, wider Hue range
    const hueRange = 20; // Wider hue tolerance
    const saturationRange = 70; // Wider saturation tolerance
    const valueRange = 70; // Wider value tolerance
    
    return {
        lower: [
            Math.max(0, Math.round(h / 2) - hueRange), 
            Math.max(0, Math.round(s * 255) - saturationRange), 
            Math.max(0, Math.round(v * 255) - valueRange)
        ],
        upper: [
            Math.min(179, Math.round(h / 2) + hueRange), 
            Math.min(255, Math.round(s * 255) + saturationRange), 
            Math.min(255, Math.round(v * 255) + valueRange)
        ]
    };
}

// Set up audio synthesizers
function setupAudio() {
    pianoKeys.forEach(key => {
        synths[key.id] = new Tone.Synth({
            oscillator: { type: 'triangle' },
            envelope: { attack: 0.005, decay: 0.1, sustain: 0.3, release: 1 }
        }).toDestination();
    });
    Tone.start().then(() => {
        console.log("setupAudio: Tone.js audio context started.");
    }).catch(err => {
        console.error("setupAudio: Error starting Tone.js audio context:", err);
        statusText.textContent = "Audio Error: Could not start audio context.";
    });
}

// Setup event listeners
function setupEventListeners() {
    document.getElementById('captureBackground').addEventListener('click', captureBackground);
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
    document.getElementById('calibrateColors').addEventListener('click', calibrateColorDetection);
    console.log("setupEventListeners: Event listeners attached.");
}

// Capture the background frame
async function captureBackground() {
    if (!cvReady) {
        alert('OpenCV is not ready yet. Please wait.');
        return;
    }
    if (!video || video.readyState < video.HAVE_ENOUGH_DATA) {
        alert('Video stream not ready. Please wait and try again.');
        return;
    }

    try {
        const videoCanvas = document.getElementById('videoCanvas');
        if (!videoCanvas) {
            throw new Error("captureBackground: videoCanvas not found.");
        }

        if (warpedBackgroundFrame) {
            warpedBackgroundFrame.delete();
            warpedBackgroundFrame = null;
        }
        if (backgroundFrame) {
            backgroundFrame.delete(); 
        }

        backgroundFrame = cv.imread(videoCanvas); 
        isBackgroundCaptured = true;
        statusText.textContent = 'Background Captured - Detecting Corners...';
        console.log("captureBackground: Background frame captured.");
        
        detectProjectedPianoCorners(backgroundFrame);

    } catch (error) {
        console.error('Error capturing background:', error);
        statusText.textContent = 'Error capturing background: ' + error.message;
        isBackgroundCaptured = false; 
    }
}

// Detect corners and compute transform matrix
function detectProjectedPianoCorners(unwarpedBgFrame) { 
    try {
        const corners = findProjectionCorners(unwarpedBgFrame);
        
        if (corners.length !== 4) {
            alert(`Detected ${corners.length} corners. Need exactly 4. Try adjusting lighting or projection area.`);
            isBackgroundCaptured = false; 
            statusText.textContent = 'Corner detection failed - try again';
            console.warn("detectProjectedPianoCorners: Incorrect number of corners detected:", corners.length);
            return;
        }

        const sortedCorners = sortCorners(corners);
        transformMatrix = computePerspectiveMatrix(sortedCorners);
        
        if (transformMatrix && backgroundFrame) { 
            if (warpedBackgroundFrame) warpedBackgroundFrame.delete(); 
            warpedBackgroundFrame = new cv.Mat();
            cv.warpPerspective(backgroundFrame, warpedBackgroundFrame, transformMatrix, new cv.Size(cameraDimX, cameraDimY));
            console.log('detectProjectedPianoCorners: Warped background frame created.');
        } else {
            console.error("detectProjectedPianoCorners: Failed to create warped background frame. Transform matrix or backgroundFrame missing.");
            isBackgroundCaptured = false;
            statusText.textContent = 'Error creating warped background. Recapture.';
            return;
        }

        const cornerDisplay = unwarpedBgFrame.clone();
        for (let i = 0; i < sortedCorners.length; i++) {
            cv.circle(cornerDisplay, new cv.Point(sortedCorners[i].x, sortedCorners[i].y), 10, [255, 0, 0, 255], -1);
        }
        cv.imshow('processCanvas', cornerDisplay);
        cornerDisplay.delete();

        statusText.textContent = 'Corners Detected - Calibrate Colors or Start Piano';
        console.log('detectProjectedPianoCorners: Background captured, transform matrix computed, warped background created.');

    } catch (error) {
        console.error('Error detecting corners or warping background:', error);
        statusText.textContent = 'Error in corner/warp: ' + error.message;
        isBackgroundCaptured = false; 
        if (transformMatrix) { transformMatrix.delete(); transformMatrix = null; }
        if (warpedBackgroundFrame) { warpedBackgroundFrame.delete(); warpedBackgroundFrame = null; }
    }
}

// Sort corners: TL, TR, BR, BL
function sortCorners(corners) {
    corners.sort((a, b) => a.y - b.y);
    const topCorners = corners.slice(0, 2).sort((a, b) => a.x - b.x); 
    const bottomCorners = corners.slice(2, 4).sort((a, b) => a.x - b.x); 
    return [topCorners[0], topCorners[1], bottomCorners[1], bottomCorners[0]];
}


// Find corners of the projected piano (heuristic, might need tuning)
function findProjectionCorners(frame) {
    const gray = new cv.Mat();
    cv.cvtColor(frame, gray, cv.COLOR_RGBA2GRAY);
    const blurred = new cv.Mat();
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
    const edges = new cv.Mat();
    cv.Canny(blurred, edges, 50, 150, 3, false); 
    
    const kernel = cv.Mat.ones(5, 5, cv.CV_8U); 
    const dilated = new cv.Mat();
    cv.dilate(edges, dilated, kernel);
    
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(dilated, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    
    let maxArea = 0;
    let pianoContour = null;
    for (let i = 0; i < contours.size(); i++) {
        const contour = contours.get(i);
        const area = cv.contourArea(contour, false); 
        if (area > maxArea) {
            if (pianoContour) pianoContour.delete(); // Delete previous max contour before cloning new one
            maxArea = area;
            pianoContour = contour.clone(); 
        }
        // Do not delete contour here if it's cloned, it will be deleted by contours.delete() later
        // If not cloning, and not taking ownership, contour.delete() would be needed if get(i) returns a new Mat
    }
    // Note: The MatVector 'contours' owns the Mats it contains unless explicitly released.
    // If pianoContour is a clone, it needs separate deletion.
    
    const corners = [];
    if (pianoContour && maxArea > 1000) { 
        const approx = new cv.Mat();
        const peri = cv.arcLength(pianoContour, true);
        cv.approxPolyDP(pianoContour, approx, 0.02 * peri, true); 
        
        if (approx.rows === 4) { 
            for (let i = 0; i < approx.rows; i++) {
                corners.push({ x: approx.data32S[i * 2], y: approx.data32S[i * 2 + 1] });
            }
        } else if (approx.rows > 4) { 
            console.warn(`Approximation found ${approx.rows} points, expected 4. Using bounding rectangle corners.`);
            const rect = cv.boundingRect(pianoContour);
            corners.push({x: rect.x, y: rect.y}); 
            corners.push({x: rect.x + rect.width, y: rect.y}); 
            corners.push({x: rect.x + rect.width, y: rect.y + rect.height}); 
            corners.push({x: rect.x, y: rect.y + rect.height}); 
        }
        
        approx.delete();
    }
    
    if (pianoContour) pianoContour.delete(); // Clean up cloned contour
    
    gray.delete(); blurred.delete(); edges.delete(); dilated.delete(); kernel.delete();
    contours.delete(); hierarchy.delete();
    
    return corners;
}

// Compute perspective transform matrix
function computePerspectiveMatrix(cornerPoints) {
    const srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
        cornerPoints[0].x, cornerPoints[0].y, cornerPoints[1].x, cornerPoints[1].y,
        cornerPoints[2].x, cornerPoints[2].y, cornerPoints[3].x, cornerPoints[3].y,
    ]);
    const dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
        0, 0, cameraDimX, 0, cameraDimX, cameraDimY, 0, cameraDimY
    ]);
    const M = cv.getPerspectiveTransform(srcPts, dstPts);
    srcPts.delete(); dstPts.delete();
    return M;
}

// Color calibration
function calibrateColorDetection() {
    if (!isBackgroundCaptured || !transformMatrix) {
        alert('Please capture background and detect corners first!');
        return;
    }
    if (!video || video.readyState < video.HAVE_ENOUGH_DATA) {
        alert('Video not ready for calibration.');
        return;
    }

    statusText.textContent = 'Calibrating Color Detection...';
    try {
        const videoCanvas = document.getElementById('videoCanvas');
        if (!videoCanvas) throw new Error("calibrateColorDetection: videoCanvas not found.");
        const frame = cv.imread(videoCanvas); 
        
        const warpedFrame = new cv.Mat();
        cv.warpPerspective(frame, warpedFrame, transformMatrix, new cv.Size(cameraDimX, cameraDimY));
        
        const hsv = new cv.Mat();
        cv.cvtColor(warpedFrame, hsv, cv.COLOR_RGBA2RGB); 
        cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
        
        pianoKeys.forEach((key, index) => {
            const roiRect = new cv.Rect(
                Math.max(0, Math.floor(key.x + key.width * 0.25)), 
                Math.max(0, Math.floor(key.y + key.height * 0.25)),
                Math.floor(key.width * 0.5), 
                Math.floor(key.height * 0.5)
            );
            if (roiRect.width <= 0 || roiRect.height <= 0 ||
                roiRect.x + roiRect.width > hsv.cols ||
                roiRect.y + roiRect.height > hsv.rows ||
                roiRect.x < 0 || roiRect.y < 0) {
                console.warn(`Key ${index} ROI is invalid or out of bounds for calibration. ROI: x=${roiRect.x}, y=${roiRect.y}, w=${roiRect.width}, h=${roiRect.height}. HSV dims: ${hsv.cols}x${hsv.rows}`);
                return; 
            }
            const roi = hsv.roi(roiRect);
            const meanHsv = cv.mean(roi); 
            
            // Fine-tune HSV ranges during calibration
            const hueCalibRange = 15; // More precise hue for calibration
            const satCalibRange = 50; // Keep saturation/value broader for lighting variations
            const valCalibRange = 50;

            key.hsv = {
                lower: [
                    Math.max(0, Math.round(meanHsv[0]) - hueCalibRange), 
                    Math.max(0, Math.round(meanHsv[1]) - satCalibRange), 
                    Math.max(0, Math.round(meanHsv[2]) - valCalibRange)  
                ],
                upper: [
                    Math.min(179, Math.round(meanHsv[0]) + hueCalibRange),
                    Math.min(255, Math.round(meanHsv[1]) + satCalibRange),
                    Math.min(255, Math.round(meanHsv[2]) + valCalibRange)
                ]
            };
            console.log(`Key ${index} (${key.noteName}) calibrated HSV: L=[${key.hsv.lower.map(v=>v.toFixed(0)).join(',')}] U=[${key.hsv.upper.map(v=>v.toFixed(0)).join(',')}] from Mean=[${meanHsv.map(v=>v.toFixed(0)).join(',')}]`);
            roi.delete();
        });
        
        cv.imshow('processCanvas', warpedFrame); 
        frame.delete(); warpedFrame.delete(); hsv.delete();
        statusText.textContent = 'Color Calibration Complete';
        console.log("calibrateColorDetection: Calibration complete.");
    } catch (error) {
        console.error('Calibration error:', error);
        statusText.textContent = 'Calibration Error: ' + error.message;
    }
}

// Main processing loop variables
let processingInterval;

// Start frame processing
function startProcessing() {
    if (isProcessingActive) return;
    if (!transformMatrix) {
        alert("Transform matrix not available. Capture background first.");
        return;
    }
    if (!warpedBackgroundFrame) { 
        alert("Warped background frame not ready. Capture background and ensure corners are detected.");
        return;
    }
    
    // Check if default HSV values are still present, suggesting calibration might be needed.
    let needsCalibrationWarning = false;
    for (const key of pianoKeys) {
        const defaultHsv = getHSVRangeForColor(key.color);
        if (key.hsv.lower.join(',') === defaultHsv.lower.join(',') && 
            key.hsv.upper.join(',') === defaultHsv.upper.join(',')) {
            needsCalibrationWarning = true;
            break;
        }
    }
    if (needsCalibrationWarning) {
        if (!confirm("Colors might not be optimally calibrated for the current lighting. Performance may vary. Continue anyway?")) {
            return;
        }
    }


    isProcessingActive = true;
    processingInterval = setInterval(() => {
        try {
            processCurrentFrame();
        } catch (error) {
            console.error('Error processing frame:', error);
            stopProcessing(); 
            statusText.textContent = 'Processing Error: ' + error.message;
        }
    }, 50); 
    console.log("startProcessing: Processing started.");
}

// Stop frame processing
function stopProcessing() {
    if (processingInterval) {
        clearInterval(processingInterval);
        processingInterval = null;
    }
    isProcessingActive = false;
    pianoKeys.forEach(key => {
        if (key.isPlaying) {
            synths[key.id].triggerRelease();
            key.isPlaying = false;
        }
        key.isActive = false;
    });
    drawPianoKeys(); 
    statusText.textContent = 'Piano Stopped';
    console.log('stopProcessing: Processing stopped');
}

// Process current video frame
function processCurrentFrame() {
    if (!video || video.readyState < video.HAVE_ENOUGH_DATA || !transformMatrix || !cvReady) {
        // console.warn('Video not ready or OpenCV/transform not initialized for processing.'); // Too noisy
        return;
    }

    const videoCanvas = document.getElementById('videoCanvas');
    if (!videoCanvas) {
        console.error("processCurrentFrame: videoCanvas not found.");
        stopProcessing();
        statusText.textContent = 'Error: videoCanvas missing.';
        return;
    }
    const currentFrameMat = cv.imread(videoCanvas); 
    
    const warpedCurrentFrame = new cv.Mat();
    cv.warpPerspective(currentFrameMat, warpedCurrentFrame, transformMatrix, new cv.Size(cameraDimX, cameraDimY));
    
    cv.imshow('processCanvas', warpedCurrentFrame); 
    
    const activeRegions = detectActivePianoRegions(warpedCurrentFrame);
    updateKeyStates(activeRegions);
    drawPianoKeys();
    
    currentFrameMat.delete();
    warpedCurrentFrame.delete();
}

// Detect active piano regions (occlusion-based)
function detectActivePianoRegions(warpedCurrentFrame) {
    if (!warpedBackgroundFrame) { 
        console.warn("Warped background frame not available for detection.");
        return [];
    }

    const hsvCurrent = new cv.Mat();
    cv.cvtColor(warpedCurrentFrame, hsvCurrent, cv.COLOR_RGBA2RGB); 
    cv.cvtColor(hsvCurrent, hsvCurrent, cv.COLOR_RGB2HSV);

    const activeRegions = [];
    const occlusionThreshold = 0.5; 

    pianoKeys.forEach(key => {
        const keyROI_Rect = new cv.Rect(
            Math.floor(key.x), Math.floor(key.y),
            Math.floor(key.width), Math.floor(key.height)
        );
        
        if (keyROI_Rect.width <= 0 || keyROI_Rect.height <= 0 || 
            keyROI_Rect.x + keyROI_Rect.width > hsvCurrent.cols ||
            keyROI_Rect.y + keyROI_Rect.height > hsvCurrent.rows ||
            keyROI_Rect.x < 0 || keyROI_Rect.y < 0 ) {
            // console.warn(`Key ${key.id} ROI is invalid or out of bounds for detection.`);
            return; 
        }

        let hsvKeyRegionCurrent = null;
        let lowerBound = null; 
        let upperBound = null; 
        let keyColorMask = null;

        try {
            hsvKeyRegionCurrent = hsvCurrent.roi(keyROI_Rect);
            
            lowerBound = new cv.Mat(hsvKeyRegionCurrent.rows, hsvKeyRegionCurrent.cols, hsvKeyRegionCurrent.type(), 
                                      [key.hsv.lower[0], key.hsv.lower[1], key.hsv.lower[2], 0]);
            upperBound = new cv.Mat(hsvKeyRegionCurrent.rows, hsvKeyRegionCurrent.cols, hsvKeyRegionCurrent.type(),
                                      [key.hsv.upper[0], key.hsv.upper[1], key.hsv.upper[2], 255]);

            keyColorMask = new cv.Mat();
            cv.inRange(hsvKeyRegionCurrent, lowerBound, upperBound, keyColorMask);
            
            const detectedKeyColorPixels = cv.countNonZero(keyColorMask);
            const totalKeyAreaPixels = key.width * key.height;
            if (totalKeyAreaPixels === 0) return;

            const keyColorCoverage = detectedKeyColorPixels / totalKeyAreaPixels;

            if (keyColorCoverage < (1.0 - occlusionThreshold)) { 
                activeRegions.push({
                    keyId: key.id,
                    area: totalKeyAreaPixels - detectedKeyColorPixels 
                });
            }
        } finally {
            if (hsvKeyRegionCurrent) hsvKeyRegionCurrent.delete();
            if (lowerBound) lowerBound.delete(); 
            if (upperBound) upperBound.delete(); 
            if (keyColorMask) keyColorMask.delete();
        }
    });

    hsvCurrent.delete();
    return activeRegions;
}


// Update key states and play/stop sounds
function updateKeyStates(activeRegions) {
    const now = Date.now();
    const releaseDelay = 50; 

    const touchedKeysInCurrentFrame = new Set(activeRegions.map(ar => ar.keyId));

    pianoKeys.forEach(key => {
        const keyIsCurrentlyPressed = touchedKeysInCurrentFrame.has(key.id);

        if (keyIsCurrentlyPressed) {
            key.lastActiveTime = now; 
            if (!key.isActive) { 
                key.isActive = true;
                if (!key.isPlaying) {
                    synths[key.id].triggerAttack(key.note);
                    key.isPlaying = true;
                    // console.log(`Attack: ${key.noteName} (ID: ${key.id})`); // Can be noisy
                }
            }
        } else { 
            if (key.isActive && (now - key.lastActiveTime > releaseDelay)) {
                key.isActive = false;
                if (key.isPlaying) {
                    synths[key.id].triggerRelease();
                    key.isPlaying = false;
                    // console.log(`Release: ${key.noteName} (ID: ${key.id})`); // Can be noisy
                }
            }
        }
    });
}


// Draw piano keys
function drawPianoKeys() {
    const outputCanvas = document.getElementById('outputCanvas');
    if (!outputCanvas) {
        console.error("drawPianoKeys: outputCanvas not found!");
        return;
    }
    const ctx = outputCanvas.getContext('2d');
    ctx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    
    pianoKeys.forEach(key => {
        ctx.globalAlpha = 1.0; 
        if (key.isActive) {
            ctx.shadowColor = '#FFFFFF'; 
            ctx.shadowBlur = 25;
            ctx.fillStyle = 'rgba(255, 255, 255, 0.5)'; 
            ctx.fillRect(key.x + key.width * 0.05, key.y + key.height * 0.05, key.width * 0.9, key.height * 0.9);
        } else {
            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;
        }
        
        ctx.fillStyle = key.color;
        ctx.fillRect(key.x, key.y, key.width, key.height);
        
        ctx.strokeStyle = key.isActive ? '#FFFFFF' : '#333333'; 
        ctx.lineWidth = key.isActive ? 4 : 2;
        ctx.strokeRect(key.x, key.y, key.width, key.height);
        
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;

        ctx.fillStyle = '#FFFFFF'; 
        ctx.font = `bold ${Math.max(16, key.width / 5)}px Arial`; 
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'rgba(0,0,0,0.7)';
        ctx.strokeText(key.noteName, key.x + key.width / 2, key.y + key.height / 2);
        ctx.fillText(key.noteName, key.x + key.width / 2, key.y + key.height / 2);
    });
}

// Unused sound functions, kept for potential direct testing.
function playSound(keyId) {
    const key = pianoKeys.find(k => k.id === keyId);
    if (key && synths[keyId]) {
        synths[keyId].triggerAttackRelease(key.note, "0.5");
    }
}
function stopSound(keyId) {
    if (synths[keyId]) {
        synths[keyId].triggerRelease();
    }
}

// Removed the immediate call to onOpenCVReady().
// The call is now solely handled by the onload attribute
// in the script tag for opencv.js in index.html.
// This ensures OpenCV is fully loaded and the DOM is more likely
// to be ready before onOpenCVReady is executed.
console.log("app.js loaded. Waiting for OpenCV to trigger onOpenCVReady().");
