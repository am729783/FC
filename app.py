from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import math
import json

app = FastAPI()

# Enable CORS for Flutter app connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ExerciseTracker:
    def __init__(self):
        self.exercise_type = None
        self.counter = 0
        self.stage = None
        self.form_feedback = ""
        self.last_hip_pos = [0, 0]
        self.initial_shoulder_hip_dist = 0
        self.movement_history = []

        # Exercise definitions
        self.exercise_info = {
            'Bicep Curl': {
                'description': 'Classic arm exercise for bicep strength',
                'steps': [
                    '1. Stand straight, dumbbells at sides',
                    '2. Curl weights up to shoulders',
                    '3. Lower slowly back down',
                    '4. Keep elbows fixed at sides'
                ],
                'position': 'Stand FACING the camera',
                'landmarks': ['LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST',
                            'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']
            },
            'Lateral Raise': {
                'description': 'Shoulder exercise for deltoid development',
                'steps': [
                    '1. Stand straight, dumbbells at sides',
                    '2. Raise arms to shoulder level',
                    '3. Keep slight bend in elbows',
                    '4. Lower controlled to start'
                ],
                'position': 'Stand FACING the camera',
                'landmarks': ['LEFT_HIP', 'LEFT_SHOULDER', 'LEFT_ELBOW',
                            'RIGHT_HIP', 'RIGHT_SHOULDER', 'RIGHT_ELBOW']
            },
            'Tricep Pushdown': {
                'description': 'Isolation exercise for tricep development',
                'steps': [
                    '1. Stand with elbows tucked at sides',
                    '2. Start with forearms parallel to ground',
                    '3. Push hands down until arms fully extended',
                    '4. Slowly return to starting position'
                ],
                'position': 'Stand FACING the camera',
                'landmarks': ['LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST',
                            'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']
            },
            'Shoulder Press': {
                'description': 'Vertical pressing movement for shoulder strength',
                'steps': [
                    '1. Start with weights at shoulder level',
                    '2. Press weights overhead until arms are extended',
                    '3. Keep core tight and avoid arching back',
                    '4. Lower weights back to shoulder level'
                ],
                'position': 'Stand FACING the camera',
                'landmarks': ['LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST',
                             'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']
            },
            'Front Raise': {
                'description': 'Isolation exercise for anterior deltoids',
                'steps': [
                    '1. Stand with weights in front of thighs',
                    '2. Raise arms straight in front to shoulder height',
                    '3. Keep slight bend in elbows',
                    '4. Lower slowly back to start'
                ],
                'position': 'Stand FACING the camera',
                'landmarks': ['LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST',
                             'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']
            },
            'Lat Pulldown': {
                'description': 'Compound exercise for back width and strength',
                'steps': [
                    '1. Sit facing the camera with arms extended overhead',
                    '2. Pull hands down toward shoulders',
                    '3. Keep chest up and shoulders back',
                    '4. Slowly return to starting position'
                ],
                'position': 'Sit FACING the camera',
                'landmarks': ['LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST',
                             'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']
            },
            'Seated Row': {
                'description': 'Compound exercise for mid-back strength',
                'steps': [
                    '1. Sit facing the camera with arms extended forward',
                    '2. Pull hands toward torso, elbows close to body',
                    '3. Squeeze shoulder blades together',
                    '4. Slowly extend arms back to start'
                ],
                'position': 'Sit FACING the camera',
                'landmarks': ['LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST',
                             'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']
            },
            'Leg Curl': {
                'description': 'Isolation exercise for hamstring development',
                'steps': [
                    '1. Stand sideways to camera',
                    '2. Bend knee to bring heel toward buttocks',
                    '3. Keep thigh stationary throughout movement',
                    '4. Lower foot back to starting position with control'
                ],
                'position': 'Stand SIDEWAYS to the camera',
                'landmarks': ['LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE',
                             'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE']
            },
            'Leg Extension': {
                'description': 'Isolation exercise for quadricep strength',
                'steps': [
                    '1. Sit sideways to camera',
                    '2. Extend knee to straighten leg',
                    '3. Hold briefly at full extension',
                    '4. Lower leg back to 90-degree position'
                ],
                'position': 'Sit SIDEWAYS to the camera',
                'landmarks': ['LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE',
                             'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE']
            },
            'Abdominal Crunch': {
                'description': 'Core exercise targeting rectus abdominis',
                'steps': [
                    '1. Lie on back with knees bent, sideways to camera',
                    '2. Place hands lightly behind or beside head',
                    '3. Curl upper body toward knees',
                    '4. Lower back down with control'
                ],
                'position': 'Lie SIDEWAYS to the camera',
                'landmarks': ['LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE',
                             'RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE']
            },
            'Chest Fly': {
                'description': 'Isolation exercise for chest muscles',
                'steps': [
                    '1. Stand with arms extended to sides at shoulder height',
                    '2. Bring arms together in front of chest',
                    '3. Feel squeeze in chest muscles',
                    '4. Return to starting position with control'
                ],
                'position': 'Stand FACING the camera',
                'landmarks': ['LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST',
                             'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']
            },
            'Shrugs': {
                'description': 'Isolation exercise for trapezius muscles',
                'steps': [
                    '1. Stand holding weights at sides',
                    '2. Elevate shoulders toward ears as high as possible',
                    '3. Hold briefly at top position',
                    '4. Lower shoulders with control'
                ],
                'position': 'Stand FACING the camera',
                'landmarks': ['LEFT_SHOULDER', 'LEFT_EAR', 'RIGHT_SHOULDER', 'RIGHT_EAR']
            }
        }

    def validate_form(self, exercise_type, landmarks, angles):
        """Validate exercise form and return feedback"""
        try:
            if exercise_type == 'Bicep Curl':
                left_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                               landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                
                if abs(left_shoulder[0] - landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x) > 0.1:
                    return "Keep left elbow closer to body"
                if abs(right_shoulder[0] - landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x) > 0.1:
                    return "Keep right elbow closer to body"
                if abs(angles['left'] - angles['right']) > 20:
                    return "Keep arms moving together"
                return ""
                
            elif exercise_type == 'Lateral Raise':
                if angles['left'] > 100 or angles['right'] > 100:
                    return "Don't raise arms above shoulder level"
                if abs(angles['left'] - angles['right']) > 15:
                    return "Keep arms moving together"
                return ""
                
            elif exercise_type == 'Tricep Pushdown':
                left_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                               landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                
                if abs(left_shoulder[0] - landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x) > 0.1:
                    return "Keep left elbow tucked at side"
                if abs(right_shoulder[0] - landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x) > 0.1:
                    return "Keep right elbow tucked at side"
                if abs(angles['left'] - angles['right']) > 20:
                    return "Keep arms moving together"
                return ""
                
            elif exercise_type == 'Shoulder Press':
                left_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                               landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                
                left_elbow = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y]
                right_elbow = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x,
                             landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y]
                
                if abs(left_elbow[0] - left_shoulder[0]) > 0.1:
                    return "Keep left elbow aligned with shoulder"
                if abs(right_elbow[0] - right_shoulder[0]) > 0.1:
                    return "Keep right elbow aligned with shoulder"
                if abs(angles['left'] - angles['right']) > 20:
                    return "Keep arms moving together"
                return ""
                
            elif exercise_type == 'Front Raise':
                if angles['left'] > 100 or angles['right'] > 100:
                    return "Don't raise arms above shoulder level"
                if abs(angles['left'] - angles['right']) > 15:
                    return "Keep arms moving together"
                return ""
                
            elif exercise_type == 'Lat Pulldown':
                left_elbow = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y]
                right_elbow = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x,
                             landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y]
                
                elbow_width = abs(left_elbow[0] - right_elbow[0])
                if elbow_width < 0.2:
                    return "Keep elbows wider apart"
                if abs(angles['left'] - angles['right']) > 15:
                    return "Keep arms moving together"
                return ""
                
            elif exercise_type == 'Seated Row':
                left_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                               landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                
                left_elbow = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y]
                right_elbow = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x,
                             landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y]
                
                if left_elbow[0] < left_shoulder[0] - 0.1:
                    return "Keep left elbow closer to body"
                if right_elbow[0] > right_shoulder[0] + 0.1:
                    return "Keep right elbow closer to body"
                if abs(angles['left'] - angles['right']) > 15:
                    return "Keep arms moving together"
                return ""
                
            elif exercise_type == 'Leg Curl':
                left_hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x,
                          landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y]
                
                if abs(left_hip[1] - self.last_hip_pos[1]) > 0.03:
                    return "Keep hip stable, isolate leg movement"
                if angles['left'] < 30:
                    return "Increase range of motion"
                return ""
                
            elif exercise_type == 'Leg Extension':
                left_hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x,
                          landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y]
                
                if abs(left_hip[1] - self.last_hip_pos[1]) > 0.03:
                    return "Keep hip stable, isolate leg movement"
                if angles['left'] < 160:
                    return "Extend leg more completely"
                return ""
                
            elif exercise_type == 'Abdominal Crunch':
                left_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                               landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x,
                          landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y]
                
                curl_distance = self.initial_shoulder_hip_dist - abs(left_shoulder[1] - left_hip[1])
                
                if curl_distance < 0.05:
                    return "Increase range of motion"
                
                neck_angle = self.calculate_angle(
                    [landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR.value].x,
                    landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR.value].y],
                    [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y],
                    [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y]
                )
                
                if neck_angle < 130:
                    return "Don't pull with neck, use abs"
                return ""
                
            elif exercise_type == 'Chest Fly':
                left_shoulder_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y
                right_shoulder_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y
                left_wrist_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y
                right_wrist_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y
                
                if abs(left_shoulder_y - left_wrist_y) > 0.1:
                    return "Keep arms at shoulder height"
                if abs(right_shoulder_y - right_wrist_y) > 0.1:
                    return "Keep arms at shoulder height"
                if abs(angles['left'] - angles['right']) > 15:
                    return "Keep arms moving together"
                return ""
                
            elif exercise_type == 'Shrugs':
                left_shoulder_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y
                right_shoulder_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y
                left_ear_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR.value].y
                right_ear_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EAR.value].y
                
                shoulder_ear_dist = abs((left_shoulder_y - left_ear_y) + (right_shoulder_y - right_ear_y)) / 2
                if shoulder_ear_dist > 0.15:
                    return "Raise shoulders higher"
                return ""
                
        except Exception as e:
            return ""

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        try:
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            if angle > 180.0:
                angle = 360-angle
            return angle
        except:
            return 0

@app.get("/")
async def home():
    return {
        "status": "running",
        "websocket_endpoint": "/ws",
        "available_exercises": list(ExerciseTracker().exercise_info.keys())
    }

@app.websocket("/ws")
async def exercise_endpoint(websocket: WebSocket):
    await websocket.accept()
    tracker = ExerciseTracker()
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    try:
        while True:
            # Receive data from Flutter
            data = await websocket.receive()
            
            if isinstance(data, bytes):
                # Process image
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                h, w, _ = image.shape
                
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    angles = {'left': 0, 'right': 0}
                    
                    # Store hip position for stability checks
                    tracker.last_hip_pos = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    
                    # Calculate shoulder-hip distance for ab exercises
                    tracker.initial_shoulder_hip_dist = abs(
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y - 
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
                    )
                    
                    if tracker.exercise_type == 'Bicep Curl':
                        # Left arm angle
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        angles['left'] = tracker.calculate_angle(left_shoulder, left_elbow, left_wrist)
                        
                        # Right arm angle
                        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        angles['right'] = tracker.calculate_angle(right_shoulder, right_elbow, right_wrist)
                        
                        # Form validation and counter logic
                        tracker.form_feedback = tracker.validate_form(tracker.exercise_type, landmarks, angles)
                        
                        if not tracker.form_feedback:
                            if angles['left'] > 160 and angles['right'] > 160:
                                tracker.stage = "down"
                            if angles['left'] < 60 and angles['right'] < 60 and tracker.stage == "down":
                                tracker.stage = "up"
                                tracker.counter += 1
                    
                    elif tracker.exercise_type == 'Lateral Raise':
                        # Calculate angles for both arms
                        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        
                        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        
                        angles['left'] = tracker.calculate_angle(left_hip, left_shoulder, left_elbow)
                        angles['right'] = tracker.calculate_angle(right_hip, right_shoulder, right_elbow)
                        
                        tracker.form_feedback = tracker.validate_form(tracker.exercise_type, landmarks, angles)
                        
                        if not tracker.form_feedback:
                            if angles['left'] < 30 and angles['right'] < 30:
                                tracker.stage = "down"
                            if angles['left'] > 85 and angles['right'] > 85 and tracker.stage == "down":
                                tracker.stage = "up"
                                tracker.counter += 1
                    
                    elif tracker.exercise_type == 'Tricep Pushdown':
                        # Calculate angles for both arms
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        
                        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        
                        angles['left'] = tracker.calculate_angle(left_shoulder, left_elbow, left_wrist)
                        angles['right'] = tracker.calculate_angle(right_shoulder, right_elbow, right_wrist)
                        
                        tracker.form_feedback = tracker.validate_form(tracker.exercise_type, landmarks, angles)
                        
                        if not tracker.form_feedback:
                            if angles['left'] < 60 and angles['right'] < 60:
                                tracker.stage = "up"
                            if angles['left'] > 150 and angles['right'] > 150 and tracker.stage == "up":
                                tracker.stage = "down"
                                tracker.counter += 1
                    
                    elif tracker.exercise_type == 'Shoulder Press':
                        # Calculate angles for both arms
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        
                        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        
                        angles['left'] = tracker.calculate_angle(left_shoulder, left_elbow, left_wrist)
                        angles['right'] = tracker.calculate_angle(right_shoulder, right_elbow, right_wrist)
                        
                        tracker.form_feedback = tracker.validate_form(tracker.exercise_type, landmarks, angles)
                        
                        if not tracker.form_feedback:
                            if angles['left'] < 60 and angles['right'] < 60:
                                tracker.stage = "down"
                            if angles['left'] > 160 and angles['right'] > 160 and tracker.stage == "down":
                                tracker.stage = "up"
                                tracker.counter += 1
                    
                    elif tracker.exercise_type == 'Front Raise':
                        # Calculate angles for both arms relative to torso
                        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        
                        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        
                        angles['left'] = tracker.calculate_angle(left_hip, left_shoulder, left_wrist)
                        angles['right'] = tracker.calculate_angle(right_hip, right_shoulder, right_wrist)
                        
                        tracker.form_feedback = tracker.validate_form(tracker.exercise_type, landmarks, angles)
                        
                        if not tracker.form_feedback:
                            if angles['left'] < 30 and angles['right'] < 30:
                                tracker.stage = "down"
                            if angles['left'] > 85 and angles['right'] > 85 and tracker.stage == "down":
                                tracker.stage = "up"
                                tracker.counter += 1
                    
                    elif tracker.exercise_type == 'Lat Pulldown':
                        # Calculate angles for both arms
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        
                        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        
                        angles['left'] = tracker.calculate_angle(left_shoulder, left_elbow, left_wrist)
                        angles['right'] = tracker.calculate_angle(right_shoulder, right_elbow, right_wrist)
                        
                        tracker.form_feedback = tracker.validate_form(tracker.exercise_type, landmarks, angles)
                        
                        if not tracker.form_feedback:
                            if angles['left'] > 150 and angles['right'] > 150:
                                tracker.stage = "up"
                            if angles['left'] < 80 and angles['right'] < 80 and tracker.stage == "up":
                                tracker.stage = "down"
                                tracker.counter += 1
                    
                    elif tracker.exercise_type == 'Seated Row':
                        # Calculate angles for both arms
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        
                        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        
                        angles['left'] = tracker.calculate_angle(left_shoulder, left_elbow, left_wrist)
                        angles['right'] = tracker.calculate_angle(right_shoulder, right_elbow, right_wrist)
                        
                        tracker.form_feedback = tracker.validate_form(tracker.exercise_type, landmarks, angles)
                        
                        if not tracker.form_feedback:
                            if angles['left'] > 140 and angles['right'] > 140:
                                tracker.stage = "start"
                            if angles['left'] < 80 and angles['right'] < 80 and tracker.stage == "start":
                                tracker.stage = "pulled"
                                tracker.counter += 1
                    
                    elif tracker.exercise_type == 'Leg Curl':
                        # Calculate angle for the leg
                        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                        
                        angles['left'] = tracker.calculate_angle(left_hip, left_knee, left_ankle)
                        
                        tracker.form_feedback = tracker.validate_form(tracker.exercise_type, landmarks, angles)
                        
                        if not tracker.form_feedback:
                            if angles['left'] > 160:
                                tracker.stage = "extended"
                            if angles['left'] < 80 and tracker.stage == "extended":
                                tracker.stage = "curled"
                                tracker.counter += 1
                    
                    elif tracker.exercise_type == 'Leg Extension':
                        # Calculate angle for the leg
                        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                        
                        angles['left'] = tracker.calculate_angle(left_hip, left_knee, left_ankle)
                        
                        tracker.form_feedback = tracker.validate_form(tracker.exercise_type, landmarks, angles)
                        
                        if not tracker.form_feedback:
                            if angles['left'] < 90:
                                tracker.stage = "bent"
                            if angles['left'] > 160 and tracker.stage == "bent":
                                tracker.stage = "extended"
                                tracker.counter += 1
                    
                    elif tracker.exercise_type == 'Abdominal Crunch':
                        # Calculate angle for the torso
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        
                        angles['left'] = tracker.calculate_angle(left_shoulder, left_hip, left_knee)
                        
                        tracker.form_feedback = tracker.validate_form(tracker.exercise_type, landmarks, angles)
                        
                        if not tracker.form_feedback:
                            if angles['left'] > 120:
                                tracker.stage = "down"
                            if angles['left'] < 80 and tracker.stage == "down":
                                tracker.stage = "up"
                                tracker.counter += 1
                    
                    elif tracker.exercise_type == 'Chest Fly':
                        # Track wrist positions relative to shoulders
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        
                        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        
                        # Calculate distance between wrists for tracking the fly movement
                        wrist_distance = np.sqrt((left_wrist[0] - right_wrist[0])**2 + 
                                       (left_wrist[1] - right_wrist[1])**2)
                        
                        tracker.form_feedback = tracker.validate_form(tracker.exercise_type, landmarks, angles)
                        
                        if not tracker.form_feedback:
                            if wrist_distance > 0.6:  # Arms wide apart
                                tracker.stage = "open"
                            if wrist_distance < 0.2 and tracker.stage == "open":  # Arms close together
                                tracker.stage = "closed"
                                tracker.counter += 1
                    
                    elif tracker.exercise_type == 'Shrugs':
                        # Track shoulder positions relative to ears
                        left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        
                        right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
                        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        
                        # Calculate distance between ears and shoulders
                        left_distance = np.sqrt((left_ear[0] - left_shoulder[0])**2 + 
                                              (left_ear[1] - left_shoulder[1])**2)
                        right_distance = np.sqrt((right_ear[0] - right_shoulder[0])**2 + 
                                               (right_ear[1] - right_shoulder[1])**2)
                        
                        avg_distance = (left_distance + right_distance) / 2
                        
                        tracker.form_feedback = tracker.validate_form(tracker.exercise_type, landmarks, angles)
                        
                        if not tracker.form_feedback:
                            if avg_distance > 0.15:  # Shoulders low
                                tracker.stage = "down"
                            if avg_distance < 0.11 and tracker.stage == "down":  # Shoulders raised
                                tracker.stage = "up"
                                tracker.counter += 1
                    
                    await websocket.send_json({
                        "counter": tracker.counter,
                        "stage": tracker.stage,
                        "feedback": tracker.form_feedback,
                        "exercise": tracker.exercise_type
                    })
            
            elif isinstance(data, str):
                try:
                    json_data = json.loads(data)
                    tracker.exercise_type = json_data.get("exercise_type")
                except:
                    pass
                    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
