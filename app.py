from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import json
import base64
from datetime import datetime
import asyncio
from typing import Dict, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Exercise Tracker API", version="1.0.0")

# Enable CORS with more restrictive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed domains
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class ExerciseTracker:
    def __init__(self):
        self.exercise_type: Optional[str] = None
        self.counter = 0
        self.stage: Optional[str] = None
        self.form_feedback = ""
        self.last_hip_pos = [0, 0]
        self.initial_shoulder_hip_dist = 0
        self.movement_history = []
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )

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

    def cleanup(self):
        """Clean up resources"""
        if self.pose:
            self.pose.close()

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
        except Exception as e:
            logger.error(f"Error calculating angle: {e}")
            return 0

    def process_exercise(self, landmarks, exercise_type: str) -> Dict:
        """Process exercise and return results"""
        try:
            angles = {'left': 0, 'right': 0}
            
            # Store hip position
            self.last_hip_pos = [
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
            ]
            
            # Calculate shoulder-hip distance
            self.initial_shoulder_hip_dist = abs(
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y - 
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
            )
            
            if exercise_type == 'Bicep Curl':
                return self._process_bicep_curl(landmarks, angles)
            elif exercise_type == 'Lateral Raise':
                return self._process_lateral_raise(landmarks, angles)
            elif exercise_type == 'Tricep Pushdown':
                return self._process_tricep_pushdown(landmarks, angles)
            elif exercise_type == 'Shoulder Press':
                return self._process_shoulder_press(landmarks, angles)
            elif exercise_type == 'Front Raise':
                return self._process_front_raise(landmarks, angles)
            elif exercise_type == 'Lat Pulldown':
                return self._process_lat_pulldown(landmarks, angles)
            elif exercise_type == 'Seated Row':
                return self._process_seated_row(landmarks, angles)
            elif exercise_type == 'Leg Curl':
                return self._process_leg_curl(landmarks, angles)
            elif exercise_type == 'Leg Extension':
                return self._process_leg_extension(landmarks, angles)
            elif exercise_type == 'Abdominal Crunch':
                return self._process_abdominal_crunch(landmarks, angles)
            elif exercise_type == 'Chest Fly':
                return self._process_chest_fly(landmarks, angles)
            elif exercise_type == 'Shrugs':
                return self._process_shrugs(landmarks, angles)
            else:
                return {
                    "counter": self.counter,
                    "stage": self.stage,
                    "feedback": f"Exercise '{exercise_type}' not fully implemented yet",
                    "angles": angles
                }
                
        except Exception as e:
            logger.error(f"Error processing exercise: {e}")
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": "Error processing exercise",
                "angles": {'left': 0, 'right': 0}
            }

    def _process_bicep_curl(self, landmarks, angles):
        """Process Bicep Curl exercise"""
        try:
            # Calculate left arm angle
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angles['left'] = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Calculate right arm angle
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            angles['right'] = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Form validation
            self.form_feedback = self._validate_bicep_curl_form(landmarks, angles)
            
            # Counter logic
            if not self.form_feedback:
                if angles['left'] > 160 and angles['right'] > 160:
                    self.stage = "down"
                elif angles['left'] < 60 and angles['right'] < 60 and self.stage == "down":
                    self.stage = "up"
                    self.counter += 1
            
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": self.form_feedback,
                "angles": angles
            }
            
        except Exception as e:
            logger.error(f"Error in bicep curl processing: {e}")
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": "Error processing bicep curl",
                "angles": angles
            }

    def _validate_bicep_curl_form(self, landmarks, angles):
        """Validate Bicep Curl form"""
        try:
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            
            if abs(left_shoulder[0] - landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x) > 0.1:
                return "Keep left elbow closer to body"
            if abs(right_shoulder[0] - landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x) > 0.1:
                return "Keep right elbow closer to body"
            if abs(angles['left'] - angles['right']) > 20:
                return "Keep arms moving together"
            return ""
        except:
            return ""

    def _process_lateral_raise(self, landmarks, angles):
        """Process Lateral Raise exercise"""
        try:
            # Calculate left arm angle
            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                      landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            angles['left'] = self.calculate_angle(left_hip, left_shoulder, left_elbow)
            
            # Calculate right arm angle
            right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            angles['right'] = self.calculate_angle(right_hip, right_shoulder, right_elbow)
            
            # Form validation
            self.form_feedback = self._validate_lateral_raise_form(landmarks, angles)
            
            # Counter logic
            if not self.form_feedback:
                if angles['left'] < 30 and angles['right'] < 30:
                    self.stage = "down"
                elif angles['left'] > 85 and angles['right'] > 85 and self.stage == "down":
                    self.stage = "up"
                    self.counter += 1
            
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": self.form_feedback,
                "angles": angles
            }
            
        except Exception as e:
            logger.error(f"Error in lateral raise processing: {e}")
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": "Error processing lateral raise",
                "angles": angles
            }

    def _validate_lateral_raise_form(self, landmarks, angles):
        """Validate Lateral Raise form"""
        try:
            if angles['left'] > 100 or angles['right'] > 100:
                return "Don't raise arms above shoulder level"
            if abs(angles['left'] - angles['right']) > 15:
                return "Keep arms moving together"
            return ""
        except:
            return ""

    def _process_tricep_pushdown(self, landmarks, angles):
        """Process Tricep Pushdown exercise"""
        try:
            # Calculate left arm angle
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angles['left'] = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Calculate right arm angle
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            angles['right'] = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Form validation
            self.form_feedback = self._validate_tricep_pushdown_form(landmarks, angles)
            
            # Counter logic
            if not self.form_feedback:
                if angles['left'] < 60 and angles['right'] < 60:
                    self.stage = "up"
                elif angles['left'] > 150 and angles['right'] > 150 and self.stage == "up":
                    self.stage = "down"
                    self.counter += 1
            
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": self.form_feedback,
                "angles": angles
            }
            
        except Exception as e:
            logger.error(f"Error in tricep pushdown processing: {e}")
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": "Error processing tricep pushdown",
                "angles": angles
            }

    def _validate_tricep_pushdown_form(self, landmarks, angles):
        """Validate Tricep Pushdown form"""
        try:
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            
            if abs(left_shoulder[0] - landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x) > 0.1:
                return "Keep left elbow tucked at side"
            if abs(right_shoulder[0] - landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x) > 0.1:
                return "Keep right elbow tucked at side"
            if abs(angles['left'] - angles['right']) > 20:
                return "Keep arms moving together"
            return ""
        except:
            return ""

    def _process_shoulder_press(self, landmarks, angles):
        """Process Shoulder Press exercise"""
        try:
            # Calculate left arm angle
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angles['left'] = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Calculate right arm angle
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            angles['right'] = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Form validation
            self.form_feedback = self._validate_shoulder_press_form(landmarks, angles)
            
            # Counter logic
            if not self.form_feedback:
                if angles['left'] < 60 and angles['right'] < 60:
                    self.stage = "down"
                elif angles['left'] > 160 and angles['right'] > 160 and self.stage == "down":
                    self.stage = "up"
                    self.counter += 1
            
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": self.form_feedback,
                "angles": angles
            }
            
        except Exception as e:
            logger.error(f"Error in shoulder press processing: {e}")
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": "Error processing shoulder press",
                "angles": angles
            }

    def _validate_shoulder_press_form(self, landmarks, angles):
        """Validate Shoulder Press form"""
        try:
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            
            left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            
            if abs(left_elbow[0] - left_shoulder[0]) > 0.1:
                return "Keep left elbow aligned with shoulder"
            if abs(right_elbow[0] - right_shoulder[0]) > 0.1:
                return "Keep right elbow aligned with shoulder"
            if abs(angles['left'] - angles['right']) > 20:
                return "Keep arms moving together"
                
            return ""
        except:
            return ""

    def _process_front_raise(self, landmarks, angles):
        """Process Front Raise exercise"""
        try:
            # Calculate left arm angle
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angles['left'] = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Calculate right arm angle
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            angles['right'] = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Form validation
            self.form_feedback = self._validate_front_raise_form(landmarks, angles)
            
            # Counter logic
            if not self.form_feedback:
                if angles['left'] < 30 and angles['right'] < 30:
                    self.stage = "down"
                elif angles['left'] > 85 and angles['right'] > 85 and self.stage == "down":
                    self.stage = "up"
                    self.counter += 1
            
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": self.form_feedback,
                "angles": angles
            }
            
        except Exception as e:
            logger.error(f"Error in front raise processing: {e}")
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": "Error processing front raise",
                "angles": angles
            }

    def _validate_front_raise_form(self, landmarks, angles):
        """Validate Front Raise form"""
        try:
            if angles['left'] > 100 or angles['right'] > 100:
                return "Don't raise arms above shoulder level"
            if abs(angles['left'] - angles['right']) > 15:
                return "Keep arms moving together"
            return ""
        except:
            return ""

    def _process_lat_pulldown(self, landmarks, angles):
        """Process Lat Pulldown exercise"""
        try:
            # Calculate left arm angle
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angles['left'] = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Calculate right arm angle
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            angles['right'] = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Form validation
            self.form_feedback = self._validate_lat_pulldown_form(landmarks, angles)
            
            # Counter logic
            if not self.form_feedback:
                if angles['left'] > 160 and angles['right'] > 160:
                    self.stage = "up"
                elif angles['left'] < 90 and angles['right'] < 90 and self.stage == "up":
                    self.stage = "down"
                    self.counter += 1
            
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": self.form_feedback,
                "angles": angles
            }
            
        except Exception as e:
            logger.error(f"Error in lat pulldown processing: {e}")
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": "Error processing lat pulldown",
                "angles": angles
            }

    def _validate_lat_pulldown_form(self, landmarks, angles):
        """Validate Lat Pulldown form"""
        try:
            left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            
            elbow_width = abs(left_elbow[0] - right_elbow[0])
            if elbow_width < 0.2:
                return "Keep elbows wider apart"
            if abs(angles['left'] - angles['right']) > 15:
                return "Keep arms moving together"
            return ""
        except:
            return ""

    def _process_seated_row(self, landmarks, angles):
        """Process Seated Row exercise"""
        try:
            # Calculate left arm angle
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angles['left'] = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Calculate right arm angle
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            angles['right'] = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Form validation
            self.form_feedback = self._validate_seated_row_form(landmarks, angles)
            
            # Counter logic
            if not self.form_feedback:
                if angles['left'] > 160 and angles['right'] > 160:
                    self.stage = "extended"
                elif angles['left'] < 90 and angles['right'] < 90 and self.stage == "extended":
                    self.stage = "contracted"
                    self.counter += 1
            
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": self.form_feedback,
                "angles": angles
            }
            
        except Exception as e:
            logger.error(f"Error in seated row processing: {e}")
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": "Error processing seated row",
                "angles": angles
            }

    def _validate_seated_row_form(self, landmarks, angles):
        """Validate Seated Row form"""
        try:
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            
            left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            
            if left_elbow[0] < left_shoulder[0] - 0.1:
                return "Keep left elbow closer to body"
            if right_elbow[0] > right_shoulder[0] + 0.1:
                return "Keep right elbow closer to body"
            if abs(angles['left'] - angles['right']) > 15:
                return "Keep arms moving together"
            return ""
        except:
            return ""

        def _process_leg_curl(self, landmarks, angles):
        """Process Leg Curl exercise"""
        try:
            # Calculate left leg angle
            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                      landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                       landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            angles['left'] = self.calculate_angle(left_hip, left_knee, left_ankle)
            
            # Calculate right leg angle
            right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            angles['right'] = self.calculate_angle(right_hip, right_knee, right_ankle)
            
            # Form validation
            self.form_feedback = self._validate_leg_curl_form(landmarks, angles)
            
            # Counter logic
            if not self.form_feedback:
                if angles['left'] > 150 and angles['right'] > 150:
                    self.stage = "extended"
                elif angles['left'] < 90 and angles['right'] < 90 and self.stage == "extended":
                    self.stage = "curled"
                    self.counter += 1
            
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": self.form_feedback,
                "angles": angles
            }
            
        except Exception as e:
            logger.error(f"Error in leg curl processing: {e}")
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": "Error processing leg curl",
                "angles": angles
            }

    def _validate_leg_curl_form(self, landmarks, angles):
        """Validate Leg Curl form"""
        try:
            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                      landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            
            if abs(left_hip[1] - self.last_hip_pos[1]) > 0.03:
                return "Keep hip stable, isolate leg movement"
            if angles['left'] < 30:
                return "Increase range of motion"
            return ""
        except:
            return ""

    def _process_leg_extension(self, landmarks, angles):
        """Process Leg Extension exercise"""
        try:
            # Calculate left leg angle
            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                      landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                       landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            angles['left'] = self.calculate_angle(left_hip, left_knee, left_ankle)
            
            # Calculate right leg angle
            right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            angles['right'] = self.calculate_angle(right_hip, right_knee, right_ankle)
            
            # Form validation
            self.form_feedback = self._validate_leg_extension_form(landmarks, angles)
            
            # Counter logic
            if not self.form_feedback:
                if angles['left'] < 90 and angles['right'] < 90:
                    self.stage = "flexed"
                elif angles['left'] > 160 and angles['right'] > 160 and self.stage == "flexed":
                    self.stage = "extended"
                    self.counter += 1
            
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": self.form_feedback,
                "angles": angles
            }
            
        except Exception as e:
            logger.error(f"Error in leg extension processing: {e}")
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": "Error processing leg extension",
                "angles": angles
            }

    def _validate_leg_extension_form(self, landmarks, angles):
        """Validate Leg Extension form"""
        try:
            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                      landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            
            if abs(left_hip[1] - self.last_hip_pos[1]) > 0.03:
                return "Keep hip stable, isolate leg movement"
            if angles['left'] < 160:
                return "Extend leg more completely"
            return ""
        except:
            return ""

    def _process_abdominal_crunch(self, landmarks, angles):
        """Process Abdominal Crunch exercise"""
        try:
            # Calculate shoulder-hip distance
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                      landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            
            curl_distance = self.initial_shoulder_hip_dist - abs(left_shoulder[1] - left_hip[1])
            
            # Calculate neck angle
            neck_angle = self.calculate_angle(
                [landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x,
                 landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y],
                [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                 landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                 landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            )
            
            # Form validation
            self.form_feedback = self._validate_abdominal_crunch_form(landmarks, curl_distance, neck_angle)
            
            # Counter logic
            if not self.form_feedback:
                if curl_distance < 0.02:
                    self.stage = "down"
                elif curl_distance > 0.05 and self.stage == "down":
                    self.stage = "up"
                    self.counter += 1
            
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": self.form_feedback,
                "angles": {'curl_distance': curl_distance, 'neck_angle': neck_angle}
            }
            
        except Exception as e:
            logger.error(f"Error in abdominal crunch processing: {e}")
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": "Error processing abdominal crunch",
                "angles": {'curl_distance': 0, 'neck_angle': 0}
            }

    def _validate_abdominal_crunch_form(self, landmarks, curl_distance, neck_angle):
        """Validate Abdominal Crunch form"""
        try:
            if curl_distance < 0.05:
                return "Increase range of motion"
            if neck_angle < 130:
                return "Don't pull with neck, use abs"
            return ""
        except:
            return ""

    def _process_chest_fly(self, landmarks, angles):
        """Process Chest Fly exercise"""
        try:
            # Calculate left arm angle
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angles['left'] = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Calculate right arm angle
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            angles['right'] = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Form validation
            self.form_feedback = self._validate_chest_fly_form(landmarks, angles)
            
            # Counter logic
            if not self.form_feedback:
                if angles['left'] > 150 and angles['right'] > 150:
                    self.stage = "extended"
                elif angles['left'] < 90 and angles['right'] < 90 and self.stage == "extended":
                    self.stage = "contracted"
                    self.counter += 1
            
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": self.form_feedback,
                "angles": angles
            }
            
        except Exception as e:
            logger.error(f"Error in chest fly processing: {e}")
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": "Error processing chest fly",
                "angles": angles
            }

    def _validate_chest_fly_form(self, landmarks, angles):
        """Validate Chest Fly form"""
        try:
            left_shoulder_y = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            right_shoulder_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            left_wrist_y = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y
            right_wrist_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y
            
            if abs(left_shoulder_y - left_wrist_y) > 0.1:
                return "Keep arms at shoulder height"
            if abs(right_shoulder_y - right_wrist_y) > 0.1:
                return "Keep arms at shoulder height"
            if abs(angles['left'] - angles['right']) > 15:
                return "Keep arms moving together"
            return ""
        except:
            return ""

    def _process_shrugs(self, landmarks, angles):
        """Process Shrugs exercise"""
        try:
            # Calculate shoulder elevation
            left_shoulder_y = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            right_shoulder_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            left_ear_y = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y
            right_ear_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y
            
            shoulder_ear_dist = abs((left_shoulder_y - left_ear_y) + (right_shoulder_y - right_ear_y)) / 2
            
            # Form validation
            self.form_feedback = self._validate_shrugs_form(shoulder_ear_dist)
            
            # Counter logic
            if not self.form_feedback:
                if shoulder_ear_dist < 0.1:
                    self.stage = "down"
                elif shoulder_ear_dist > 0.15 and self.stage == "down":
                    self.stage = "up"
                    self.counter += 1
            
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": self.form_feedback,
                "angles": {'shoulder_elevation': shoulder_ear_dist}
            }
            
        except Exception as e:
            logger.error(f"Error in shrugs processing: {e}")
            return {
                "counter": self.counter,
                "stage": self.stage,
                "feedback": "Error processing shrugs",
                "angles": {'shoulder_elevation': 0}
            }

    def _validate_shrugs_form(self, shoulder_ear_dist):
        """Validate Shrugs form"""
        try:
            if shoulder_ear_dist > 0.15:
                return "Raise shoulders higher"
            return ""
        except:
            return ""

# Global connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.trackers: Dict[str, ExerciseTracker] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.trackers[client_id] = ExerciseTracker()
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.trackers:
            self.trackers[client_id].cleanup()
            del self.trackers[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(json.dumps(message))

manager = ConnectionManager()

@app.get("/")
async def home():
    return {
        "status": "running",
        "websocket_endpoint": "/ws",
        "available_exercises": list(ExerciseTracker().exercise_info.keys()),
        "version": "1.0.0",
        "endpoints": {
            "websocket": "/ws",
            "analyze": "/analyze",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(manager.active_connections)
    }

@app.websocket("/ws")
async def exercise_endpoint(websocket: WebSocket):
    client_id = f"client_{datetime.now().timestamp()}"
    
    try:
        await manager.connect(websocket, client_id)
        tracker = manager.trackers[client_id]
        
        while True:
            try:
                # استقبال البيانات بشكل صحيح
                data = await websocket.receive_text()
                json_data = json.loads(data)
                
                # Handle exercise type selection
                if 'exercise_type' in json_data:
                    exercise_type = json_data['exercise_type']
                    if exercise_type in tracker.exercise_info:
                        tracker.exercise_type = exercise_type
                        tracker.counter = 0
                        tracker.stage = None
                        await manager.send_personal_message({
                            "type": "exercise_selected",
                            "exercise": exercise_type,
                            "info": tracker.exercise_info[exercise_type]
                        }, client_id)
                    else:
                        await manager.send_personal_message({
                            "type": "error",
                            "message": f"Unknown exercise: {exercise_type}"
                        }, client_id)
                        continue
                
                # Handle frame data
                if 'frame' in json_data and tracker.exercise_type:
                    try:
                        # تحويل البيانات
                        frame_data = json_data['frame']
                        if ',' in frame_data:
                            frame_bytes = base64.b64decode(frame_data.split(',')[1])
                        else:
                            frame_bytes = base64.b64decode(frame_data)
                        
                        image = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
                        
                        if image is None:
                            await manager.send_personal_message({
                                "type": "error",
                                "message": "Invalid image data"
                            }, client_id)
                            continue
                        
                        # Process image
                        results = tracker.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        
                        if results.pose_landmarks:
                            landmarks = results.pose_landmarks.landmark
                            result = tracker.process_exercise(landmarks, tracker.exercise_type)
                            
                            await manager.send_personal_message({
                                "type": "analysis_result",
                                "counter": result["counter"],
                                "stage": result["stage"],
                                "feedback": result["feedback"],
                                "exercise": tracker.exercise_type,
                                "timestamp": datetime.now().isoformat()
                            }, client_id)
                        else:
                            await manager.send_personal_message({
                                "type": "warning",
                                "message": "No pose detected. Make sure you're visible in the camera."
                            }, client_id)
                            
                    except Exception as e:
                        logger.error(f"Frame processing error: {e}")
                        await manager.send_personal_message({
                            "type": "error",
                            "message": "Error processing frame"
                        }, client_id)
                        
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON data"
                }, client_id)
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Error processing message"
                }, client_id)
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
    finally:
        manager.disconnect(client_id)

@app.post("/analyze")
async def analyze_frame(request: Request):
    """HTTP endpoint for single frame analysis"""
    try:
        data = await request.json()
        
        if 'frame' not in data:
            raise HTTPException(status_code=400, detail="Missing 'frame' field")
        
        exercise_type = data.get('exercise_type', 'Bicep Curl')
        
        # Create temporary tracker
        tracker = ExerciseTracker()
        tracker.exercise_type = exercise_type
        
        try:
            # Process frame
            frame_data = data['frame']
            if ',' in frame_data:
                frame_bytes = base64.b64decode(frame_data.split(',')[1])
            else:
                frame_bytes = base64.b64decode(frame_data)
            
            image = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image data")
            
            # Process image
            results = tracker.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                result = tracker.process_exercise(landmarks, exercise_type)
                
                return {
                    "success": True,
                    "counter": result["counter"],
                    "stage": result["stage"],
                    "feedback": result["feedback"],
                    "exercise": exercise_type,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "message": "No pose detected in the image"
                }
                
        finally:
            tracker.cleanup()
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {exc}")
    return {
        "error": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
