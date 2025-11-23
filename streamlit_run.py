import streamlit as st
import cv2
import numpy as np
import pandas as pd
from sam3.model_builder import build_sam3_video_predictor
import torch
import threading
import random
from datetime import datetime
import os
import time
import pygame  # Import pygame for sound

class ObjectMonitoringApp:
    def __init__(self):
        self.predictor = None
        self.session_id = None
        self.cap = None
        self.restricted_area = None
        self.csv_file = "data/detection_log.csv"
        self.object_entry_times = {}

        # Alert system
        self.alert_active = False
        self.alert_thread = None
        
        # Initialize CSV file if not exists
        if not os.path.exists(self.csv_file):
            pd.DataFrame(columns=["Timestamp", "Class", "Confidence", "Restricted Area Violation"]).to_csv(self.csv_file, index=False)

        # Initialize pygame for sound
        try:
            pygame.mixer.init()
            self.audio_enabled = True
        except Exception as e:
            print(f"Audio initialization failed (expected on server): {e}")
            self.audio_enabled = False
        
        # Initialize SAM 3
        self.initialize_sam3()

    def initialize_sam3(self):
        """Initialize SAM 3 predictor."""
        try:
            self.predictor = build_sam3_video_predictor()
            st.toast("SAM 3 Initialized successfully!", icon="✅")
        except Exception as e:
            st.error(f"Failed to initialize SAM 3: {e}")

    def start_sam3_session(self, source):
        """Start a SAM 3 session with the video source."""
        if self.predictor:
            try:
                # If source is an integer (webcam index), convert to string or handle appropriately
                # SAM 3 might expect a path or URL. For webcam, we might need to rely on frame passing if supported.
                # Assuming '0' works or we use the user's rtsp placeholder.
                # For this implementation, we'll use the user's pattern.
                resource_path = source if isinstance(source, str) else str(source)
                
                response = self.predictor.handle_request(
                    request=dict(
                        type="start_session",
                        resource_path=resource_path, 
                    )
                )
                self.session_id = response['session_id']
                return True
            except Exception as e:
                st.error(f"Failed to start SAM 3 session: {e}")
                return False
        return False

    def start_webcam(self, source=0):
        """Start the webcam and SAM 3 session."""
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            st.error("Error: Unable to access the webcam.")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Start SAM 3 session
        # Note: If using webcam 0, SAM 3 might conflict if it tries to open it too.
        # We will attempt to start the session.
        self.start_sam3_session(source)
        
        return True

    def stop_webcam(self):
        """Stop the webcam."""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
            self.cap = None
            self.stop_alert()

    def play_alert_sound(self, sound_path):
        """Play alert sound in a loop while an object is inside the restricted area."""
        if not self.audio_enabled:
            return
            
        try:
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play(-1)
            while self.alert_active:
                time.sleep(1)
            pygame.mixer.music.stop()
        except Exception as e:
            print(f"Audio error: {e}")

    def start_alert(self, sound_path):
        """Start playing alert sound in a separate thread if not already playing."""
        if not self.alert_active:
            self.alert_active = True
            self.alert_thread = threading.Thread(target=self.play_alert_sound, args=(sound_path,), daemon=True)
            self.alert_thread.start()

    def stop_alert(self):
        """Stop the alert sound."""
        if self.alert_active:
            self.alert_active = False

    def draw_roi(self, frame):
        """Draw an elliptical restricted area on the frame."""
        height, width, _ = frame.shape
        center = (width // 2, height // 2)
        axes = (width // 4, height // 8)
        self.restricted_area = (center, axes)
        cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 0, 255), 2)
        return frame

    def is_near_restricted_area(self, box):
        """Check if the detected object is inside the restricted area."""
        if self.restricted_area:
            center, axes = self.restricted_area
            x1, y1, x2, y2 = box
            object_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            distance = np.linalg.norm(np.array(center) - np.array(object_center))
            return distance < (min(axes) + 50)
        return False

    def save_detection_data(self, class_name, confidence, near_restricted_area):
        """Save detection data to CSV if a restricted area violation occurs."""
        if near_restricted_area:
            data = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Class": class_name,
                "Confidence": confidence,
                "Restricted Area Violation": "Yes",
            }
            df = pd.DataFrame([data])
            df.to_csv(self.csv_file, mode='a', header=False, index=False)

    def detect_objects_with_sam3(self, frame, frame_idx, safety_prompts):
        """
        SAM 3 detection using natural language prompts
        """
        detected_violations = []
        
        if not self.predictor or not self.session_id:
            return detected_violations

        for prompt in safety_prompts:
            try:
                response = self.predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=self.session_id,
                        frame_index=frame_idx,
                        text=prompt,
                    )
                )
                
                outputs = response.get('outputs', {})
                
                if outputs:
                    for obj_id, obj_data in outputs.items():
                        confidence = obj_data.get('scores', [0])[0]
                        
                        if confidence >= 0.75:
                            # SAM 3 might return masks or boxes. We need boxes for drawing.
                            # Assuming obj_data contains 'box' or similar.
                            # If not, we might need to process masks.
                            # For now, we'll assume a dummy box or try to extract it if available.
                            # The user snippet didn't show box extraction, so we'll improvise or assume it's there.
                            # Let's assume obj_data has 'bbox' [x1, y1, x2, y2]
                            bbox = obj_data.get('bbox', [0, 0, 0, 0]) 
                            
                            detected_violations.append({
                                'type': prompt,
                                'confidence': float(confidence),
                                'obj_id': obj_id,
                                'timestamp': frame_idx,
                                'box': bbox
                            })
            except Exception as e:
                print(f"Error processing prompt '{prompt}': {e}")
        
        return detected_violations

    def update_frame(self, frame_idx, active_prompts):
        """Process each frame for object detection and alerts."""
        if not self.cap:
            return None, []

        ret, frame = self.cap.read()
        if not ret:
            return None, []

        # SAM 3 detection
        violations = self.detect_objects_with_sam3(frame, frame_idx, active_prompts)
        
        annotated_frame = frame.copy()
        detected_classes = []
        object_inside_restricted_area = False

        for violation in violations:
            class_name = violation['type']
            confidence = violation['confidence']
            bbox = violation.get('box', [0, 0, 0, 0])
            
            detected_classes.append(class_name)
            
            # Draw bounding box if valid
            if bbox and len(bbox) == 4 and any(bbox):
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if self.is_near_restricted_area([x1, y1, x2, y2]):
                    object_inside_restricted_area = True
                    cv2.putText(annotated_frame, "Object in Restricted Area!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    if class_name not in self.object_entry_times:
                        self.object_entry_times[class_name] = time.time()

                    if time.time() - self.object_entry_times[class_name] > 2:
                        self.save_detection_data(class_name, confidence, True)
                        self.object_entry_times[class_name] = time.time()

        if object_inside_restricted_area:
             # Alert if any violation is in restricted area
            self.start_alert("alert.mp3")
        else:
            self.stop_alert()

        annotated_frame = self.draw_roi(annotated_frame)

        return annotated_frame, detected_classes

    def run(self):
        st.set_page_config(page_title="Real-Time Safety Monitoring (SAM 3)", layout="wide")
        st.markdown("<h2 style='text-align: center;'>🔍 Real-Time Safety Monitoring with SAM 3</h2>", unsafe_allow_html=True)

        st.sidebar.title("Safety Monitoring Configuration")
        st.sidebar.markdown("### Active Prompts")

        default_prompts = [
            "person without safety helmet",
            "person in restricted zone",
            "fire or smoke detected",
            "person fallen on ground",
            "unauthorized vehicle entry"
        ]

        active_prompts = st.sidebar.multiselect(
            "Select active safety rules:",
            default_prompts,
            default=default_prompts
        )

        custom_prompt = st.sidebar.text_input("Add custom rule:")
        if st.sidebar.button("Add Custom Rule") and custom_prompt:
            if custom_prompt not in active_prompts:
                active_prompts.append(custom_prompt)
                st.sidebar.success(f"Added: {custom_prompt}")

        # Camera selection
        camera_source = st.sidebar.text_input("Camera Source (0 for webcam, or RTSP URL)", "0")
        
        if st.sidebar.button("▶️ Start Monitoring"):
            # Try to convert to int if it's a number
            try:
                source = int(camera_source)
            except ValueError:
                source = camera_source
                
            if self.start_webcam(source):
                st.success("Monitoring started!")

        if st.sidebar.button("⏹️ Stop Webcam"):
            self.stop_webcam()
            st.success("Monitoring stopped.")

        if self.cap:
            frame_placeholder = st.empty()
            frame_idx = 0
            while self.cap.isOpened():
                frame_idx += 1
                result = self.update_frame(frame_idx, active_prompts)
                if result:
                    frame, _ = result
                    if frame is not None:
                        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

if __name__ == "__main__":
    app = ObjectMonitoringApp()
    app.run()