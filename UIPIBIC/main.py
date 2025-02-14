import cv2
import yaml
import numpy as np
from pathlib import Path
import customtkinter as ctk
from threading import Thread, Lock, Event
from queue import Queue
import pyttsx3
import traceback
from typing import Dict, Any, List, Tuple, Optional
import time
from datetime import datetime
import os
import sys
from ultralytics import YOLO
from tkinter import messagebox
from pynput import keyboard
from PIL import Image
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('navigation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ObjectDetector:
    """Enhanced object detection for navigation assistance."""

    def __init__(self, model_path: str):
        """Initialize the YOLO model with error handling."""
        try:
            self.model = YOLO(model_path)
            self.model.to('cpu')  # Explicitly set to CPU
            logger.info(f"Successfully loaded YOLO model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, frame: np.ndarray, confidence_threshold: float) -> Tuple[List, np.ndarray]:
        """Perform object detection with enhanced error handling."""
        if frame is None:
            raise ValueError("Input frame is None")
        
        try:
            results = self.model(frame, conf=confidence_threshold, verbose=False)[0]
            return self._process_detections(results, frame)
        except Exception as e:
            logger.error(f"Detection error: {e}")
            raise

    def _process_detections(self, results, frame: np.ndarray):
        """Process detections with improved position calculation and distance estimation."""
        processed_frame = frame.copy()
        detections = []

        frame_height, frame_width = frame.shape[:2]
        
        for det in results.boxes.data:
            x1, y1, x2, y2, conf, cls = det.tolist()
            label = results.names[int(cls)]

            # Bound box coordinates to frame dimensions
            x1, y1, x2, y2 = [int(max(0, coord)) for coord in [x1, y1, x2, y2]]
            x2 = min(x2, frame_width - 1)
            y2 = min(y2, frame_height - 1)

            # Calculate center position and relative size
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            box_width = x2 - x1
            box_height = y2 - y1
            relative_size = (box_width * box_height) / (frame_width * frame_height)

            # Enhanced position detection
            horizontal_pos = (
                "center" if 0.4 < center_x / frame_width < 0.6
                else "left" if center_x / frame_width <= 0.4
                else "right"
            )

            # Estimate relative distance based on size
            distance = (
                "very close" if relative_size > 0.5
                else "close" if relative_size > 0.25
                else "medium distance" if relative_size > 0.1
                else "far"
            )

            detections.append({
                "label": label,
                "position": horizontal_pos,
                "distance": distance,
                "confidence": round(conf, 2)
            })

            # Enhanced visualization
            color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{label} ({horizontal_pos}, {distance}, {conf:.2f})"
            cv2.putText(processed_frame, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return detections, processed_frame

class AudioFeedback:
    """Enhanced audio feedback manager with priority queue and filtering."""

    def __init__(self):
        """Initialize TTS engine with configurable properties."""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 175)
            self.engine.setProperty("volume", 0.9)
            self.queue = Queue()
            self.speaking = False
            self.stop_event = Event()
            self.last_messages = {}  # Store last message for each object type
            self.thread = Thread(target=self._worker, daemon=True)
            self.thread.start()
            logger.info("Audio feedback system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize audio feedback: {e}")
            raise

    def _worker(self):
        """Enhanced worker thread with message filtering."""
        while not self.stop_event.is_set():
            try:
                text = self.queue.get(timeout=1.0)
                if text is None:
                    break
                self.speaking = True
                self.engine.say(text)
                self.engine.runAndWait()
            except Queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Speech error: {e}")
            finally:
                self.speaking = False
                self.queue.task_done()

    def speak(self, text: str, priority: bool = False, object_type: str = None):
        """Enhanced speak method with duplicate filtering."""
        if object_type:
            current_time = time.time()
            if (object_type in self.last_messages and 
                current_time - self.last_messages[object_type]["time"] < 3.0 and
                text == self.last_messages[object_type]["text"]):
                return  # Skip duplicate messages

            self.last_messages[object_type] = {
                "text": text,
                "time": current_time
            }

        if priority:
            with self.queue.mutex:
                self.queue.queue.clear()
        self.queue.put(text)

    def stop(self):
        """Clean shutdown of audio feedback."""
        self.stop_event.set()
        self.engine.stop()
        with self.queue.mutex:
            self.queue.queue.clear()
        self.queue.put(None)
        self.thread.join(timeout=2.0)

class NavigationApp:
    """Enhanced main application class with additional features."""

    def __init__(self):
        """Initialize the application with enhanced error handling."""
        try:
            self.config = self.load_config()
            self.detector = ObjectDetector(self.config["yolo_model_path"])
            self.audio_feedback = AudioFeedback()
            self.running = False
            self.frame_lock = Lock()
            self.last_audio_time = time.time()
            self.frame_count = 0
            self.fps = 0
            self.last_fps_update = time.time()
            self.setup_ui()
            self.setup_shortcuts()
            logger.info("Navigation app initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize navigation app: {e}")
            raise

    def load_config(self) -> Dict[str, Any]:
        """Load and validate configuration."""
        try:
            script_dir = Path(__file__).resolve().parent
            config_path = script_dir / "config.yaml"

            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Validate required config fields
            required_fields = ["yolo_model_path", "camera_index", "confidence_threshold"]
            missing_fields = [field for field in required_fields if field not in config]
            if missing_fields:
                raise ValueError(f"Missing required config fields: {missing_fields}")

            return config
        except Exception as e:
            logger.error(f"Config loading error: {e}")
            raise

    def setup_ui(self):
        """Enhanced UI setup with additional controls."""
        self.window = ctk.CTk()
        self.window.title("Enhanced Navigation Assistance")
        self.window.geometry("1024x768")

        # Create main container
        main_container = ctk.CTkFrame(self.window)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Control panel
        control_panel = ctk.CTkFrame(main_container)
        control_panel.pack(side="top", fill="x", pady=5)

        self.toggle_button = ctk.CTkButton(
            control_panel, 
            text="Start", 
            command=self.toggle_detection,
            width=120
        )
        self.toggle_button.pack(side="left", padx=5)

        # FPS counter
        self.fps_label = ctk.CTkLabel(control_panel, text="FPS: 0")
        self.fps_label.pack(side="right", padx=5)

        # Video display
        self.video_label = ctk.CTkLabel(main_container, text="")
        self.video_label.pack(expand=True, pady=10)

        # Settings panel
        settings_panel = ctk.CTkFrame(main_container)
        settings_panel.pack(fill="x", pady=5)

        # Confidence threshold slider
        confidence_frame = ctk.CTkFrame(settings_panel)
        confidence_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(confidence_frame, text="Confidence:").pack(side="left", padx=5)
        
        self.confidence_slider = ctk.CTkSlider(
            confidence_frame, 
            from_=0, 
            to=1, 
            number_of_steps=100
        )
        self.confidence_slider.set(self.config["confidence_threshold"])
        self.confidence_slider.pack(side="left", fill="x", expand=True, padx=5)
        
        self.confidence_label = ctk.CTkLabel(
            confidence_frame, 
            text=f"{self.config['confidence_threshold']:.2f}"
        )
        self.confidence_label.pack(side="left", padx=5)

        # Status bar
        self.status_label = ctk.CTkLabel(main_container, text="Ready")
        self.status_label.pack(fill="x", pady=5)

        # Bind slider events
        self.confidence_slider.bind("<B1-Motion>", self.update_confidence_label)
        self.confidence_slider.bind("<ButtonRelease-1>", self.update_confidence_label)

    def update_confidence_label(self, event=None):
        """Update confidence threshold display."""
        value = self.confidence_slider.get()
        self.confidence_label.configure(text=f"{value:.2f}")

    def setup_shortcuts(self):
        """Setup enhanced keyboard shortcuts."""
        def on_press(key):
            try:
                if key == keyboard.KeyCode(char=' '):
                    self.toggle_detection()
                elif key == keyboard.Key.esc:
                    self.cleanup()
                    self.window.quit()
            except AttributeError:
                pass

        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()

    def toggle_detection(self):
        """Enhanced detection toggle with proper resource management."""
        if not self.running:
            try:
                self.camera = cv2.VideoCapture(self.config["camera_index"])
                if not self.camera.isOpened():
                    raise Exception("Could not open camera")
                
                # Set camera properties for better performance
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.camera.set(cv2.CAP_PROP_FPS, 30)
                
                self.running = True
                self.toggle_button.configure(text="Stop")
                self.detection_thread = Thread(target=self.detection_loop, daemon=True)
                self.detection_thread.start()
                self.audio_feedback.speak("Detection started", priority=True)
                self.update_status("Running")
                logger.info("Detection started")
            except Exception as e:
                logger.error(f"Failed to start detection: {e}")
                self.show_error(f"Error starting detection: {e}")
                self.running = False
        else:
            self.stop_detection()

    def stop_detection(self):
        """Clean shutdown of detection system."""
        self.running = False
        self.toggle_button.configure(text="Start")
        if hasattr(self, 'camera') and self.camera is not None:
            self.camera.release()
        self.audio_feedback.speak("Detection stopped", priority=True)
        self.update_status("Stopped")
        if hasattr(self, 'detection_thread') and self.detection_thread is not None:
            self.detection_thread.join(timeout=2.0)
        logger.info("Detection stopped")

    def detection_loop(self):
        """Enhanced detection loop with FPS calculation and error handling."""
        while self.running:
            try:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    logger.error("Failed to capture frame")
                    self.show_error("Failed to capture frame")
                    self.running = False
                    break

                # Update FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_update >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_fps_update = current_time
                    self.fps_label.configure(text=f"FPS: {self.fps}")

                # Perform detection
                detections, processed_frame = self.detector.detect(
                    frame, 
                    self.confidence_slider.get()
                )
                
                self.update_interface(processed_frame)
                self.handle_audio_feedback(detections)

            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                self.show_error(f"Detection error: {e}")
                self.running = False
                break

    def handle_audio_feedback(self, detections: List[Dict]):
        """Enhanced audio feedback with priority system."""
        current_time = time.time()
        if current_time - self.last_audio_time >= 2.0:
            if detections:
                # Sort detections by confidence
                sorted_detections = sorted(
                    detections, 
                    key=lambda x: x['confidence'], 
                    reverse=True
                )
                
                # Prioritize closer objects
                priority_detections = [
                    d for d in sorted_detections 
                    if d['distance'] in ['very close', 'close']
                ]

                if priority_detections:
                    # Announce high-priority detections first
                    scene_description = ", ".join(
                        [f"{d['label']} {d['distance']} on the {d['position']}" 
                         for d in priority_detections[:3]]  # Limit to top 3 priority detections
                    )
                    self.audio_feedback.speak(
                        f"Warning: {scene_description}", 
                        priority=True,
                        object_type="priority"
                    )
                else:
                    # Regular announcements for other detections
                    scene_description = ", ".join(
                        [f"{d['label']} on the {d['position']}" 
                         for d in sorted_detections[:5]]  # Limit to top 5 detections
                    )
                    self.audio_feedback.speak(
                        f"I see: {scene_description}",
                        object_type="regular"
                    )
            else:
                self.audio_feedback.speak(
                    "Path is clear",
                    object_type="status"
                )
            self.last_audio_time = current_time

    def update_interface(self, frame: np.ndarray):
        """Update the video display with enhanced error handling."""
        try:
            with self.frame_lock:
                # Resize frame for display while maintaining aspect ratio
                display_width = 800
                aspect_ratio = frame.shape[1] / frame.shape[0]
                display_height = int(display_width / aspect_ratio)
                
                resized_frame = cv2.resize(
                    frame, 
                    (display_width, display_height),
                    interpolation=cv2.INTER_AREA
                )
                
                frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                photo = ctk.CTkImage(img, size=(display_width, display_height))
                self.video_label.configure(image=photo)
                self.video_label.image = photo  # Keep a reference
        except Exception as e:
            logger.error(f"Error updating interface: {e}")
            self.show_error("Error updating display")

    def update_status(self, message: str):
        """Update status with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_label.configure(text=f"[{timestamp}] {message}")
        logger.info(f"Status update: {message}")

    def show_error(self, message: str):
        """Enhanced error display with logging."""
        logger.error(message)
        self.window.after(0, lambda: messagebox.showerror("Error", message))
        self.update_status(f"Error: {message}")

    def cleanup(self):
        """Enhanced cleanup with proper resource management."""
        try:
            logger.info("Starting cleanup...")
            
            # Stop detection if running
            if self.running:
                self.stop_detection()
            
            # Clean up camera
            if hasattr(self, 'camera') and self.camera is not None:
                self.camera.release()
                logger.info("Camera released")
            
            # Stop audio feedback
            if hasattr(self, 'audio_feedback'):
                self.audio_feedback.stop()
                logger.info("Audio feedback stopped")
            
            # Stop keyboard listener
            if hasattr(self, 'keyboard_listener'):
                self.keyboard_listener.stop()
                logger.info("Keyboard listener stopped")
            
            # Destroy window
            if hasattr(self, 'window'):
                self.window.destroy()
                logger.info("Window destroyed")
            
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def main():
    """Main entry point with enhanced error handling."""
    try:
        # Set up application-wide exception handler
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        
        sys.excepthook = handle_exception
        
        # Create and run application
        app = NavigationApp()
        app.window.mainloop()
    except Exception as e:
        logger.critical(f"Fatal error during initialization: {e}")
        messagebox.showerror("Fatal Error", str(e))
    finally:
        try:
            app.cleanup()
        except NameError:
            pass

if __name__ == "__main__":
    main()