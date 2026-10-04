import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import pygetwindow as gw
import screen_brightness_control as sbc
import numpy as np
import math

def main():
    # Initialize MediaPipe PoseLandmarker
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        num_poses=1)
    
    # We use vision.PoseLandmarker explicitly
    detector = vision.PoseLandmarker.create_from_options(options)

    # Capture Video from Webcam
    cap = cv2.VideoCapture(0)

    # State variables
    calibration_duration = 5.0 # seconds
    slouch_time_limit = 10.0 # seconds
    slouch_threshold = 0.75 # 75% of baseline

    start_time = time.time()
    baseline_distances = []
    baseline_distance = None

    slouch_start_time = None
    is_punished = False

    # Store original brightness to restore later
    try:
        original_brightness = sbc.get_brightness()[0]
    except Exception as e:
        print(f"Warning: Could not get screen brightness: {e}")
        original_brightness = 100

    print("--- ShrimpDetector Started ---")
    print("Sit up straight! Calibrating for 5 seconds...")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the image to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Detect pose
        detection_result = detector.detect(mp_image)

        current_time = time.time()
        elapsed_time = current_time - start_time

        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            landmarks = detection_result.pose_landmarks[0]

            # Draw circles on ears and shoulders
            h, w, _ = image.shape
            for idx in [7, 8, 11, 12]:
                landmark = landmarks[idx]
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (cx, cy), 5, (245, 117, 66), -1)

            # Left ear = 7, Right ear = 8, Left shoulder = 11, Right shoulder = 12
            left_ear = landmarks[7]
            right_ear = landmarks[8]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]

            # Calculate average Y coordinates
            avg_ear_y = (left_ear.y + right_ear.y) / 2.0
            avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0

            # Measure vertical drop of the head
            vertical_dist = avg_shoulder_y - avg_ear_y
            
            # Measure shoulder width to create a scale-invariant ratio
            shoulder_width = math.sqrt((left_shoulder.x - right_shoulder.x)**2 + (left_shoulder.y - right_shoulder.y)**2)
            if shoulder_width < 0.01:
                shoulder_width = 0.01

            # Scale-invariant posture metric (removes the "distance to camera" bug)
            current_distance = vertical_dist / shoulder_width

            if elapsed_time < calibration_duration:
                # Calibration Phase
                baseline_distances.append(current_distance)
                cv2.putText(image, f"Calibrating... {calibration_duration - elapsed_time:.1f}s", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                if baseline_distance is None:
                    if baseline_distances:
                        baseline_distance = np.mean(baseline_distances)
                        print(f"Calibration complete! Baseline distance: {baseline_distance:.4f}")
                    else:
                        print("Failed to detect pose during calibration. Restarting calibration.")
                        start_time = time.time()
                        continue

                # Monitoring Phase
                cv2.putText(image, f"Baseline: {baseline_distance:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f"Current: {current_distance:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                threshold_distance = baseline_distance * slouch_threshold

                if current_distance < threshold_distance:
                    # User is slouching
                    if slouch_start_time is None:
                        slouch_start_time = time.time()
                    
                    slouch_elapsed = time.time() - slouch_start_time
                    cv2.putText(image, f"Slouching! {slouch_time_limit - slouch_elapsed:.1f}s to punishment", 
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if slouch_elapsed >= slouch_time_limit and not is_punished:
                        print("PUNISHMENT INITIATED!")
                        
                        # 1. Minimize Active Window
                        try:
                            active_window = gw.getActiveWindow()
                            if active_window is not None:
                                active_window.minimize()
                        except Exception as e:
                            print(f"Error minimizing window: {e}")

                        # 2. Dim Screen
                        try:
                            sbc.set_brightness(10)
                        except Exception as e:
                            print(f"Error setting brightness: {e}")

                        is_punished = True
                else:
                    # User is sitting up straight
                    if slouch_start_time is not None:
                        slouch_start_time = None
                    
                    if is_punished:
                        print("Good posture restored. Reverting punishment.")
                        try:
                            sbc.set_brightness(original_brightness)
                        except Exception as e:
                            print(f"Error restoring brightness: {e}")
                        is_punished = False
                    
                    cv2.putText(image, "Posture: OK", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
             if elapsed_time < calibration_duration:
                 cv2.putText(image, "No pose detected for calibration!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the image
        cv2.imshow('ShrimpDetector', image)

        # Exit on pressing 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if is_punished:
        try:
            sbc.set_brightness(original_brightness)
        except:
            pass

if __name__ == "__main__":
    main()
