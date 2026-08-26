import cv2
import dlib
import numpy as np
import time
from scipy.spatial import distance as dist
from config import settings

class IrisCapture:
    """
    Handles real-time iris capture, liveness detection, and normalization.
    """
    def __init__(self, landmark_path=None):
        self.detector = dlib.get_frontal_face_detector()
        path = landmark_path or settings.DLIB_LANDMARK_PATH
        try:
            self.predictor = dlib.shape_predictor(path)
        except Exception as e:
            print(f"Warning: Could not load dlib model from {path}. Ensure file exists.")
            self.predictor = None
            
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # State for liveness
        self.blink_count = 0
        self.last_ear = 1.0
        self.eye_state_history = []
        self.iris_center_history = []
        
    def get_eye_aspect_ratio(self, eye_points):
        """Compute Eye Aspect Ratio (EAR) for blink detection."""
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        C = dist.euclidean(eye_points[0], eye_points[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_iris_pupil(self, eye_roi):
        """
        Detect iris and pupil boundaries using Hough Transform.
        Daugman's integro-differential operator simplified with Hough.
        """
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        blurred = cv2.medianBlur(gray, 5)
        
        # Detect Pupil (inner, darker)
        # Relaxed param2 for better sensitivity on standard webcams
        pupil_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                        param1=50, param2=20, minRadius=8, maxRadius=45)
        
        # Detect Iris Boundary (outer)
        iris_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                       param1=50, param2=25, minRadius=40, maxRadius=120)
        
        if pupil_circles is not None and iris_circles is not None:
            p = np.round(pupil_circles[0, 0]).astype("int")
            i = np.round(iris_circles[0, 0]).astype("int")
            
            # Validation: iris > pupil and concentricity
            dist_centers = np.sqrt((p[0]-i[0])**2 + (p[1]-i[1])**2)
            if i[2] > p[2] and dist_centers < settings.CONCENTRIC_TOLERANCE:
                return (p[0], p[1], p[2]), (i[0], i[1], i[2])
        
        return None, None

    def normalize_iris(self, image, pupil_circle, iris_circle, M=64, N=512):
        """
        Daugman's Rubber Sheet Model: Map annular iris region to rectangular strip.
        M: radial resolution, N: angular resolution.
        """
        xp, yp, rp = pupil_circle
        xi, yi, ri = iris_circle
        
        normalized = np.zeros((M, N), dtype=np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        
        theta = np.linspace(0, 2*np.pi, N)
        for j in range(N):
            # Points on pupil boundary
            xp_theta = xp + rp * np.cos(theta[j])
            yp_theta = yp + rp * np.sin(theta[j])
            
            # Points on iris boundary
            xi_theta = xi + ri * np.cos(theta[j])
            yi_theta = yi + ri * np.sin(theta[j])
            
            for i in range(M):
                # Linear interpolation between pupil and iris boundary
                r = i / M
                x = (1 - r) * xp_theta + r * xi_theta
                y = (1 - r) * yp_theta + r * yi_theta
                
                # Bi-linear interpolation or direct mapping
                if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                    normalized[i, j] = gray[int(y), int(x)]
        
        return normalized

    def check_quality(self, normalized_strip):
        """Biometric quality checks."""
        # Sharpness using Laplacian variance
        sharpness = cv2.Laplacian(normalized_strip, cv2.CV_64F).var()
        
        # Basic visibility check (simulated by checking if pixel values are too low/high)
        # In a real scenario, this would involve eyelid occlusion detection
        visibility = 1.0 - (np.sum(normalized_strip < 10) / normalized_strip.size)
        
        is_good = sharpness > settings.SHARPNESS_THRESHOLD and visibility > settings.IRIS_VISIBILITY_THRESHOLD
        return is_good, sharpness, visibility

    def detect_from_static_image(self, frame):
        """Robust detection for a static image: Face -> Eye Landmarks -> Iris."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if not faces:
            return None
            
        # Take the largest face
        face = max(faces, key=lambda f: (f.right()-f.left()) * (f.bottom()-f.top()))
        landmarks = self.predictor(gray, face)
        
        # Try both eyes and pick the best
        best_normalized = None
        best_score = -1
        
        for i_range in [range(36, 42), range(42, 48)]:
            eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in i_range])
            x_min, y_min = np.min(eye_pts, axis=0) - 30
            x_max, y_max = np.max(eye_pts, axis=0) + 30
            
            eye_roi = frame[max(0, y_min):y_max, max(0, x_min):x_max]
            if eye_roi.size == 0: continue
            
            pupil, iris = self.detect_iris_pupil(eye_roi)
            if pupil and iris:
                normalized = self.normalize_iris(eye_roi, pupil, iris)
                is_good, sharp, vis = self.check_quality(normalized)
                score = sharp + vis * 100
                if score > best_score:
                    best_score = score
                    best_normalized = normalized
                    
        return best_normalized

    def run_pipeline(self):
        """
        Main loop for capturing the best quality iris frame with liveness detection.
        """
        cap = cv2.VideoCapture(0)
        best_frame = None
        best_score = -1
        captured_count = 0
        start_time = time.time()
        
        print("Starting Iris Capture... Please look at the camera and blink.")
        
        while captured_count < 10:
            if time.time() - start_time > 15:
                print("Liveness session timed out.")
                break
                
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            
            for face in faces:
                landmarks = self.predictor(gray, face)
                
                # Left Eye (points 36-41), Right Eye (points 42-47)
                left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
                right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
                
                # Blink Detection (EAR)
                ear_l = self.get_eye_aspect_ratio(left_eye_pts)
                ear_r = self.get_eye_aspect_ratio(right_eye_pts)
                avg_ear = (ear_l + ear_r) / 2.0
                
                if avg_ear < 0.2:
                    self.blink_count += 1
                
                # Process each eye
                found_any_eye = False
                for eye_pts in [left_eye_pts, right_eye_pts]:
                    # Increase ROI padding for easier capture
                    x_min, y_min = np.min(eye_pts, axis=0) - 30
                    x_max, y_max = np.max(eye_pts, axis=0) + 30
                    
                    eye_roi = frame[max(0, y_min):y_max, max(0, x_min):x_max]
                    if eye_roi.size == 0: continue
                    
                    pupil, iris = self.detect_iris_pupil(eye_roi)
                    if pupil and iris:
                        normalized = self.normalize_iris(eye_roi, pupil, iris)
                        is_good, sharp, vis = self.check_quality(normalized)
                        
                        if is_good:
                            score = sharp + vis * 100
                            if score > best_score:
                                best_score = score
                                best_frame = normalized
                            captured_count += 1
                            print(f"\rCaptured {captured_count}/10 quality frames...", end="")
                            found_any_eye = True
                    
                    # Debug visualization
                    cv2.polylines(frame, [eye_pts], True, (0, 255, 0), 1)
                
                if not found_any_eye and time.time() - start_time > 5:
                    print("\rLooking for eyes... (Try move closer or improve lighting)", end="")
            
            # Show face landmark for debugging
            for face in faces:
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

            cv2.imshow("Iris Capture Preview - Look & Blink", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        print("\n")
        cap.release()
        cv2.destroyAllWindows()
        
        # Verify liveness (at least one blink)
        if self.blink_count > 0 and best_frame is not None:
            return best_frame
        else:
            return None

if __name__ == "__main__":
    # Test execution
    scanner = IrisCapture()
    iris_strip = scanner.run_pipeline()
    if iris_strip is not None:
        cv2.imwrite("captured_iris.png", iris_strip)
        print("Iris captured successfully.")
    else:
        print("Failed to capture iris or liveness check failed.")
