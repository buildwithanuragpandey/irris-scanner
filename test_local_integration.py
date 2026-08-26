import cv2
import base64
import requests
import time
from iris_capture import IrisCapture

def test_local_flow():
    """
    Simulates the full flow: Capture -> Enroll -> Verify.
    Ensure api.py is running in another terminal.
    """
    API_URL = "http://localhost:8000"
    VOTER_ID = "test_voter_001"
    AUTH_TOKEN = "valid-test-token"
    HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}"}
    
    scanner = IrisCapture()
    
    print("--- Phase 1: Capture Iris for Enrollment ---")
    print("Look at the camera and blink when prompted.")
    enroll_strip = scanner.run_pipeline()
    
    if enroll_strip is None:
        print("Failed to capture iris for enrollment.")
        return
        
    _, buffer = cv2.imencode('.png', enroll_strip)
    enroll_b64 = base64.b64encode(buffer).decode('utf-8')
    
    print("\n--- Phase 2: Enrolling ---")
    enroll_resp = requests.post(
        f"{API_URL}/enroll",
        json={"voter_id": VOTER_ID, "image_base64": enroll_b64, "eye_side": "left"},
        headers=HEADERS
    )
    print(f"Enroll Response: {enroll_resp.json()}")

    print("\nWait 2 seconds...")
    time.sleep(2)

    print("\n--- Phase 3: Capture Iris for Verification ---")
    verify_strip = scanner.run_pipeline()
    
    if verify_strip is None:
        print("Failed to capture iris for verification.")
        return

    _, buffer = cv2.imencode('.png', verify_strip)
    verify_b64 = base64.b64encode(buffer).decode('utf-8')

    print("\n--- Phase 4: Verifying ---")
    verify_resp = requests.post(
        f"{API_URL}/verify",
        json={"voter_id": VOTER_ID, "image_base64": verify_b64, "eye_side": "left"},
        headers=HEADERS
    )
    print(f"Verify Response: {verify_resp.json()}")

if __name__ == "__main__":
    try:
        test_local_flow()
    except Exception as e:
        print(f"Error: {e}")
        print("Is api.py running? Start it with: uvicorn api:app --reload")
