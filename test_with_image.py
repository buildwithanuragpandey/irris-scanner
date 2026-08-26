import cv2
import base64
import requests
import sys
import os

def test_with_image(voter_id, image_path, mode="enroll"):
    """
    Tests enrollment or verification using a static image file.
    """
    API_URL = "http://localhost:8000"
    AUTH_TOKEN = "valid-test-token"
    HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}"}
    
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found.")
        return

    # Load and encode image
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    endpoint = "/enroll" if mode == "enroll" else "/verify"
    payload = {
        "voter_id": voter_id,
        "image_base64": img_b64,
        "eye_side": "left"
    }
    
    print(f"--- Sending to {endpoint} ---")
    resp = requests.post(f"{API_URL}{endpoint}", json=payload, headers=HEADERS)
    
    if resp.status_code == 200:
        print(f"Success! Response: {resp.json()}")
    else:
        print(f"Failed (Status {resp.status_code}): {resp.text}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 test_with_image.py <enroll|verify> <voter_id> <path_to_image>")
        print("Example: python3 test_with_image.py enroll user123 my_eye.jpg")
    else:
        mode = sys.argv[1]
        voter_id = sys.argv[2]
        img_path = sys.argv[3]
        test_with_image(voter_id, img_path, mode)
