"""
Raspberry Pi Palm Capture Terminal Client.

Supports two deployment modes:
  Mode 1 (v1 Direct POST): Pi captures frame directly from camera and POSTs multipart form data to FastAPI endpoints.
  Mode 2 (v2 WebSocket Relay): Pi connects to backend WS relay (ws://<host>/ws/session/{token}),
          streams live low-res preview frames for UI feedback, and posts high-res multi-frame capture when triggered.
"""

import argparse
import asyncio
import base64
import json
import time
import cv2
import requests

try:
    import websockets
except ImportError:
    websockets = None


def get_camera_frame(cap):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to grab camera frame")
    return frame


def mode_direct_post(api_url: str, merchant_id: str):
    print(f"[*] Starting Direct POST Mode targeting {api_url}...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Error: Could not open camera device.")
        return

    print("Press SPACE to capture and scan palm, or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, "Press SPACE to Scan Palm", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Raspberry Pi Palm Terminal", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            print("[*] Capturing frame for identification...")
            _, img_encoded = cv2.imencode('.jpg', frame)
            files = {'palm_photo': ('capture.jpg', img_encoded.tobytes(), 'image/jpeg')}
            data = {'merchant_id': merchant_id}

            try:
                res = requests.post(f"{api_url}/session/identify", files=files, data=data)
                print(f"[>] Server Response ({res.status_code}):", res.json())
            except Exception as e:
                print(f"[!] Request error: {e}")

    cap.release()
    cv2.destroyAllWindows()


async def mode_websocket_relay(ws_url: str):
    if websockets is None:
        print("[!] websockets package not installed. Run pip install websockets")
        return

    print(f"[*] Connecting to WebSocket Relay Hub at {ws_url}...")
    cap = cv2.VideoCapture(0)

    async with websockets.connect(ws_url) as ws:
        print("[✓] Connected to relay hub! Streaming video frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.05)
                continue

            # Resize frame for live UI streaming feed
            small = cv2.resize(frame, (320, 240))
            _, buf = cv2.imencode('.jpg', small, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            b64_frame = base64.b64encode(buf).decode('utf-8')

            msg = {
                "type": "video_frame",
                "frame": b64_frame,
                "detection_state": "positioning",
                "feedback": "Align hand inside frame"
            }
            await ws.send(json.dumps(msg))

            # Listen for incoming commands (non-blocking sleep)
            try:
                resp_text = await asyncio.wait_for(ws.recv(), timeout=0.03)
                resp = json.loads(resp_text)
                print("[<] Relay Command Received:", resp)
            except asyncio.TimeoutError:
                pass

            await asyncio.sleep(0.04)

    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Palm Capture Terminal Client")
    parser.add_argument("--api-url", default="http://localhost:8000", help="FastAPI backend URL")
    parser.add_argument("--merchant-id", default="merch_terminal_pi01", help="Merchant Terminal ID")
    parser.add_argument("--relay-token", default=None, help="WebSocket pairing token for Relay Mode")
    args = parser.parse_args()

    if args.relay_token:
        ws_url = f"ws://{args.api_url.replace('http://', '').replace('https://', '')}/ws/session/{args.relay_token}"
        asyncio.run(mode_websocket_relay(ws_url))
    else:
        mode_direct_post(args.api_url, args.merchant_id)


if __name__ == "__main__":
    main()
