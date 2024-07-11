from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import pygame
from pygame.locals import *
import socket
import cv2
import numpy as np
import gi
import base64
from threading import Thread
from ultralytics import YOLO
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp

app = Flask(__name__)
socketio = SocketIO(app)

# Global variables
raspberry_pi_ip = 'input your raspberry_pi_ip'
port = 'input your port'
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# GStreamer initialization
Gst.init(None)
pipeline = Gst.parse_launch(
    'udpsrc port=5055 ! application/x-rtp, payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=RGB ! appsink name=appsink0 emit-signals=True sync=False'
)
pipeline.set_state(Gst.State.PLAYING)
appsink = pipeline.get_by_name('appsink0')

# YOLO model initialization
model = YOLO('input your model')

@app.route('/')
def index():
    return render_template('index.html')

def send_to_raspberry_pi(message):
    sock.sendto(message.encode(), (raspberry_pi_ip, port))

def scale_value(value, min_scale, max_scale):
    return value * (max_scale - min_scale) / 2

@socketio.on('connect')
def connect():
    print("Client connected")

@socketio.on('disconnect')
def disconnect():
    print("Client disconnected")

def start_joystick():
    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    try:
        while True:
            for event in pygame.event.get():
                if event.type == JOYAXISMOTION:
                    if event.axis == 0:
                        scaled_value = scale_value(event.value, -135, 135)
                        message = f"steering:{scaled_value}"
                        send_to_raspberry_pi(message)
                        socketio.emit('update_angle', {'angle': scaled_value})
                        print(message)
                    elif event.axis == 4:
                        message = f"brake:{event.value}"
                        send_to_raspberry_pi(message)
                    elif event.axis == 5:
                        message = f"accelerate:{event.value}"
                        send_to_raspberry_pi(message)
                elif event.type == JOYBUTTONDOWN:
                    if event.button == 4:
                        message = "gear:1"
                    elif event.button == 3:
                        message = "gear:-1"
                    send_to_raspberry_pi(message)
    except KeyboardInterrupt:
        print("Program terminated.")
    finally:
        pygame.quit()
        sock.close()

def new_sample(sink):
    sample = sink.emit('pull-sample')
    if sample:
        buf = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        height = structure.get_value('height')
        width = structure.get_value('width')
        # Get buffer data
        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            return None
        try:
            # Extract buffer data
            data = mapinfo.data
            # Determine the expected size based on format
            format_str = structure.get_name()
            if format_str == 'video/x-raw':
                format = structure.get_value('format')
                if format == 'RGB':
                    expected_size = width * height * 3
                else:
                    print(f"Unsupported format: {format}")
                    return None
            if len(data) != expected_size:
                print(f"Data size mismatch: expected {expected_size}, got {len(data)}")
                return None
            # Convert buffer to numpy array
            arr = np.ndarray(
                (height, width, 3),
                buffer=data,
                dtype=np.uint8
            ).copy()  # Make sure the array is writable
        finally:
            buf.unmap(mapinfo)
        return arr
    return None

def video_stream():
    try:
        while True:
            frame = new_sample(appsink)
            if frame is not None:
                # YOLO model for object detection
                results = model(frame)
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = model.names[int(box.cls[0])]
                    confidence = box.conf[0]
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Put label and confidence
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_data = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('video_frame', {'frame': frame_data})
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    Thread(target=start_joystick).start()
    Thread(target=video_stream).start()
    socketio.run(app, host='0.0.0.0', port=5000)