import pygame
from pygame.locals import *
import cv2
import numpy as np
import gi
import time

gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp

class JoystickHandler:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        else:
            raise Exception("No joystick connected")

    def handle_x(self):
        pygame.event.pump()  # Ensure the joystick state is updated
        raw_value = self.joystick.get_axis(0)
        scaled_value = raw_value * 135  # Scale from -1..1 to -135..135
        return scaled_value

    def gear(self):
        pygame.event.pump()  # Ensure the joystick state is updated
        events = pygame.event.get()
        for event in events:
            if event.type == JOYBUTTONDOWN:
                if event.button == 4:
                    return "gear:1"  # 전진
                elif event.button == 3:
                    return "gear:-1"  # 후진
        return "gear:0"  # 중립

    def accel(self):
        pygame.event.pump()  # Ensure the joystick state is updated
        raw_value = self.joystick.get_axis(5)
        scaled_value = (raw_value + 1) / 2  # Scale from -1..1 to 0..1
        return scaled_value

    def brake(self):
        pygame.event.pump()  # Ensure the joystick state is updated
        raw_value = self.joystick.get_axis(4)
        scaled_value = (raw_value + 1) / 2  # Scale from -1..1 to 0..1
        return scaled_value

class VideoStream:
    def __init__(self):
        Gst.init(None)
        self.pipeline = Gst.parse_launch(
            'udpsrc port=5001 ! application/x-rtp, payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=RGB ! appsink name=appsink0 emit-signals=True sync=False'
        )
        self.pipeline.set_state(Gst.State.PLAYING)
        self.appsink = self.pipeline.get_by_name('appsink0')

    def new_sample(self):
        sample = self.appsink.emit('pull-sample')
        if sample:
            buf = sample.get_buffer()
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            height = structure.get_value('height')
            width = structure.get_value('width')

            success, mapinfo = buf.map(Gst.MapFlags.READ)
            if not success:
                return None

            try:
                data = mapinfo.data
                format_str = structure.get_name()
                if format_str == 'video/x-raw':
                    format = structure.get_value('format')
                    if format == 'RGB':
                        expected_size = width * height * 3
                    elif format == 'GRAY8':
                        expected_size = width * height
                    else:
                        print(f"Unsupported format: {format}")
                        return None

                if len(data) != expected_size:
                    print(f"Data size mismatch: expected {expected_size}, got {len(data)}")
                    return None

                if format == 'RGB':
                    arr = np.ndarray(
                        (height, width, 3),
                        buffer=data,
                        dtype=np.uint8
                    )
                elif format == 'GRAY8':
                    arr = np.ndarray(
                        (height, width),
                        buffer=data,
                        dtype=np.uint8
                    )
            finally:
                buf.unmap(mapinfo)

            return arr
        return None

    def take_screenshot(self, frame):
        timestamp = int(time.time())
        filename = f"screenshot_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"Screenshot taken and saved as {filename}")

    def close(self):
        self.pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()

def read_joystick():
    joystick_handler = JoystickHandler()
    handle_x = joystick_handler.handle_x()
    brake = joystick_handler.brake()
    accel = joystick_handler.accel()
    gear = joystick_handler.gear()
    return handle_x, gear, accel, brake
