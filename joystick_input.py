import pygame
from pygame.locals import *

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

def read_joystick():
    joystick_handler = JoystickHandler()
    handle_x = joystick_handler.handle_x()
    brake = joystick_handler.brake()
    accel = joystick_handler.accel()
    gear = joystick_handler.gear()
    return handle_x, gear, accel, brake