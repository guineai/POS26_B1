import pygame
from pygame.locals import *
import socket

# Pygame 초기화
pygame.init()
pygame.joystick.init()

# 첫 번째 조이스틱 초기화
joystick = pygame.joystick.Joystick(0)
joystick.init()

# UDP 소켓 설정
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
raspberry_pi_ip = '192.168.0.3'  # 라즈베리파이의 IP 주소로 변경
port = 12345  # 라즈베리파이에서 수신 대기 중인 포트 번호

def send_to_raspberry_pi(message):
    sock.sendto(message.encode(), (raspberry_pi_ip, port))

try:
    while True:
        for event in pygame.event.get():
            if event.type == JOYAXISMOTION:
                if event.axis == 0:  # 핸들 입력
                    message = f"steering:{event.value}"
                elif event.axis == 4:  # 브레이크 페달
                    message = f"brake:{event.value}"
                elif event.axis == 5:  # 엑셀 페달
                    message = f"accelerate:{event.value}"
                send_to_raspberry_pi(message)
            elif event.type == JOYBUTTONDOWN:
                # 기어 위치 확인 및 전송
                if event.button == 4:
                    message = "gear:1"  # 전진
                elif event.button == 3:
                    message = "gear:-1"  # 후진
                send_to_raspberry_pi(message)
except KeyboardInterrupt:
    print("프로그램이 종료되었습니다.")
finally:
    pygame.quit()
    sock.close()
