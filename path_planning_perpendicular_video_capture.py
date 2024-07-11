import cv2
import matplotlib.pyplot as plt
from here_lineplease import calculate_steering_angle
from google.cloud import texttospeech
from playsound import playsound
import time
import os
import sys
import socket

# Set up Google Cloud Text-to-Speech client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\\Users\\user\\desktop\\code\\algorithm\\sweett.json"
client = texttospeech.TextToSpeechClient()


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
raspberry_pi_ip = '192.168.0.31'
port = 12345


def send_to_raspberry_pi(message):
    sock.sendto(message.encode(), (raspberry_pi_ip, port))
    
def get_output_path(step_number):
    return r"C:\\Users\\user\\desktop\\code\\algorithm\\output_step_{}.mp3".format(step_number)

def speak(text, step_number):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",  # 한국어로 설정
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    output_path = get_output_path(step_number)  # 단계별 파일 경로 가져오기
    with open(output_path, "wb") as out:
        out.write(response.audio_content)
    print(f"Audio content written to file: {output_path}")

    # Play the audio using playsound
    playsound(output_path)

# 현재 디렉토리를 sys.path에 추가하여 모듈을 찾을 수 있도록 합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# from joystick_input import read_joystick
from capture_module import VideoStream, read_joystick


def get_center_coordinates(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    height, width = image.shape[:2]
    center_x = width // 2
    center_y = height // 2
    return center_x, center_y

def cm_to_pixels(cm, dpi=72):
    inches = cm / 2.54
    pixels = inches * dpi
    return int(pixels)

speak("1. 탑뷰 영상을 참고해서 차의 중앙이 주차 슬롯 폭의 중간부분까지 오도록 직진하세요", 1)

# Video stream 초기화
video_stream = VideoStream()

# 엑셀 값을 받아오는 함수 호출 (엑셀 값이 처음으로 생기고 나서 3초 동안 엑셀 값이 없으면 다음 단계로 넘어감)
start_time = None
accel_active = False
while True:
    _, _, accel, _ = read_joystick()
    # print(f"Steering: {handle_x:.2f}, Brake: {brake:.2f}, Accelerate: {accel:.2f}, Gear: {gear}")

    if accel > 0 or accel < 0:
        print("차가 움직이는 중입니다...")
        send_to_raspberry_pi(f'accelerate:{accel}')
        accel_active = True
        start_time = None  # 엑셀 값이 0보다 크면 시간을 초기화
    else:
        if accel_active and start_time is None:
            start_time = time.time()  # 엑셀 값이 0이 된 시간을 기록
        elif start_time is not None and time.time() - start_time >= 3:
            print("엑셀 값을 3초 이상 0으로 유지했습니다. 다음 단계로 넘어갑니다.")
            break
    time.sleep(0.1)  # 짧은 시간 동안 대기

speak("2. 핸들을 왼쪽으로 최대한 꺾으세요", 2)

while True:
    handle_x, _, _, _ = read_joystick()
    # print(f"Steering: {handle_x:.2f}, Brake: {brake:.2f}, Accelerate: {accel:.2f}, Gear: {gear}")

    if handle_x <= -135:
        speak("잘하셨어요! 3. 이제 천천히 엑셀을 밟아서 나아가세요", 3)
        break
    elif handle_x > 0:
        speak("핸들을 반대로 꺾으셨어요. 다시 꺾으세요", 3)
    time.sleep(0.1)  # 짧은 시간 동안 대기

# 엑셀 값을 받아오는 함수 호출 (엑셀 값이 처음으로 생기고 나서 3초 동안 엑셀 값이 없으면 다음 단계로 넘어감)
start_time = None
accel_active = False
while True:
    _, _, accel, _ = read_joystick()
    # print(f"Steering: {handle_x:.2f}, Brake: {brake:.2f}, Accelerate: {accel:.2f}, Gear: {gear}")

    if accel != 0:
        print("차가 움직이는 중입니다...")
        accel_active = True
        start_time = None  # 엑셀 값이 0보다 크면 시간을 초기화
    else:
        if accel_active and start_time is None:
            start_time = time.time()  # 엑셀 값이 0이 된 시간을 기록
        elif start_time is not None and time.time() - start_time >= 3:
            print("엑셀 값을 3초 이상 0으로 유지했습니다. 다음 단계로 넘어갑니다.")
            break
    time.sleep(0.1)  # 짧은 시간 동안 대기

robot_length_cm = 24.0
wheelbase_cm = 12.0


# Video stream에서 이미지를 캡쳐
frame = video_stream.new_sample()

if frame is not None:
    # 해당 경로에 captured_image.jpg 이름으로 캡쳐파일 저장
    image_path = "C:\\Users\\user\\desktop\\code\\algorithm\\captured_image.jpg"
    cv2.imwrite(image_path, frame)
else:
    raise FileNotFoundError("비디오 스트림에서 이미지를 캡쳐할 수 없습니다.")

# image_path = "C:\\Users\\user\\desktop\\code\\algorithm\\IMG_4440.jpg"


steering_angle, result_image_path = calculate_steering_angle(image_path, wheelbase_cm)

if steering_angle is not None:
    if steering_angle >= 32.74:
        speak("4. 현재 위치에서는 최적 경로로 주차할 수 없습니다.", 4)
    elif steering_angle >= 0:
        speak(f"4. 핸들을 오른쪽으로 {steering_angle:.2f} 도 꺾으세요", 4)
else:
    speak("각도를 산출하지 못했습니다.", 4)

# # 사용자가 핸들을 지정된 각도로 맞출 때까지 대기
# while True:
#     handle_x, brake, accel, gear = read_joystick()
#     print(f"Steering: {handle_x:.2f}, Brake: {brake:.2f}, Accelerate: {accel:.2f}, Gear: {gear}")

#     if handle_x >= steering_angle:  # 일단 그냥 테스트 환경에서 쉽게 넘어가려고
#         speak("핸들을 정확히 맞췄습니다. 다음 단계로 넘어갑니다.", 5)
#         break
#     else:
#         print(f"현재 핸들 값: {handle_x}, 목표 핸들 값: {steering_angle}")
#     time.sleep(1)  # 짧은 시간 동안 대기


    # 사용자가 핸들을 지정된 각도로 맞출 때까지 대기
handle_start_time = None
while True:
    handle_x, _, _, _ = read_joystick()
    if steering_angle - 0.5 <= handle_x <= steering_angle + 0.5:
        if handle_start_time is None:
            handle_start_time = time.time()
        elif time.time() - handle_start_time >= 3: # 3초간 목표 조향각의 +-0.5도 사이로 맞추면 통과
            speak("핸들을 정확히 맞췄습니다. 다음 단계로 넘어갑니다.", 5)
            break
    else:
        handle_start_time = None
        print(f"현재 핸들 값: {handle_x}, 목표 핸들 값: {steering_angle}")

    time.sleep(0.1)  


# speak("5. 이제 후진기어로 바꾸세요", 6)

# previous_gear = None
# while True:
#     _, gear, _, _ = read_joystick()
#     # print(f"Steering: {handle_x:.2f}, Brake: {brake:.2f}, Accelerate: {accel:.2f}, Gear: {gear}")

#     if gear != previous_gear:
#         previous_gear = gear
#         if gear == -1:
#             speak("잘하셨어요", 7)
#             break
#         else:
#             print("기어를 잘못 설정하셨어요. 후진기어로 변경하세요. 후진기어는 아래로 내리면 돼요")
#     time.sleep(0.1)  # 짧은 시간 동안 대기

speak("6. 자 이제 후진할게요! 엑셀을 밟고 주차 슬롯에 차가 중간부분까지 들어오면 멈추세요", 6)


# 엑셀 값을 받아오는 함수 호출 (엑셀 값이 처음으로 생기고 나서 3초 동안 엑셀 값이 없으면 다음 단계로 넘어감)
start_time = None
accel_active = False
while True:
    handle_x, brake, accel, gear = read_joystick()
    # print(f"Steering: {handle_x:.2f}, Brake: {brake:.2f}, Accelerate: {accel:.2f}, Gear: {gear}")

    if accel != 0:
        print("차가 움직이는 중입니다...")
        accel_active = True
        start_time = None  # 엑셀 값이 0보다 크면 시간을 초기화
    else:
        if accel_active and start_time is None:
            start_time = time.time()  # 엑셀 값이 0이 된 시간을 기록
        elif start_time is not None and time.time() - start_time >= 3:
            print("엑셀 값을 3초 이상 0으로 유지했습니다. 다음 단계로 넘어갑니다.")
            break
    time.sleep(0.1)  # 짧은 시간 동안 대기

speak("7. 핸들을 정렬하세요", 7)


start_time = None
while True:
    handle_x, brake, accel, gear = read_joystick()
    # print(f"Steering: {handle_x:.2f}, Brake: {brake:.2f}, Accelerate: {accel:.2f}, Gear: {gear}")

    if -3.0 <= handle_x <= 3.0:
        if start_time is None:
            start_time = time.time()
        elif time.time() - start_time >= 2:
            speak("잘하셨어요! 탑뷰를 보면서 차가 슬롯 안에 적절히 들어가도록 그대로 후진하세요!", 8)
            break
    else:
        start_time = None  # 핸들 값이 범위를 벗어나면 시간을 초기화
    time.sleep(0.1)  # 짧은 시간 동안 대기


# 엑셀 값을 받아오는 함수 호출 (엑셀 값이 처음으로 생기고 나서 3초 동안 엑셀 값이 없으면 다음 단계로 넘어감)
start_time = None
accel_active = False
while True:
    handle_x, brake, accel, gear = read_joystick()
    # print(f"Steering: {handle_x:.2f}, Brake: {brake:.2f}, Accelerate: {accel:.2f}, Gear: {gear}")

    if accel != 0:
        print("차가 움직이는 중입니다...")
        accel_active = True
        start_time = None  # 엑셀 값이 0보다 크면 시간을 초기화
    else:
        if accel_active and start_time is None:
            start_time = time.time()  # 엑셀 값이 0이 된 시간을 기록
        elif start_time is not None and time.time() - start_time >= 3:
            speak("주차를 종료합니다.", 9)
            break
    time.sleep(0.1)  # 짧은 시간 동안 대기