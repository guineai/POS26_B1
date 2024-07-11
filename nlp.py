import openai
import whisper
import pyaudio
import wave
from google.cloud import texttospeech
import os
from pydub import AudioSegment
import io
import numpy as np
import time
import warnings
from difflib import SequenceMatcher

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# OpenAI API 키 설정
openai.api_key = 'input your API Key'

# Whisper 모델 로드
whisper_model = whisper.load_model("base")

def record_audio(filename, max_duration=5, silence_threshold=500, silence_duration=1.5):
    chunk = 1024  # 버퍼 크기
    format = pyaudio.paInt16  # 16비트 포맷
    channels = 1  # 모노
    rate = 44100  # 샘플링 레이트

    p = pyaudio.PyAudio()

    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("Recording...")
    frames = []
    start_time = time.time()
    last_sound_time = start_time

    while True:
        data = stream.read(chunk)
        frames.append(data)
        audio_data = np.frombuffer(data, np.int16)
        silence = np.abs(audio_data).mean() < silence_threshold

        current_time = time.time()
        
        if not silence:
            last_sound_time = current_time
        
        # 종료 조건: max_duration을 넘고, 마지막 소리 이후로 silence_duration이 지남
        if current_time - start_time > max_duration and current_time - last_sound_time > silence_duration:
            break

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def recognize_speech_from_audio(filename):
    try:
        result = whisper_model.transcribe(filename)
        detected_language = result['language']
        text = result['text']
        print(f"Detected language: {detected_language}")
        print(f"Transcription result: {text}")
        return text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_gpt_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                 {
                    "role": "system", 
                    "content": (
                        "너는 주차 관련 질문에 답변해주는 음성비서다."
                        "운전자의 주차 관련 질문에 대해서 간략하게 답변해줘."
                        "'안녕하세요', '~에 근거하면'과 같이 정보가 아닌 미사여구는 자제해줘."
                        "한국의 도로교통법을 근거로 답변해야 하는데, numeratic information을 꼭 포함해서 답변해줘."
                        "답변을 학습된 프롬프트 그대로 읽지 말고, 조금씩 변형해서 말해줘. 예를 들어 특수문자는 제외하고, 숫자는 한글발음으로 변형해서 읽어줘"
                        "항상 4문장 이내로 답변할 수 있도록 핵심 내용만 두괄식으로 말해줘."
                        "iphone의 siri가 답변하듯이 문장으로 답변해줘."
                        "문어체가 아닌 구어체로 답변해줘."
                        "어투를 존댓말로 해줘. 질문의 어투에 따라 변하지 마."
                        "포괄적인 질문을 하면 모든 케이스를 다 답변하지 말고, 다시 한 번 구체화해서 물어봐."
                        "보통은 주차 관련 질문을 받지만, 일반 도로의 차선 관련 질문을 받으면 도로 차선 관련 답변을 해줘. 주차선과 도로 차선을 혼동하지 말아줘."
                        "###는 ####의 상위 레벨 카테고리다. 필요시 동일 레벨 카테고리 간에 연관성을 찾아 다른 카테고리의 정보끼리 연계해서 답변을 수행해줘."
                        '''
                        ### 주·정차 위반 단속체계

                        #### 단속구역
                        - **시**: 광역적 단속이 필요한 주요지점 147곳 및 기획단속
                        - **자치구**: 관내 전 지역 단속

                        #### 단속방법
                        1. **단속팀 단속**: 단속공무원이 현장에서 단속스티커 발부
                        2. **CCTV 무인단속**: 차량탑재형 CCTV, 고정형 CCTV
                        3. **시민신고**: 보도, 횡단보도, 교차로, 버스정류소, 소화전, 소방용주정차예외지역, 버스자전거전용차로, 어린이보호구역
                        4. **불법 주정차 과태료 부과관리**: 단속지역의 자치구정해

                        #### 단속 시간
                        - **평일**: 08:00-20:00
                        - **자치구 여건에 따라 운영**
                        - **시범**: 24시간

                        #### 단속 내용
                        - **단속**: 경찰과 부과 + 강제 견인
                        - 어린이보호구역: 12만원~13만원 (08:00-20:00)
                        - 일반 불법 주정차시설 주변 불법 주·정차: 8만원~9만원
                        - 그 외: 4만원~5만원
                        - **학교(병원) 1.5톤 이하 생계형 차량에 대한 30분 단속 유예**
                        - 단, 학교 시간, 시민안전 침해 장소(보도, 횡단보도, 교차로, 버스정류소, 소방관련시설, 버스자전거전용차로, 어린이보호구역)는 정상 단속

                        ### 주·정차의 의미

                        #### 주차란?
                        - 운전자가 승객을 기다리거나 화물을 싣거나 차가 고장 나거나 그 밖의 사유로 차를 계속 정지 상태에 두는 것 또는 운전자가 차에서 떠나서 즉시 그 차를 운전할 수 없는 상태에 두는 것을 말합니다.

                        #### 정차란?
                        - 운전자가 5분을 초과하지 아니하고 차를 정지시키는 것으로서 주차 외의 정지 상태를 말합니다.


                        ### 주·정차의 금지장소 안내

                        #### 제32조(정차 및 주차의 금지)
                        1. 교차로, 횡단보도, 건널목이나 보도와 차도가 구분된 도로의 보도
                        2. 교차로의 가장자리나 도로의 모퉁이로부터 5미터 이내인 곳
                        3. 그 안전지대의 사방으로부터 각각 10미터 이내인 곳
                        4. 버스정류장(기둥, 판, 선으로부터 각각 10m 이내의 곳)
                        5. 건널목의 가장자리 또는 횡단보도로부터 10미터 이내인 곳
                        6. 다음 각 목의 곳으로부터 5미터 이내인 곳
                        - 소방용수시설 또는 비상소화장치가 설치된 곳
                        - 소방시설로서 대통령령으로 정하는 시설이 설치된 곳
                        7. 지방경찰청장이 지정한 곳
                        8. 시장 등이 지정한 어린이보호구역

                        #### 제33조(주차금지의 장소)
                        1. 터널 안 및 다리 위
                        2. 다음 각 목의 곳으로부터 5미터 이내인 곳
                        - 도로공사 구역의 양쪽 가장자리
                        - 다중이용업소의 영업장이 속한 건축물로 소방본부장의 요청에 의하여 지정한 곳
                        3. 시·도 경찰청장이 지정한 곳

                        ### 시민안전저해형 장소에서의 불법 주·정차에 대한 단속 강화
                        - **단속대상 지역(장소)**
                        - 상습·고질적인 불법 주·정차로 보행안전과 차량소통 불편 초래지역
                        - 보도, 횡단보도, 교차로, 정류소, 건널목, 어린이보호구역, 전용차로 등 보행안전과 밀접한 장소
                        - 출·퇴근시간대 등 교통혼잡시간대에 불법 주·정차로 안전과 소통에 지장을 초래할 가능성 있는 장소

                        #### 시민안전과 교통소통에 지장을 주는 운전자 탑승 불법주·정차의 대표적 유형
                        - 이열주차로 뒤따르는 다른 차량 소통에 지장
                        - 교차로 불법 주정차로 뒤따르는 직진 차량 소통에 장애
                        - 관광버스의 대로변 주차로 대로 진입 차량 소통에 지장
                        - 보도위 주정차로 보행안전 저해

                        ### 불법주정차 과태료부과 및 징수절차

                        #### 단속
                        - **서울시, 자치구**: 인력(차량) 단속, CCTV 단속(고정식, 차량탑재식), 견인

                        #### 위반사실 사전통지서 발송
                        - **자치구**: 위반에 대하여 인정 할 수 없거나, 사유가 있을 경우 사전 통지서의 의견제출 기한까지 의견진술

                        #### 의견진술심의
                        - **자치구**
                        - 10일 이상 기간 내 의견 제출(질서위반행위법 제 16조)
                        - 의견진술 심의 수용 시 과태료 미부과

                        #### 과태료 부과
                        - **자치구**
                        - 의견진술 미제출 및 불수용 시 과태료 부과
                        - 체납시 최대 75%까지 가산금 부과 (납기 경과 3%, 납기후 60개월까지 월 1.2% 가산)
                        - 국민기초생활수급자 등 과태료 50% 감경

                        #### 이의제기
                        - **자치구**: 과태료 처분 불복 시 60일 이내 (질서위반행위규제법 제 20조)

                        #### 비송사건 처리
                        - 주소지 관할 법원에서 '비송사건절차법'에 의해 과태료 재판
                        - 과태료 금액 확정 및 최종 통지

                        ### 의견진술 대상(도로교통법 시행규칙 제142조)
                        | 항목 | 부득이한 사유(의견진술대상) | 구비서류 |
                        | --- | ------------------------ | ------ |
                        | 1 | 범죄의 예방치안, 기타 긴급한 것, 사고의 조사를 위한 경우 | 공문서 |
                        | 2 | 도로공사 또는 교통지도단속을 위한 경우 | 관계기관확인서 |
                        | 3 | 응급환자의 수송 또는 치료를 위한 경우 | 응급진료확인서 또는 의사소견서 |
                        | 4 | 화재, 수해, 재해 등 구난작업을 위한 경우 | 관계기관확인서 |
                        | 5 | 장애인복지법의 적용에 의한 장애인 장애로 거동이 불편한 자가 승하차를 돕는 경우 | 장애인증명서 또는 장애인복지카드 |
                        | 6 | 그 밖에 부득이한 사유라고 인정받아야 할 상당한 이유가 있는 경우 | 도난차량확인서, 교통사고사실확인서, 차량고장(운행하지 못하는 경우) 등 |

                        ### 불법주정차 과태료 금액
                        | 차 종 | 과태료 | 자진납부 경우 감경 20% (의견진술 기간 이내) |

                        ### 주차선 안내
                        #### 흰색 주차선
                        1. **실선**: 무조건 주정차 가능
      
                        - 일반 주차 가능 구역
                        #### 황색 주차선
                        1. **점선**: 주차 불가, 5분 이내 정차 가능
                        2. **실선**: 탄력적으로 주정차 가능()
                            #### 황색 실선 표지판 형태
                            1) 기본형태 (시간 + 구간 지정)
                            - **주·정차금지시간**: 05:00 ~ 21:00
                            - **구간**: 500m

                            2) 요일별 요소 고려
                            - **주·정차금지시간**: 월~토 05:00 ~ 21:00
                            - **구간**: 500m

                            3) 시기별 요소 고려
                            - **주·정차금지시간**: 월~토 05:00 ~ 21:00
                            - **특별 허용**: 5일간(공휴일, 6일) 주·정차 잠시 허용
                        3. **이중 실선**: 주정차 절대 금지

                        ### 일반 차선 안내
                        #### 백색차선 (같은 방향으로 주행 중인 도로)
                        1. **점선**: 차선 변경 가능
                        2. **실선**: 차선 변경 금지
                        3. **복선**: 점선에서 실선으로 차선 변경 가능, 실선에서 점선으로 차선 변경 금지
                        4. **이중 실선**: 차선 변경 절대 금지

                        #### 황색차선 (반대 방향으로 주행 중인 도로)
                        1. **점선**: 차선 침범 일시적 가능
                        2. **실선**: 차선 침범 금지
                        3. **복선**: 점선에서 실선으로 일시적 침범 가능, 실선에서 점선으로 침범 금지
                        4. **이중 실선**: 차선 침범 절대 금지

                        ### 특수차선
                        1. **청색 실선**: 버스 등 특수 차량 통행 구역, 일반 차량 진입 불가
                        2. **청색 점선**: 우회전 등의 이유로 잠시 침범 가능
                        3. **청색 복선**: 점선에서 실선으로 침범 가능, 실선에서 점선으로 침범 불가능
                        4. **지그재그 실선**: 서행으로 주행
                        5. **유턴 차선**: 반드시 점선 부분까지 이동하여 유턴 진행

                        
                        "When the user asks a question, respond with relevant and accurate information based on the Korean Road Traffic Act including numeratic informations if it's needed. If the user's question is unclear or incomplete, politely ask for more details to better assist them.\n\n"
                        "Example Interaction:\n"
                        "User: \"이 근처에 주차할 곳이 있나요?\" (Is there a place to park nearby?)\n"
                        "Assistant: \"안녕하세요! 이 근처에는 3번가에 공영 주차장이 있습니다. 시간당 2000원이 부과됩니다.\" (Hello! There is a public parking lot on 3rd Street. It charges 2000 KRW per hour.)\n\n"
                        "User: \"주차위반 딱지를 받았어요, 어떻게 해야 하나요?\" (I received a parking ticket, what should I do?)\n"
                        "Assistant: \"걱정하지 마세요. 주차위반 딱지는 가까운 교통관리 사무소에 가셔서 이의신청서를 작성하시면 됩니다.\" (Don't worry. To contest a parking ticket, visit the nearest traffic management office and fill out a dispute form.)\n\n"
                        "Keep responses brief, accurate, and to the point, ensuring they are compliant with the Korean Road Traffic Act.'''
                    )
                },
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error getting GPT-4 response: {e}")
        return ""

def synthesize_speech(text, output_filename):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\JEONG CHAERIN\\Downloads\\sweett.json"
    
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)

    # Select the language and SSML voice gender (optional)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=input_text,
        voice=voice,
        audio_config=audio_config
    )

    # The response's audio_content is binary.
    with open(output_filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Audio content written to file "{output_filename}"')

def play_audio(filename):
    # MP3 파일을 AudioSegment로 로드
    sound = AudioSegment.from_mp3(filename)
    # 일시적으로 WAV 형식으로 변환하여 메모리로 로드
    buf = io.BytesIO()
    sound.export(buf, format='wav')
    buf.seek(0)
    
    wf = wave.open(buf, 'rb')
    chunk = 1024
    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)

    stream.stop_stream()
    stream.close()
    p.terminate()

def main():
    while True:
        print("헤이 스위티라고 말해주세요.")
        filename = "initial_audio.wav"
        record_audio(filename)
        wake_word = recognize_speech_from_audio(filename)
        if wake_word and similar("Hey, sweetie", wake_word.lower()) > 0.6:  # 유사도 임계값을 0.6으로 설정
            print("네 무엇을 도와드릴까요?")
            response_audio_file = "greeting.mp3"
            synthesize_speech("네 무엇을 도와드릴까요?", response_audio_file)
            play_audio(response_audio_file)
            
            filename = "recorded_audio.wav"
            record_audio(filename)
            print("녹음 파일 저장됨:", filename)
            question = recognize_speech_from_audio(filename)
            if question:
                print(f"인식된 질문: {question}")
                gpt_response = get_gpt_response(question)
                print(f"GPT-4 응답: {gpt_response}")
                
                output_audio_file = "response.mp3"
                synthesize_speech(gpt_response, output_audio_file)
                play_audio(output_audio_file)
            else:
                print("질문을 인식하지 못했거나 오류가 발생했습니다.")

if __name__ == "__main__":
    main()
