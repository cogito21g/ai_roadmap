### 9주차: Raspberry Pi 개요

**강의 목표:**  
Raspberry Pi 보드의 기본 구성과 설정 방법을 학습하고, GPIO를 사용하여 간단한 LED 점멸 프로그램을 작성하고 실행합니다.

**강의 내용:**

1. **Raspberry Pi 개요**
   - Raspberry Pi의 역사와 목적
   - 다양한 Raspberry Pi 모델과 특징
   - Raspberry Pi의 주요 구성 요소

2. **Raspberry Pi 설정**
   - Raspberry Pi OS 설치 및 초기 설정
   - 네트워크 설정 (Wi-Fi, Ethernet)
   - 원격 접속 설정 (SSH, VNC)

3. **GPIO 제어**
   - GPIO 핀의 개요와 역할
   - GPIO 핀 번호 체계
   - GPIO 핀 모드 설정 (INPUT, OUTPUT)
   - GPIO 핀 읽기/쓰기 (Python RPi.GPIO 라이브러리)

4. **간단한 LED 점멸 프로그램**
   - LED 점멸 프로그램 작성 방법
   - RPi.GPIO 라이브러리를 사용한 GPIO 제어
   - LED 점멸 프로그램 실행

**실습 내용:**

1. **Raspberry Pi 설정**
   - Raspberry Pi OS 설치 및 초기 설정
   - 네트워크 설정 (Wi-Fi, Ethernet)
   - SSH 및 VNC 설정

2. **GPIO 제어 실습**
   - GPIO 핀을 사용하여 LED를 켜고 끄는 프로그램 작성
   - 버튼 입력을 읽어 LED를 제어하는 프로그램 작성

---

**강의 진행:**

1. **강의 시작 (10분)**
   - 강의 목표와 주제를 소개합니다.
   - Raspberry Pi의 기본 개념과 역사에 대해 설명합니다.

2. **Raspberry Pi 개요 (20분)**
   - 다양한 Raspberry Pi 모델과 특징을 설명합니다.
   - Raspberry Pi의 주요 구성 요소를 설명합니다.

3. **Raspberry Pi 설정 (20분)**
   - Raspberry Pi OS 설치 방법을 시연합니다.
   - 초기 설정과 네트워크 설정 방법을 설명합니다.
   - SSH 및 VNC를 통한 원격 접속 설정 방법을 시연합니다.

4. **GPIO 제어 (20분)**
   - GPIO 핀의 개요와 역할을 설명합니다.
   - GPIO 핀 번호 체계와 모드 설정 방법을 설명합니다.
   - RPi.GPIO 라이브러리를 사용하여 GPIO 핀을 제어하는 방법을 시연합니다.

5. **LED 점멸 프로그램 작성 및 실행 (20분)**
   - 학생들과 함께 간단한 LED 점멸 프로그램을 작성합니다.
   - 프로그램을 실행하여 결과를 확인합니다.

6. **Q&A 및 실습 준비 (10분)**
   - Q&A 시간을 통해 학생들의 질문에 답변하고, 이해도를 확인합니다.
   - 다음 주 실습을 위한 준비 사항을 안내합니다.

**실습 내용:**

1. **Raspberry Pi 설정**

**Raspberry Pi OS 설치 및 초기 설정:**

- Raspberry Pi OS 다운로드 및 SD 카드 작성
- Raspberry Pi 첫 부팅 및 초기 설정 (언어, 시간대, 네트워크 등)
- SSH 및 VNC 설정

2. **GPIO 제어 실습**

**LED 점멸 프로그램 예제 코드 (Python):**

```python
import RPi.GPIO as GPIO
import time

# 핀 번호 설정
LED_PIN = 18

# GPIO 설정
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)

try:
    while True:
        GPIO.output(LED_PIN, GPIO.HIGH)  # LED 켜기
        time.sleep(1)                    # 1초 대기
        GPIO.output(LED_PIN, GPIO.LOW)   # LED 끄기
        time.sleep(1)                    # 1초 대기
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()                       # GPIO 정리
```

**버튼 입력을 이용한 LED 제어 예제 코드 (Python):**

```python
import RPi.GPIO as GPIO
import time

# 핀 번호 설정
LED_PIN = 18
BUTTON_PIN = 23

# GPIO 설정
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

try:
    while True:
        button_state = GPIO.input(BUTTON_PIN)
        if button_state == GPIO.LOW:  # 버튼이 눌리면
            GPIO.output(LED_PIN, GPIO.HIGH)  # LED 켜기
        else:
            GPIO.output(LED_PIN, GPIO.LOW)   # LED 끄기
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()                       # GPIO 정리
```

**과제:**

1. **Raspberry Pi 설정 및 네트워크 연결**
   - Raspberry Pi OS를 설치하고 초기 설정을 완료합니다.
   - 네트워크 연결을 설정하고 SSH 및 VNC를 통해 원격 접속이 가능하도록 설정합니다.
   - 설정 과정을 사진이나 동영상으로 촬영하여 제출합니다.

2. **GPIO 제어 프로그램 작성**
   - GPIO 핀을 사용하여 LED를 제어하는 프로그램을 작성합니다.
   - 버튼 입력을 읽어 LED를 제어하는 프로그램을 작성합니다.
   - 작성한 프로그램을 업로드하고, 실행 결과를 동영상으로 촬영하여 제출합니다.

이 강의 계획을 통해 학생들은 Raspberry Pi 보드의 기본 구성과 설정 방법을 익히고, GPIO를 사용하여 간단한 프로그램을 작성하고 실행해봄으로써 Raspberry Pi를 활용한 임베디드 시스템 프로그래밍에 대한 기초를 다질 수 있습니다.