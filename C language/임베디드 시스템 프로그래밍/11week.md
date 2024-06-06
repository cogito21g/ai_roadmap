### 11주차: Raspberry Pi 네트워크 설정 및 간단한 웹 서버 구축

**강의 목표:**  
Raspberry Pi의 네트워크 설정을 이해하고, 간단한 웹 서버를 구축하여 센서 데이터를 웹 페이지로 출력하는 방법을 학습합니다.

**강의 내용:**

1. **Raspberry Pi 네트워크 설정**
   - 유선 및 무선 네트워크 설정
   - 고정 IP 주소 설정
   - 네트워크 연결 확인

2. **웹 서버 개요**
   - 웹 서버의 기본 개념과 동작 원리
   - HTTP 프로토콜 개요
   - 웹 서버 소프트웨어 소개 (Apache, Nginx, Flask 등)

3. **간단한 웹 서버 구축**
   - Flask를 이용한 웹 서버 구축
   - Flask 설치 및 기본 설정
   - 간단한 웹 페이지 작성

4. **센서 데이터를 웹 페이지로 출력**
   - 센서 데이터 읽기
   - 웹 페이지에서 센서 데이터 표시

**실습 내용:**

1. **네트워크 설정**
   - 유선 및 무선 네트워크 설정
   - 고정 IP 주소 설정
   - 네트워크 연결 상태 확인

2. **Flask 웹 서버 구축**
   - Flask 설치 및 기본 설정
   - 간단한 웹 페이지 작성 및 서버 실행

3. **센서 데이터 웹 페이지 출력**
   - 센서 데이터 읽기 코드 작성
   - 웹 페이지에서 센서 데이터 표시

---

**강의 진행:**

1. **강의 시작 (10분)**
   - 강의 목표와 주제를 소개합니다.
   - Raspberry Pi의 네트워크 설정의 중요성을 설명합니다.

2. **네트워크 설정 (20분)**
   - 유선 및 무선 네트워크 설정 방법을 설명합니다.
   - 고정 IP 주소 설정 방법을 시연합니다.
   - 네트워크 연결 상태를 확인하는 방법을 설명합니다.

3. **웹 서버 개요 (20분)**
   - 웹 서버의 기본 개념과 동작 원리를 설명합니다.
   - HTTP 프로토콜 개요와 웹 서버 소프트웨어를 소개합니다.

4. **Flask 웹 서버 구축 (20분)**
   - Flask 설치 및 기본 설정 방법을 설명합니다.
   - 간단한 웹 페이지 작성 및 서버 실행 방법을 시연합니다.

5. **센서 데이터를 웹 페이지로 출력 (20분)**
   - 센서 데이터를 읽는 방법을 설명합니다.
   - 웹 페이지에서 센서 데이터를 표시하는 방법을 시연합니다.

6. **Q&A 및 실습 준비 (10분)**
   - Q&A 시간을 통해 학생들의 질문에 답변하고, 이해도를 확인합니다.
   - 다음 주 실습을 위한 준비 사항을 안내합니다.

**실습 내용:**

1. **네트워크 설정**

**고정 IP 주소 설정 예제 (Raspberry Pi OS):**

```sh
sudo nano /etc/dhcpcd.conf

# 파일 끝에 다음 내용 추가
interface eth0
static ip_address=192.168.1.100/24
static routers=192.168.1.1
static domain_name_servers=192.168.1.1
```

2. **Flask 웹 서버 구축**

**Flask 설치 및 기본 설정 (Python):**

```sh
sudo apt update
sudo apt install python3-flask
```

**간단한 웹 페이지 작성 (app.py):**

```python
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**웹 페이지 템플릿 (templates/index.html):**

```html
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sensor Data</title>
</head>
<body>
    <h1>Sensor Data</h1>
    <p>Temperature: {{ temperature }} °C</p>
</body>
</html>
```

3. **센서 데이터 웹 페이지 출력**

**센서 데이터 읽기 및 웹 페이지 출력 (app.py 수정):**

```python
from flask import Flask, render_template
import smbus2
import time

app = Flask(__name__)

# I2C 버스 초기화
bus = smbus2.SMBus(1)
i2c_address = 0x48

@app.route('/')
def index():
    # I2C 센서 데이터 읽기
    data = bus.read_i2c_block_data(i2c_address, 0x00, 2)
    temp = (data[0] << 8 | data[1]) >> 4
    temperature = temp * 0.0625

    return render_template('index.html', temperature=temperature)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**과제:**

1. **네트워크 설정 및 고정 IP 주소 설정**
   - Raspberry Pi의 유선 및 무선 네트워크를 설정하고, 고정 IP 주소를 설정합니다.
   - 네트워크 설정 과정을 사진이나 동영상으로 촬영하여 제출합니다.

2. **Flask 웹 서버 구축 및 실행**
   - Flask를 이용하여 간단한 웹 서버를 구축하고 실행합니다.
   - 웹 페이지에 "Hello, World!"를 출력하는 간단한 프로그램을 작성합니다.
   - 작성한 프로그램을 업로드하고, 실행 결과를 동영상으로 촬영하여 제출합니다.

3. **센서 데이터를 웹 페이지로 출력**
   - 센서 데이터를 읽어 웹 페이지에 출력하는 프로그램을 작성합니다.
   - 작성한 프로그램을 업로드하고, 실행 결과를 동영상으로 촬영하여 제출합니다.

이 강의 계획을 통해 학생들은 Raspberry Pi의 네트워크 설정 방법을 익히고, 간단한 웹 서버를 구축하여 센서 데이터를 웹 페이지로 출력하는 기술을 습득할 수 있습니다.