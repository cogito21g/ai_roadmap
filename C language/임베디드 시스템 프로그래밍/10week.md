### 10주차: Raspberry Pi 센서 인터페이스

**강의 목표:**  
Raspberry Pi를 이용하여 I2C, SPI, UART 통신을 설정하고 사용하는 방법을 학습하며, 다양한 센서를 연결하여 데이터를 읽고 처리합니다.

**강의 내용:**

1. **I2C 통신**
   - I2C 통신의 기본 개념
   - I2C의 구조와 동작 원리
   - Raspberry Pi에서 I2C 설정 및 사용 방법
   - I2C 센서 연결 및 데이터 읽기

2. **SPI 통신**
   - SPI 통신의 기본 개념
   - SPI의 구조와 동작 원리
   - Raspberry Pi에서 SPI 설정 및 사용 방법
   - SPI 디바이스 연결 및 데이터 읽기

3. **UART 통신**
   - UART 통신의 기본 개념
   - UART의 구조와 동작 원리
   - Raspberry Pi에서 UART 설정 및 사용 방법
   - UART를 이용한 시리얼 통신

**실습 내용:**

1. **I2C 센서 데이터 읽기**
   - I2C 통신을 이용하여 센서 데이터를 읽고 처리하는 프로그램 작성

2. **SPI 디바이스 제어**
   - SPI 통신을 이용하여 디바이스를 제어하는 프로그램 작성

3. **UART 통신을 이용한 데이터 전송**
   - UART 통신을 이용하여 시리얼 데이터를 송수신하는 프로그램 작성

---

**강의 진행:**

1. **강의 시작 (10분)**
   - 강의 목표와 주제를 소개합니다.
   - I2C, SPI, UART 통신 프로토콜의 기본 개념을 설명합니다.

2. **I2C 통신 (20분)**
   - I2C 통신의 구조와 동작 원리를 설명합니다.
   - Raspberry Pi에서 I2C 설정 방법을 시연합니다.
   - I2C 센서를 연결하고 데이터를 읽는 방법을 설명합니다.

3. **SPI 통신 (20분)**
   - SPI 통신의 구조와 동작 원리를 설명합니다.
   - Raspberry Pi에서 SPI 설정 방법을 시연합니다.
   - SPI 디바이스를 연결하고 데이터를 읽는 방법을 설명합니다.

4. **UART 통신 (20분)**
   - UART 통신의 구조와 동작 원리를 설명합니다.
   - Raspberry Pi에서 UART 설정 방법을 시연합니다.
   - UART를 이용하여 시리얼 데이터를 송수신하는 방법을 설명합니다.

5. **실습 준비 및 Q&A (10분)**
   - Q&A 시간을 통해 학생들의 질문에 답변하고, 이해도를 확인합니다.
   - 다음 주 실습을 위한 준비 사항을 안내합니다.

**실습 내용:**

1. **I2C 센서 데이터 읽기**

**I2C 센서 데이터 읽기 예제 코드 (Python):**

```python
import smbus2
import time

# I2C 버스 초기화
bus = smbus2.SMBus(1)
i2c_address = 0x48

try:
    while True:
        # I2C 센서 데이터 읽기
        data = bus.read_i2c_block_data(i2c_address, 0x00, 2)
        temp = (data[0] << 8 | data[1]) >> 4
        temperature = temp * 0.0625

        # 시리얼 모니터에 출력
        print(f"Temperature: {temperature:.2f} C")
        time.sleep(1)  # 1초 대기
except KeyboardInterrupt:
    pass
```

2. **SPI 디바이스 제어**

**SPI 디바이스 제어 예제 코드 (Python):**

```python
import spidev
import time

# SPI 버스 초기화
spi = spidev.SpiDev()
spi.open(0, 0)  # bus: 0, device: 0
spi.max_speed_hz = 50000

try:
    while True:
        # SPI 디바이스에 데이터 전송
        resp = spi.xfer2([0x01, 0x02])
        print(f"Data sent via SPI: {resp}")
        time.sleep(1)  # 1초 대기
except KeyboardInterrupt:
    pass
finally:
    spi.close()
```

3. **UART 통신을 이용한 데이터 전송**

**UART 통신 예제 코드 (Python):**

```python
import serial
import time

# UART 초기화
ser = serial.Serial('/dev/ttyS0', 115200)

try:
    while True:
        # UART를 이용한 데이터 전송
        ser.write(b"Hello, UART!\r\n")
        time.sleep(1)  # 1초 대기
except KeyboardInterrupt:
    pass
finally:
    ser.close()
```

**과제:**

1. **I2C 센서 데이터를 읽어 시리얼 모니터에 출력하는 프로그램 작성**
   - I2C 통신을 이용하여 센서 데이터를 읽고 시리얼 모니터에 출력하는 프로그램을 작성합니다.
   - 작성한 프로그램을 업로드하고, 실행 결과를 동영상으로 촬영하여 제출합니다.

2. **SPI 디바이스를 제어하는 프로그램 작성**
   - SPI 통신을 이용하여 디바이스를 제어하는 프로그램을 작성합니다.
   - 작성한 프로그램을 업로드하고, 실행 결과를 동영상으로 촬영하여 제출합니다.

3. **UART 통신을 이용한 데이터 송수신 프로그램 작성**
   - UART 통신을 이용하여 데이터를 송수신하는 프로그램을 작성합니다.
   - 작성한 프로그램을 업로드하고, 실행 결과를 동영상으로 촬영하여 제출합니다.

이 강의 계획을 통해 학생들은 Raspberry Pi에서 I2C, SPI, UART 통신 프로토콜을 설정하고 사용하는 방법을 학습하며, 다양한 센서를 연결하여 데이터를 읽고 처리하는 기술을 습득할 수 있습니다.