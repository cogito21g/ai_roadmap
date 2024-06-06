### 4주차: Arduino 센서 인터페이스

**강의 목표:**  
Arduino를 이용하여 아날로그 입력을 읽고, PWM을 제어하는 방법을 학습하며, 온도 센서와 같은 다양한 센서를 연결하여 데이터를 읽고 처리합니다.

**강의 내용:**

1. **아날로그 입력 읽기**
   - 아날로그 입력 핀 개요
   - 아날로그 입력 읽기 (analogRead 함수)
   - 센서 값 변환 및 사용

2. **PWM 제어**
   - PWM 개요
   - PWM 출력 제어 (analogWrite 함수)
   - PWM을 이용한 밝기 조절 및 모터 속도 제어

3. **센서 인터페이스**
   - 다양한 센서 소개 (온도 센서, 광 센서, 초음파 센서 등)
   - 센서 연결 방법
   - 센서 데이터를 읽고 처리하는 방법

**실습 내용:**

1. **아날로그 입력 읽기**
   - 포텐셔미터를 사용하여 아날로그 입력을 읽고, 그 값을 시리얼 모니터에 출력

2. **PWM 제어**
   - PWM을 이용하여 LED 밝기를 조절하는 프로그램 작성
   - 서보 모터를 제어하는 프로그램 작성

3. **온도 센서 사용**
   - 온도 센서를 연결하고, 온도 값을 읽어 시리얼 모니터에 출력
   - 특정 온도 이상일 때 LED를 켜는 프로그램 작성

---

**강의 진행:**

1. **강의 시작 (10분)**
   - 강의 목표와 주제를 소개합니다.
   - 아날로그 입력과 PWM의 기본 개념을 설명합니다.

2. **아날로그 입력 읽기 (20분)**
   - 아날로그 입력 핀의 역할과 사용 방법을 설명합니다.
   - analogRead 함수를 사용하여 아날로그 값을 읽고, 시리얼 모니터에 출력하는 방법을 시연합니다.

3. **PWM 제어 (20분)**
   - PWM의 개념과 작동 원리를 설명합니다.
   - analogWrite 함수를 사용하여 PWM 신호를 생성하고, 이를 이용해 LED 밝기와 모터 속도를 제어하는 방법을 시연합니다.

4. **센서 인터페이스 (30분)**
   - 다양한 센서를 소개하고, 각 센서의 연결 방법을 설명합니다.
   - 온도 센서를 Arduino에 연결하고, 온도 값을 읽어 시리얼 모니터에 출력하는 방법을 시연합니다.
   - 특정 온도 이상일 때 LED를 켜는 프로그램을 작성하는 방법을 설명합니다.

5. **실습 준비 및 Q&A (10분)**
   - Q&A 시간을 통해 학생들의 질문에 답변하고, 이해도를 확인합니다.
   - 다음 주 실습을 위한 준비 사항을 안내합니다.

**실습 내용:**

1. **아날로그 입력 읽기**
   - 포텐셔미터를 사용하여 아날로그 입력을 읽고, 그 값을 시리얼 모니터에 출력

**아날로그 입력 읽기 예제 코드:**

```cpp
// 아날로그 핀 정의
const int analogPin = A0;

void setup() {
  // 시리얼 통신 시작
  Serial.begin(9600);
}

void loop() {
  // 아날로그 값 읽기
  int sensorValue = analogRead(analogPin);

  // 시리얼 모니터에 출력
  Serial.print("Analog Value: ");
  Serial.println(sensorValue);

  delay(500); // 0.5초 대기
}
```

2. **PWM 제어**
   - PWM을 이용하여 LED 밝기를 조절하는 프로그램 작성

**PWM 제어 예제 코드:**

```cpp
// PWM 핀 정의
const int pwmPin = 9;

void setup() {
  // PWM 핀을 출력으로 설정
  pinMode(pwmPin, OUTPUT);
}

void loop() {
  // LED 밝기 조절 (0~255)
  for (int brightness = 0; brightness <= 255; brightness++) {
    analogWrite(pwmPin, brightness);
    delay(10);
  }

  for (int brightness = 255; brightness >= 0; brightness--) {
    analogWrite(pwmPin, brightness);
    delay(10);
  }
}
```

3. **온도 센서 사용**
   - 온도 센서를 연결하고, 온도 값을 읽어 시리얼 모니터에 출력

**온도 센서 사용 예제 코드:**

```cpp
// 아날로그 핀 정의
const int tempSensorPin = A0;

void setup() {
  // 시리얼 통신 시작
  Serial.begin(9600);
}

void loop() {
  // 아날로그 값 읽기
  int sensorValue = analogRead(tempSensorPin);

  // 온도 계산 (예시: TMP36 온도 센서 사용)
  float voltage = sensorValue * (5.0 / 1023.0);
  float temperatureC = (voltage - 0.5) * 100.0;

  // 시리얼 모니터에 출력
  Serial.print("Temperature: ");
  Serial.print(temperatureC);
  Serial.println(" C");

  delay(1000); // 1초 대기
}
```

4. **온도에 따른 LED 제어 예제 코드:**

```cpp
// 핀 정의
const int tempSensorPin = A0;
const int ledPin = 13;

void setup() {
  // 시리얼 통신 시작
  Serial.begin(9600);
  // LED 핀을 출력으로 설정
  pinMode(ledPin, OUTPUT);
}

void loop() {
  // 아날로그 값 읽기
  int sensorValue = analogRead(tempSensorPin);

  // 온도 계산 (예시: TMP36 온도 센서 사용)
  float voltage = sensorValue * (5.0 / 1023.0);
  float temperatureC = (voltage - 0.5) * 100.0;

  // 시리얼 모니터에 출력
  Serial.print("Temperature: ");
  Serial.print(temperatureC);
  Serial.println(" C");

  // 온도에 따라 LED 제어
  if (temperatureC > 25.0) {
    digitalWrite(ledPin, HIGH); // LED 켜기
  } else {
    digitalWrite(ledPin, LOW);  // LED 끄기
  }

  delay(1000); // 1초 대기
}
```

**과제:**

1. **온도 센서를 이용한 알람 시스템 제작**
   - 특정 온도 이상일 때 경고음을 발생시키는 프로그램을 작성합니다.
   - 작성한 프로그램을 업로드하고, 실행 결과를 동영상으로 촬영하여 제출합니다.

2. **다양한 센서 인터페이스 조사**
   - 다양한 센서(I2C, SPI)를 조사하고, 연결 방법과 사용 예제를 작성하여 제출합니다.
   - 각 센서의 특징과 사용 사례를 포함하여 작성합니다.

이 강의 계획을 통해 학생들은 Arduino를 이용하여 아날로그 입력을 읽고, PWM을 제어하는 방법을 익히며, 다양한 센서를 연결하여 데이터를 읽고 처리하는 기술을 습득할 수 있습니다.