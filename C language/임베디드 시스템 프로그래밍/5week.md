### 5주차: Arduino 액추에이터 제어

**강의 목표:**  
Arduino를 이용하여 서보 모터 및 DC 모터와 같은 액추에이터를 제어하는 방법을 학습하고, 다양한 액추에이터를 사용한 프로젝트를 구현합니다.

**강의 내용:**

1. **서보 모터 제어**
   - 서보 모터의 개요
   - 서보 모터의 작동 원리
   - 서보 모터 제어를 위한 라이브러리 (Servo.h)
   - 서보 모터의 각도 제어

2. **DC 모터 제어**
   - DC 모터의 개요
   - H-브리지와 모터 드라이버의 사용
   - DC 모터의 속도 및 방향 제어

3. **액추에이터 제어 프로젝트**
   - 서보 모터와 DC 모터를 이용한 간단한 로봇 팔 제어
   - PWM을 이용한 모터 속도 제어

**실습 내용:**

1. **서보 모터 제어 실습**
   - 서보 모터를 연결하고, 각도를 제어하는 프로그램 작성
   - 서보 모터를 사용하여 간단한 움직임 구현

2. **DC 모터 제어 실습**
   - 모터 드라이버를 이용하여 DC 모터를 연결
   - DC 모터의 속도 및 방향을 제어하는 프로그램 작성

3. **액추에이터 제어 프로젝트**
   - 서보 모터와 DC 모터를 이용한 간단한 로봇 팔 제어 프로젝트 구현

---

**강의 진행:**

1. **강의 시작 (10분)**
   - 강의 목표와 주제를 소개합니다.
   - 서보 모터와 DC 모터의 개요를 설명합니다.

2. **서보 모터 제어 (20분)**
   - 서보 모터의 작동 원리와 제어 방법을 설명합니다.
   - Servo.h 라이브러리를 사용하여 서보 모터를 제어하는 방법을 시연합니다.
   - 서보 모터의 각도 제어 예제를 설명합니다.

3. **DC 모터 제어 (20분)**
   - DC 모터의 작동 원리와 제어 방법을 설명합니다.
   - H-브리지와 모터 드라이버의 사용법을 설명합니다.
   - DC 모터의 속도 및 방향을 제어하는 예제를 설명합니다.

4. **액추에이터 제어 프로젝트 (20분)**
   - 서보 모터와 DC 모터를 이용한 간단한 로봇 팔 제어 프로젝트를 설명합니다.
   - 프로젝트 구현을 위한 기본적인 설계와 프로그램 구조를 설명합니다.

5. **실습 준비 및 Q&A (10분)**
   - Q&A 시간을 통해 학생들의 질문에 답변하고, 이해도를 확인합니다.
   - 다음 주 실습을 위한 준비 사항을 안내합니다.

**실습 내용:**

1. **서보 모터 제어 실습**

**서보 모터 제어 예제 코드:**

```cpp
#include <Servo.h>

// 서보 모터 객체 생성
Servo myservo;

// 서보 모터 핀 정의
const int servoPin = 9;

void setup() {
  // 서보 모터 초기화
  myservo.attach(servoPin);
}

void loop() {
  // 서보 모터를 0도에서 180도까지 회전
  for (int pos = 0; pos <= 180; pos++) {
    myservo.write(pos);
    delay(15); // 15ms 대기
  }

  // 서보 모터를 180도에서 0도까지 회전
  for (int pos = 180; pos >= 0; pos--) {
    myservo.write(pos);
    delay(15); // 15ms 대기
  }
}
```

2. **DC 모터 제어 실습**

**DC 모터 제어 예제 코드:**

```cpp
// 모터 드라이버 핀 정의
const int motorPin1 = 5; // 모터 드라이버 입력1
const int motorPin2 = 6; // 모터 드라이버 입력2
const int enablePin = 9; // 모터 드라이버 PWM 입력

void setup() {
  // 모터 핀을 출력으로 설정
  pinMode(motorPin1, OUTPUT);
  pinMode(motorPin2, OUTPUT);
  pinMode(enablePin, OUTPUT);
}

void loop() {
  // 모터를 앞으로 회전
  digitalWrite(motorPin1, HIGH);
  digitalWrite(motorPin2, LOW);
  analogWrite(enablePin, 255); // 최대 속도로 회전
  delay(2000); // 2초 대기

  // 모터를 반대로 회전
  digitalWrite(motorPin1, LOW);
  digitalWrite(motorPin2, HIGH);
  analogWrite(enablePin, 255); // 최대 속도로 회전
  delay(2000); // 2초 대기

  // 모터를 멈춤
  digitalWrite(motorPin1, LOW);
  digitalWrite(motorPin2, LOW);
  analogWrite(enablePin, 0); // 속도 0으로 설정
  delay(2000); // 2초 대기
}
```

3. **액추에이터 제어 프로젝트**

**로봇 팔 제어 예제 코드:**

```cpp
#include <Servo.h>

// 서보 모터 객체 생성
Servo baseServo;
Servo armServo;

// 서보 모터 핀 정의
const int baseServoPin = 9;
const int armServoPin = 10;

void setup() {
  // 서보 모터 초기화
  baseServo.attach(baseServoPin);
  armServo.attach(armServoPin);
}

void loop() {
  // 로봇 팔을 기본 위치로 이동
  baseServo.write(90); // 90도로 이동
  armServo.write(90);  // 90도로 이동
  delay(1000); // 1초 대기

  // 로봇 팔을 다양한 위치로 이동
  for (int pos = 0; pos <= 180; pos += 10) {
    baseServo.write(pos);
    armServo.write(180 - pos);
    delay(500); // 0.5초 대기
  }
}
```

**과제:**

1. **서보 모터를 이용한 물체 이동**
   - 서보 모터를 이용하여 작은 물체를 이동시키는 프로그램을 작성합니다.
   - 작성한 프로그램을 업로드하고, 실행 결과를 동영상으로 촬영하여 제출합니다.

2. **DC 모터를 이용한 팬 속도 제어**
   - DC 모터를 이용하여 팬 속도를 제어하는 프로그램을 작성합니다.
   - PWM을 이용하여 팬의 속도를 3단계로 조절합니다.
   - 작성한 프로그램을 업로드하고, 실행 결과를 동영상으로 촬영하여 제출합니다.

이 강의 계획을 통해 학생들은 Arduino를 이용하여 서보 모터 및 DC 모터와 같은 액추에이터를 제어하는 방법을 학습하고, 다양한 프로젝트를 통해 실제 응용 능력을 키울 수 있습니다.