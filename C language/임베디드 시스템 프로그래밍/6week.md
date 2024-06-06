### 6주차: STM32 기초 프로그래밍

**강의 목표:**  
STM32 마이크로컨트롤러 보드의 기본 구성과 GPIO 제어 방법을 학습하고, 간단한 LED 점멸 프로그램을 작성하고 실행합니다.

**강의 내용:**

1. **STM32 보드 구성 요소**
   - STM32 보드 개요
   - 주요 구성 요소 설명 (마이크로컨트롤러, 디지털/아날로그 핀, 전원 핀, 리셋 버튼 등)

2. **STM32CubeIDE 사용법**
   - STM32CubeIDE 설치 및 설정
   - 프로젝트 생성 및 보드 설정
   - 기본적인 사용법 (코드 작성, 컴파일, 업로드)

3. **GPIO 제어**
   - GPIO 개요
   - GPIO 핀 모드 설정 (INPUT, OUTPUT)
   - GPIO 핀 읽기/쓰기 (HAL_GPIO_ReadPin, HAL_GPIO_WritePin)

4. **간단한 LED 점멸 프로그램**
   - LED 점멸 프로그램 작성 방법
   - HAL 라이브러리 사용법
   - LED 점멸 프로그램 업로드 및 실행

**실습 내용:**

1. **STM32 보드 구성 요소 이해**
   - STM32 보드의 주요 구성 요소를 관찰하고 이해합니다.

2. **STM32CubeIDE 사용법 익히기**
   - STM32CubeIDE를 열고, 새로운 프로젝트를 생성하여 기본적인 코드를 작성합니다.
   - 보드 설정, 컴파일 및 업로드 과정을 실습합니다.

3. **GPIO 제어 실습**
   - GPIO 핀을 사용하여 LED를 켜고 끄는 프로그램을 작성합니다.
   - 버튼 입력을 읽어 LED를 제어하는 프로그램을 작성합니다.

---

**강의 진행:**

1. **강의 시작 (10분)**
   - 강의 목표와 주제를 소개합니다.
   - STM32 보드의 기본 구성 요소를 설명합니다.

2. **STM32 보드 구성 요소 (20분)**
   - STM32 보드의 주요 구성 요소를 설명합니다.
   - 마이크로컨트롤러, 디지털/아날로그 핀, 전원 핀, 리셋 버튼 등의 역할과 기능을 설명합니다.

3. **STM32CubeIDE 사용법 (20분)**
   - STM32CubeIDE의 기본 사용법을 설명합니다.
   - 새로운 프로젝트 생성, 보드 설정, 코드 작성, 컴파일 및 업로드 과정을 시연합니다.

4. **GPIO 제어 (20분)**
   - GPIO 핀 모드 설정 (INPUT, OUTPUT)과 GPIO 핀 읽기/쓰기(HAL_GPIO_ReadPin, HAL_GPIO_WritePin) 방법을 설명합니다.
   - 간단한 LED 점멸 프로그램 작성 방법을 설명합니다.

5. **LED 점멸 프로그램 작성 및 실행 (20분)**
   - 학생들과 함께 간단한 LED 점멸 프로그램을 작성합니다.
   - HAL 라이브러리를 사용하여 LED를 제어하는 방법을 설명합니다.
   - 프로그램을 업로드하고 실행하여 결과를 확인합니다.

6. **Q&A 및 실습 준비 (10분)**
   - Q&A 시간을 통해 학생들의 질문에 답변하고, 이해도를 확인합니다.
   - 다음 주 실습을 위한 준비 사항을 안내합니다.

**실습 내용:**

1. **STM32 보드 구성 요소 이해**
   - 각자 STM32 보드를 관찰하고, 구성 요소의 역할과 기능을 이해합니다.

2. **STM32CubeIDE 사용법 익히기**
   - 새로운 프로젝트를 생성하고, 간단한 코드를 작성하여 컴파일 및 업로드 과정을 실습합니다.

3. **GPIO 제어 실습**

**LED 점멸 프로그램 예제 코드:**

**main.c 파일:**

```c
#include "main.h"

// 함수 선언
void SystemClock_Config(void);
static void MX_GPIO_Init(void);

int main(void) {
  // HAL 라이브러리 초기화
  HAL_Init();

  // 시스템 클록 설정
  SystemClock_Config();

  // GPIO 초기화
  MX_GPIO_Init();

  // 메인 루프
  while (1) {
    // LED 켜기
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET);
    HAL_Delay(1000); // 1초 대기

    // LED 끄기
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET);
    HAL_Delay(1000); // 1초 대기
  }
}

// 시스템 클록 설정 함수 (자동 생성됨)
void SystemClock_Config(void) {
  // 클록 설정 코드
}

// GPIO 초기화 함수 (자동 생성됨)
static void MX_GPIO_Init(void) {
  // GPIO 설정 구조체
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  // GPIO 클록 활성화
  __HAL_RCC_GPIOA_CLK_ENABLE();

  // GPIO 핀 설정 (PA5: LED 핀)
  GPIO_InitStruct.Pin = GPIO_PIN_5;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
}
```

**button_led.c 파일:**

```c
#include "main.h"

// 함수 선언
void SystemClock_Config(void);
static void MX_GPIO_Init(void);

int main(void) {
  // HAL 라이브러리 초기화
  HAL_Init();

  // 시스템 클록 설정
  SystemClock_Config();

  // GPIO 초기화
  MX_GPIO_Init();

  // 메인 루프
  while (1) {
    // 버튼 상태 읽기
    GPIO_PinState buttonState = HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_13);

    // 버튼이 눌리면 LED 켜기, 아니면 끄기
    if (buttonState == GPIO_PIN_RESET) { // 버튼이 눌리면 (풀업 저항 사용)
      HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET);
    } else {
      HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET);
    }

    HAL_Delay(100); // 100ms 대기
  }
}

// 시스템 클록 설정 함수 (자동 생성됨)
void SystemClock_Config(void) {
  // 클록 설정 코드
}

// GPIO 초기화 함수 (자동 생성됨)
static void MX_GPIO_Init(void) {
  // GPIO 설정 구조체
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  // GPIO 클록 활성화
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();

  // GPIO 핀 설정 (PA5: LED 핀)
  GPIO_InitStruct.Pin = GPIO_PIN_5;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  // GPIO 핀 설정 (PC13: 버튼 핀)
  GPIO_InitStruct.Pin = GPIO_PIN_13;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP; // 풀업 저항 사용
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
}
```

**과제:**

1. **LED 점멸 프로그램 변형**
   - LED 점멸 프로그램을 변형하여 LED가 더 빠르게 깜빡이도록 변경합니다.
   - LED 점멸 주기를 500ms로 설정하고, 프로그램을 작성하여 실행 결과를 확인합니다.

2. **버튼 입력을 이용한 LED 제어 프로그램 작성**
   - 버튼 입력을 이용하여 LED를 제어하는 프로그램을 작성합니다.
   - 버튼을 누르고 있을 때 LED가 켜지고, 버튼을 누르지 않으면 LED가 꺼지도록 설정합니다.
   - 작성한 프로그램을 업로드하고, 실행 결과를 동영상으로 촬영하여 제출합니다.

이 강의 계획을 통해 학생들은 STM32 마이크로컨트롤러의 기본 구성과 GPIO 제어 방법을 익히고, 간단한 프로그램을 작성하고 실행해봄으로써 임베디드 시스템 프로그래밍에 대한 기초를 다질 수 있습니다.