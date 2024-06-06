### 12주차: 리얼타임 운영체제 (RTOS) 개요

**강의 목표:**  
리얼타임 운영체제(RTOS)의 개념과 필요성을 이해하고, 주요 RTOS(예: FreeRTOS, μC/OS-II)의 기본 구조와 기능을 학습합니다. 간단한 태스크 생성 및 스케줄링을 실습합니다.

**강의 내용:**

1. **RTOS 개요**
   - RTOS의 정의와 필요성
   - RTOS와 일반 운영체제의 차이점
   - RTOS의 주요 특징

2. **주요 RTOS 소개**
   - FreeRTOS 개요
   - μC/OS-II 개요
   - 기타 주요 RTOS 개요

3. **RTOS의 기본 개념**
   - 태스크(Task) 개념
   - 태스크 스케줄링
   - 태스크 간 통신 (큐, 세마포어, 뮤텍스 등)

4. **FreeRTOS 설정 및 기본 사용법**
   - FreeRTOS 설치 및 설정 방법
   - 간단한 태스크 생성 및 스케줄링
   - FreeRTOS API 소개

**실습 내용:**

1. **FreeRTOS 설치 및 설정**
   - FreeRTOS 설치 방법
   - 프로젝트 생성 및 설정

2. **간단한 태스크 생성 및 스케줄링**
   - 태스크 생성 코드 작성
   - 태스크 스케줄링 및 실행

3. **태스크 간 통신 예제**
   - 큐를 이용한 태스크 간 데이터 전송
   - 세마포어를 이용한 태스크 동기화

---

**강의 진행:**

1. **강의 시작 (10분)**
   - 강의 목표와 주제를 소개합니다.
   - RTOS의 개념과 필요성을 설명합니다.

2. **RTOS 개요 (20분)**
   - RTOS의 정의와 일반 운영체제와의 차이점을 설명합니다.
   - RTOS의 주요 특징과 사용 사례를 설명합니다.

3. **주요 RTOS 소개 (20분)**
   - FreeRTOS와 μC/OS-II의 기본 구조와 특징을 설명합니다.
   - 각 RTOS의 장단점과 사용 사례를 비교합니다.

4. **RTOS의 기본 개념 (20분)**
   - 태스크와 태스크 스케줄링의 개념을 설명합니다.
   - 태스크 간 통신 방법(큐, 세마포어, 뮤텍스 등)을 소개합니다.

5. **FreeRTOS 설정 및 기본 사용법 (20분)**
   - FreeRTOS 설치 및 설정 방법을 시연합니다.
   - 간단한 태스크 생성 및 스케줄링 예제를 시연합니다.
   - FreeRTOS의 주요 API를 소개합니다.

6. **Q&A 및 실습 준비 (10분)**
   - Q&A 시간을 통해 학생들의 질문에 답변하고, 이해도를 확인합니다.
   - 다음 주 실습을 위한 준비 사항을 안내합니다.

**실습 내용:**

1. **FreeRTOS 설치 및 설정**

**FreeRTOS 설치 및 설정 예제:**

- FreeRTOS 공식 사이트에서 소스 코드 다운로드
- 프로젝트 생성 및 FreeRTOS 소스 코드 추가
- FreeRTOS 설정 파일(config.h) 수정

2. **간단한 태스크 생성 및 스케줄링**

**간단한 태스크 생성 및 스케줄링 예제 코드:**

```c
#include "FreeRTOS.h"
#include "task.h"
#include "stm32f4xx_hal.h"

// 함수 선언
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
void vTask1(void *pvParameters);
void vTask2(void *pvParameters);

int main(void) {
    // HAL 라이브러리 초기화
    HAL_Init();
    // 시스템 클록 설정
    SystemClock_Config();
    // GPIO 초기화
    MX_GPIO_Init();

    // 태스크 생성
    xTaskCreate(vTask1, "Task 1", 128, NULL, 1, NULL);
    xTaskCreate(vTask2, "Task 2", 128, NULL, 1, NULL);

    // 스케줄러 시작
    vTaskStartScheduler();

    // 프로그램이 여기 도달하면 오류
    while (1);
}

// 태스크 1 함수
void vTask1(void *pvParameters) {
    while (1) {
        HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5); // LED 토글
        vTaskDelay(pdMS_TO_TICKS(1000)); // 1초 대기
    }
}

// 태스크 2 함수
void vTask2(void *pvParameters) {
    while (1) {
        HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_6); // LED 토글
        vTaskDelay(pdMS_TO_TICKS(500)); // 0.5초 대기
    }
}

// 시스템 클록 설정 함수 (자동 생성됨)
void SystemClock_Config(void) {
    // 클록 설정 코드
}

// GPIO 초기화 함수 (자동 생성됨)
static void MX_GPIO_Init(void) {
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    __HAL_RCC_GPIOA_CLK_ENABLE();

    // GPIO 핀 설정 (PA5, PA6: LED 핀)
    GPIO_InitStruct.Pin = GPIO_PIN_5 | GPIO_PIN_6;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
}
```

3. **태스크 간 통신 예제**

**큐를 이용한 태스크 간 데이터 전송 예제 코드:**

```c
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "stm32f4xx_hal.h"

// 함수 선언
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
void vSenderTask(void *pvParameters);
void vReceiverTask(void *pvParameters);

QueueHandle_t xQueue;

int main(void) {
    // HAL 라이브러리 초기화
    HAL_Init();
    // 시스템 클록 설정
    SystemClock_Config();
    // GPIO 초기화
    MX_GPIO_Init();

    // 큐 생성
    xQueue = xQueueCreate(5, sizeof(uint32_t));

    // 태스크 생성
    xTaskCreate(vSenderTask, "Sender Task", 128, NULL, 1, NULL);
    xTaskCreate(vReceiverTask, "Receiver Task", 128, NULL, 1, NULL);

    // 스케줄러 시작
    vTaskStartScheduler();

    // 프로그램이 여기 도달하면 오류
    while (1);
}

// 송신 태스크 함수
void vSenderTask(void *pvParameters) {
    uint32_t ulValueToSend = 0;
    while (1) {
        ulValueToSend++;
        xQueueSend(xQueue, &ulValueToSend, portMAX_DELAY);
        vTaskDelay(pdMS_TO_TICKS(1000)); // 1초 대기
    }
}

// 수신 태스크 함수
void vReceiverTask(void *pvParameters) {
    uint32_t ulReceivedValue;
    while (1) {
        xQueueReceive(xQueue, &ulReceivedValue, portMAX_DELAY);
        // 수신한 값에 따라 LED 제어
        if (ulReceivedValue % 2 == 0) {
            HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET);
        } else {
            HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET);
        }
    }
}

// 시스템 클록 설정 함수 (자동 생성됨)
void SystemClock_Config(void) {
    // 클록 설정 코드
}

// GPIO 초기화 함수 (자동 생성됨)
static void MX_GPIO_Init(void) {
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    __HAL_RCC_GPIOA_CLK_ENABLE();

    // GPIO 핀 설정 (PA5: LED 핀)
    GPIO_InitStruct.Pin = GPIO_PIN_5;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
}
```

**과제:**

1. **FreeRTOS 설치 및 설정**
   - FreeRTOS를 설치하고, 프로젝트를 생성하여 설정을 완료합니다.
   - 설치 및 설정 과정을 문서화하여 제출합니다.

2. **간단한 태스크 생성 및 스케줄링**
   - 두 개의 태스크를 생성하여 각각 다른 주기로 LED를 깜빡이도록 프로그램을 작성합니다.
   - 작성한 프로그램을 업로드하고, 실행 결과를 동영상으로 촬영하여 제출합니다.

3. **큐를 이용한 태스크 간 데이터 전송**
   - 큐를 이용하여 태스크 간 데이터를 전송하는 프로그램을 작성합니다.
   - 송신 태스크와 수신 태스크를 구현하고, 송신된 데이터에 따라 LED를 제어합니다.
   - 작성한 프로그램을 업로드하고, 실행 결과를 동영상으로 촬영하여 제출합니다.

이 강의 계획을 통해 학생들은 리얼타임 운영체제(RTOS)의 기본 개념과 주요 기능을 이해하고, FreeRTOS를 사용하여 간단한 태스크를 생성하고 스케줄링하는 방법을 습득할 수 있습니다. 또한, 태스크 간 통신 방법을 학습

하여 실습을 통해 RTOS의 응용 능력을 키울 수 있습니다.