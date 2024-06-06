### 13주차: RTOS 태스크 관리

**강의 목표:**  
RTOS에서 태스크 간 통신과 동기화 방법을 학습하고, 큐, 세마포어, 뮤텍스를 사용하여 여러 태스크 간의 데이터 교환과 동기화를 구현합니다.

**강의 내용:**

1. **큐 (Queue)**
   - 큐의 개념과 역할
   - 큐의 생성 및 사용 방법
   - 큐를 이용한 태스크 간 데이터 전송

2. **세마포어 (Semaphore)**
   - 세마포어의 개념과 역할
   - 이진 세마포어와 카운팅 세마포어의 차이점
   - 세마포어를 이용한 태스크 동기화

3. **뮤텍스 (Mutex)**
   - 뮤텍스의 개념과 역할
   - 뮤텍스와 세마포어의 차이점
   - 뮤텍스를 이용한 태스크 간 자원 보호

4. **RTOS의 주요 API 사용법**
   - 큐, 세마포어, 뮤텍스를 생성하고 사용하는 주요 API 설명
   - 예제 코드를 통한 이해

**실습 내용:**

1. **큐를 이용한 태스크 간 데이터 전송**
   - 큐를 생성하고, 송신 태스크와 수신 태스크를 작성하여 데이터를 전송

2. **세마포어를 이용한 태스크 동기화**
   - 이진 세마포어를 생성하여 두 태스크 간의 동기화 구현
   - 카운팅 세마포어를 사용하여 다수의 태스크 간의 동기화 구현

3. **뮤텍스를 이용한 태스크 간 자원 보호**
   - 뮤텍스를 생성하여 공유 자원을 보호하는 태스크 작성

---

**강의 진행:**

1. **강의 시작 (10분)**
   - 강의 목표와 주제를 소개합니다.
   - 큐, 세마포어, 뮤텍스의 개념과 역할을 간단히 설명합니다.

2. **큐 (20분)**
   - 큐의 개념과 역할을 설명합니다.
   - 큐의 생성 및 사용 방법을 설명하고, 주요 API를 소개합니다.
   - 큐를 이용한 태스크 간 데이터 전송 예제를 설명합니다.

3. **세마포어 (20분)**
   - 세마포어의 개념과 역할을 설명합니다.
   - 이진 세마포어와 카운팅 세마포어의 차이점을 설명합니다.
   - 세마포어를 이용한 태스크 동기화 예제를 설명합니다.

4. **뮤텍스 (20분)**
   - 뮤텍스의 개념과 역할을 설명합니다.
   - 뮤텍스와 세마포어의 차이점을 설명합니다.
   - 뮤텍스를 이용한 태스크 간 자원 보호 예제를 설명합니다.

5. **RTOS 주요 API 사용법 (20분)**
   - 큐, 세마포어, 뮤텍스를 생성하고 사용하는 주요 API를 설명합니다.
   - 예제 코드를 통해 이해를 돕습니다.

6. **Q&A 및 실습 준비 (10분)**
   - Q&A 시간을 통해 학생들의 질문에 답변하고, 이해도를 확인합니다.
   - 다음 주 실습을 위한 준비 사항을 안내합니다.

**실습 내용:**

1. **큐를 이용한 태스크 간 데이터 전송**

**큐를 이용한 데이터 전송 예제 코드:**

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

2. **세마포어를 이용한 태스크 동기화**

**이진 세마포어를 이용한 동기화 예제 코드:**

```c
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"
#include "stm32f4xx_hal.h"

// 함수 선언
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
void vTask1(void *pvParameters);
void vTask2(void *pvParameters);

SemaphoreHandle_t xBinarySemaphore;

int main(void) {
    // HAL 라이브러리 초기화
    HAL_Init();
    // 시스템 클록 설정
    SystemClock_Config();
    // GPIO 초기화
    MX_GPIO_Init();

    // 이진 세마포어 생성
    xBinarySemaphore = xSemaphoreCreateBinary();

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
        // LED 켜기
        HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET);
        // 1초 대기
        vTaskDelay(pdMS_TO_TICKS(1000));
        // LED 끄기
        HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET);
        // 세마포어 해제
        xSemaphoreGive(xBinarySemaphore);
    }
}

// 태스크 2 함수
void vTask2(void *pvParameters) {
    while (1) {
        // 세마포어 대기
        if (xSemaphoreTake(xBinarySemaphore, portMAX_DELAY) == pdTRUE) {
            // LED 켜기
            HAL_GPIO_WritePin(GPIOA, GPIO_PIN_6, GPIO_PIN_SET);
            // 1초 대기
            vTaskDelay(pdMS_TO_TICKS(1000));
            // LED 끄기
            HAL_GPIO_WritePin(GPIOA, GPIO_PIN_6, GPIO_PIN_RESET);
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

    // GPIO 핀 설정 (PA5, PA6: LED 핀)
    GPIO_InitStruct.Pin = GPIO_PIN_5 | GPIO_PIN_6;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
}
```

3. **뮤텍스를 이용한 태스크 간 자원 보호**

**뮤텍스를 이용한 자원 보호 예제 코드:**

```c
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"
#include "stm32f4xx_hal.h"

// 함수 선언
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
void vTask1(void *pvParameters);
void vTask2(void *pvParameters);

SemaphoreHandle_t

 xMutex;

int main(void) {
    // HAL 라이브러리 초기화
    HAL_Init();
    // 시스템 클록 설정
    SystemClock_Config();
    // GPIO 초기화
    MX_GPIO_Init();

    // 뮤텍스 생성
    xMutex = xSemaphoreCreateMutex();

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
        // 뮤텍스 획득
        if (xSemaphoreTake(xMutex, portMAX_DELAY) == pdTRUE) {
            // LED 켜기
            HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET);
            // 1초 대기
            vTaskDelay(pdMS_TO_TICKS(1000));
            // LED 끄기
            HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET);
            // 뮤텍스 해제
            xSemaphoreGive(xMutex);
        }
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

// 태스크 2 함수
void vTask2(void *pvParameters) {
    while (1) {
        // 뮤텍스 획득
        if (xSemaphoreTake(xMutex, portMAX_DELAY) == pdTRUE) {
            // LED 켜기
            HAL_GPIO_WritePin(GPIOA, GPIO_PIN_6, GPIO_PIN_SET);
            // 1초 대기
            vTaskDelay(pdMS_TO_TICKS(1000));
            // LED 끄기
            HAL_GPIO_WritePin(GPIOA, GPIO_PIN_6, GPIO_PIN_RESET);
            // 뮤텍스 해제
            xSemaphoreGive(xMutex);
        }
        vTaskDelay(pdMS_TO_TICKS(500));
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

**과제:**

1. **큐를 이용한 태스크 간 데이터 전송**
   - 큐를 생성하여 두 태스크 간 데이터를 전송하는 프로그램을 작성합니다.
   - 작성한 프로그램을 업로드하고, 실행 결과를 동영상으로 촬영하여 제출합니다.

2. **세마포어를 이용한 태스크 동기화**
   - 이진 세마포어를 사용하여 두 태스크 간 동기화를 구현하는 프로그램을 작성합니다.
   - 작성한 프로그램을 업로드하고, 실행 결과를 동영상으로 촬영하여 제출합니다.

3. **뮤텍스를 이용한 태스크 간 자원 보호**
   - 뮤텍스를 사용하여 공유 자원을 보호하는 프로그램을 작성합니다.
   - 작성한 프로그램을 업로드하고, 실행 결과를 동영상으로 촬영하여 제출합니다.

이 강의 계획을 통해 학생들은 RTOS에서 태스크 간 통신과 동기화 방법을 학습하고, 큐, 세마포어, 뮤텍스를 사용하여 여러 태스크 간의 데이터 교환과 동기화를 구현하는 기술을 습득할 수 있습니다.