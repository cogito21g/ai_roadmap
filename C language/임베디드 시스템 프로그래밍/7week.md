### 7주차: STM32 타이머 및 인터럽트

**강의 목표:**  
STM32 마이크로컨트롤러에서 타이머와 인터럽트를 설정하고 사용하는 방법을 학습하며, 타이머를 이용한 주기적 작업 실행과 인터럽트를 이용한 버튼 제어를 구현합니다.

**강의 내용:**

1. **타이머 개요**
   - 타이머의 기본 개념
   - 타이머의 주요 기능과 사용 사례
   - STM32 타이머 종류 및 특성

2. **타이머 설정**
   - STM32CubeMX를 이용한 타이머 설정
   - HAL 라이브러리를 이용한 타이머 초기화 및 사용
   - 주기적 작업을 위한 타이머 설정

3. **인터럽트 개요**
   - 인터럽트의 기본 개념
   - 인터럽트의 종류와 사용 사례
   - 인터럽트 처리 순서

4. **인터럽트 설정 및 사용**
   - STM32CubeMX를 이용한 인터럽트 설정
   - HAL 라이브러리를 이용한 인터럽트 초기화 및 처리
   - 버튼 인터럽트를 이용한 LED 제어

**실습 내용:**

1. **타이머를 이용한 주기적 작업**
   - 타이머를 이용하여 주기적으로 LED를 점멸시키는 프로그램 작성

2. **인터럽트를 이용한 버튼 제어**
   - 버튼 인터럽트를 이용하여 LED를 제어하는 프로그램 작성

---

**강의 진행:**

1. **강의 시작 (10분)**
   - 강의 목표와 주제를 소개합니다.
   - 타이머와 인터럽트의 기본 개념을 설명합니다.

2. **타이머 개요 및 설정 (20분)**
   - 타이머의 주요 기능과 사용 사례를 설명합니다.
   - STM32CubeMX를 이용하여 타이머를 설정하는 방법을 시연합니다.
   - HAL 라이브러리를 이용하여 타이머를 초기화하고 사용하는 방법을 설명합니다.

3. **타이머를 이용한 주기적 작업 (20분)**
   - 타이머를 이용하여 주기적으로 LED를 점멸시키는 프로그램을 작성합니다.
   - 프로그램을 업로드하고 실행하여 결과를 확인합니다.

4. **인터럽트 개요 및 설정 (20분)**
   - 인터럽트의 종류와 사용 사례를 설명합니다.
   - STM32CubeMX를 이용하여 인터럽트를 설정하는 방법을 시연합니다.
   - HAL 라이브러리를 이용하여 인터럽트를 초기화하고 처리하는 방법을 설명합니다.

5. **인터럽트를 이용한 버튼 제어 (20분)**
   - 버튼 인터럽트를 이용하여 LED를 제어하는 프로그램을 작성합니다.
   - 프로그램을 업로드하고 실행하여 결과를 확인합니다.

6. **Q&A 및 실습 준비 (10분)**
   - Q&A 시간을 통해 학생들의 질문에 답변하고, 이해도를 확인합니다.
   - 다음 주 실습을 위한 준비 사항을 안내합니다.

**실습 내용:**

1. **타이머를 이용한 주기적 작업**

**타이머를 이용한 LED 점멸 예제 코드:**

**main.c 파일:**

```c
#include "main.h"
#include "stm32f4xx_hal.h"

// 함수 선언
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_TIM3_Init(void);

TIM_HandleTypeDef htim3;

int main(void) {
  // HAL 라이브러리 초기화
  HAL_Init();

  // 시스템 클록 설정
  SystemClock_Config();

  // GPIO 초기화
  MX_GPIO_Init();

  // 타이머 초기화
  MX_TIM3_Init();

  // 타이머 시작
  HAL_TIM_Base_Start_IT(&htim3);

  // 메인 루프
  while (1) {
  }
}

// 타이머 인터럽트 콜백 함수
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) {
  if (htim->Instance == TIM3) {
    HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);
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

  GPIO_InitStruct.Pin = GPIO_PIN_5;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
}

// 타이머 초기화 함수 (자동 생성됨)
static void MX_TIM3_Init(void) {
  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 8399;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 9999;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim3) != HAL_OK) {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim3, &sClockSourceConfig) != HAL_OK) {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK) {
    Error_Handler();
  }
}
```

2. **인터럽트를 이용한 버튼 제어**

**버튼 인터럽트를 이용한 LED 제어 예제 코드:**

**main.c 파일:**

```c
#include "main.h"
#include "stm32f4xx_hal.h"

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
  }
}

// 외부 인터럽트 콜백 함수
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin) {
  if (GPIO_Pin == GPIO_PIN_13) {
    HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);
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
  __HAL_RCC_GPIOC_CLK_ENABLE();

  // LED 핀 설정 (PA5)
  GPIO_InitStruct.Pin = GPIO_PIN_5;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  // 버튼 핀 설정 (PC13)
  GPIO_InitStruct.Pin = GPIO_PIN_13;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  // 외부 인터럽트 설정
  HAL_NVIC_SetPriority(EXTI15_10_IRQn, 2, 0);
  HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);
}
```

**과제:**

1. **타이머를 이용한 주기적 작업 프로그램 작성**
   - 타이머를 이용하여 주기적으로 LED를 깜빡이는 프로그램을 작성합니다.
   - 작성한 프로그램을 업로드하고, 실행 결과를 동영상으로 촬영하여 제출합니다.

2. **버튼 인터럽트를 이용한 LED 제어 프로그램 작성**
   - 버튼 인터럽트를 이용하여 LED를 제어하는 프로그램을 작성합니다.
   - 버튼을 누를 때마다 LED가 켜졌다가 꺼지도록 설정합니다.
   - 작성한 프로그램을 업로드하고, 실행 결과를 동영상으로 촬영하여 제출합니다.

이 강의 계획을 통해 학생들은 STM32 마이크로컨트롤러에서 타이머와 인터럽트를 설정하고 사용하는 방법을 학습하며, 주기적 작업 실행과 인터럽트를 이용한 제어 프로그램을 작성하는 기술을 습득할 수 있습니다.