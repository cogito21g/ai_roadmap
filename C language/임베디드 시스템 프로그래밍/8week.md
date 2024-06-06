### 8주차: STM32 통신 프로토콜

**강의 목표:**  
STM32 마이크로컨트롤러에서 I2C, SPI, UART 통신 프로토콜을 설정하고 사용하는 방법을 학습하며, 다양한 센서를 연결하여 데이터를 읽고 처리합니다.

**강의 내용:**

1. **I2C 통신**
   - I2C 통신의 기본 개념
   - I2C의 구조와 동작 원리
   - STM32에서 I2C 설정 및 사용 방법
   - I2C 센서 연결 및 데이터 읽기

2. **SPI 통신**
   - SPI 통신의 기본 개념
   - SPI의 구조와 동작 원리
   - STM32에서 SPI 설정 및 사용 방법
   - SPI 디바이스 연결 및 데이터 읽기

3. **UART 통신**
   - UART 통신의 기본 개념
   - UART의 구조와 동작 원리
   - STM32에서 UART 설정 및 사용 방법
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
   - STM32에서 I2C 설정 방법을 시연합니다.
   - I2C 센서를 연결하고 데이터를 읽는 방법을 설명합니다.

3. **SPI 통신 (20분)**
   - SPI 통신의 구조와 동작 원리를 설명합니다.
   - STM32에서 SPI 설정 방법을 시연합니다.
   - SPI 디바이스를 연결하고 데이터를 읽는 방법을 설명합니다.

4. **UART 통신 (20분)**
   - UART 통신의 구조와 동작 원리를 설명합니다.
   - STM32에서 UART 설정 방법을 시연합니다.
   - UART를 이용하여 시리얼 데이터를 송수신하는 방법을 설명합니다.

5. **실습 준비 및 Q&A (10분)**
   - Q&A 시간을 통해 학생들의 질문에 답변하고, 이해도를 확인합니다.
   - 다음 주 실습을 위한 준비 사항을 안내합니다.

**실습 내용:**

1. **I2C 센서 데이터 읽기**

**I2C 센서 데이터 읽기 예제 코드:**

**main.c 파일:**

```c
#include "main.h"
#include "stm32f4xx_hal.h"

// 함수 선언
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_I2C1_Init(void);

I2C_HandleTypeDef hi2c1;
uint8_t i2cData[2];

int main(void) {
  // HAL 라이브러리 초기화
  HAL_Init();

  // 시스템 클록 설정
  SystemClock_Config();

  // GPIO 초기화
  MX_GPIO_Init();

  // I2C 초기화
  MX_I2C1_Init();

  // 메인 루프
  while (1) {
    // I2C 센서 데이터 읽기
    HAL_I2C_Mem_Read(&hi2c1, 0x48 << 1, 0x00, 1, i2cData, 2, HAL_MAX_DELAY);
    int temp = (i2cData[0] << 8 | i2cData[1]) >> 4;

    // 시리얼 모니터에 출력
    printf("Temperature: %d\n", temp);
    HAL_Delay(1000); // 1초 대기
  }
}

// 시스템 클록 설정 함수 (자동 생성됨)
void SystemClock_Config(void) {
  // 클록 설정 코드
}

// GPIO 초기화 함수 (자동 생성됨)
static void MX_GPIO_Init(void) {
  // GPIO 설정 코드
}

// I2C 초기화 함수 (자동 생성됨)
static void MX_I2C1_Init(void) {
  hi2c1.Instance = I2C1;
  hi2c1.Init.ClockSpeed = 100000;
  hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK) {
    Error_Handler();
  }
}
```

2. **SPI 디바이스 제어**

**SPI 디바이스 제어 예제 코드:**

**main.c 파일:**

```c
#include "main.h"
#include "stm32f4xx_hal.h"

// 함수 선언
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_SPI1_Init(void);

SPI_HandleTypeDef hspi1;
uint8_t spiData[2];

int main(void) {
  // HAL 라이브러리 초기화
  HAL_Init();

  // 시스템 클록 설정
  SystemClock_Config();

  // GPIO 초기화
  MX_GPIO_Init();

  // SPI 초기화
  MX_SPI1_Init();

  // 메인 루프
  while (1) {
    // SPI 디바이스에 데이터 전송
    spiData[0] = 0x01; // 명령어
    spiData[1] = 0x02; // 데이터
    HAL_SPI_Transmit(&hspi1, spiData, 2, HAL_MAX_DELAY);

    // 시리얼 모니터에 출력
    printf("Data sent via SPI\n");
    HAL_Delay(1000); // 1초 대기
  }
}

// 시스템 클록 설정 함수 (자동 생성됨)
void SystemClock_Config(void) {
  // 클록 설정 코드
}

// GPIO 초기화 함수 (자동 생성됨)
static void MX_GPIO_Init(void) {
  // GPIO 설정 코드
}

// SPI 초기화 함수 (자동 생성됨)
static void MX_SPI1_Init(void) {
  hspi1.Instance = SPI1;
  hspi1.Init.Mode = SPI_MODE_MASTER;
  hspi1.Init.Direction = SPI_DIRECTION_2LINES;
  hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;
  hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;
  hspi1.Init.NSS = SPI_NSS_SOFT;
  hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_16;
  hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi1.Init.CRCPolynomial = 10;
  if (HAL_SPI_Init(&hspi1) != HAL_OK) {
    Error_Handler();
  }
}
```

3. **UART 통신을 이용한 데이터 전송**

**UART 통신 예제 코드:**

**main.c 파일:**

```c
#include "main.h"
#include "stm32f4xx_hal.h"

// 함수 선언
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);

UART_HandleTypeDef huart2;
uint8_t uartData[] = "Hello, UART!\r\n";

int main(void) {
  // HAL 라이브러리 초기화
  HAL_Init();

  // 시스템 클록 설정
  SystemClock_Config();

  // GPIO 초기화
  MX_GPIO_Init();

  // UART 초기화
  MX_USART2_UART_Init();

  // 메인 루프
  while (1) {
    // UART를 이용한 데이터 전송
    HAL_UART_Transmit(&huart2, uartData, sizeof(uartData) - 1, HAL_MAX_DELAY);

    // 시리얼 모니터에 출력
    HAL_UART_Transmit(&huart2, (uint8_t *)"Data sent via UART\n", 19, HAL_MAX_DELAY);
    HAL_Delay(1000); // 1초 대기
  }
}

// 시스템 클록 설정 함수 (자동 생성됨)
void SystemClock_Config(void) {
  // 클록 설정 코드
}

// GPIO 초기화 함수 (자동 생성됨)
static void MX_GPIO_Init(void) {
  // GPIO 설정 코드
}

// UART 초기화 함수 (자동 생성됨)
static void MX_USART2_UART_Init(void) {
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity =

 UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK) {
    Error_Handler();
  }
}
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

이 강의 계획을 통해 학생들은 STM32 마이크로컨트롤러에서 I2C, SPI, UART 통신 프로토콜을 설정하고 사용하는 방법을 학습하며, 다양한 센서를 연결하여 데이터를 읽고 처리하는 기술을 습득할 수 있습니다.