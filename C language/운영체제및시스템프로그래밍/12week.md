### 12주차: 입출력 시스템

**강의 목표:** 입출력 시스템의 개념과 중요성을 이해하고, 폴링과 인터럽트 기반 I/O를 학습합니다. 또한, DMA(Direct Memory Access)와 입출력 버스의 작동 원리를 이해하고 시뮬레이션합니다.

**강의 내용:**

1. **입출력 시스템 개념**
   - 입출력 시스템이란 무엇인가?
     - 컴퓨터와 외부 장치 간의 데이터 전송을 담당하는 시스템
   - 입출력 시스템의 구성 요소
     - 입출력 장치, 입출력 제어기, 장치 드라이버

2. **폴링과 인터럽트 기반 I/O**
   - 폴링 (Polling)
     - CPU가 주기적으로 I/O 장치를 확인하여 데이터 전송을 처리
     - 장점: 간단한 구현
     - 단점: CPU 자원 낭비
   - 인터럽트 (Interrupt)
     - I/O 장치가 데이터를 전송할 준비가 되면 CPU에 신호를 보내 작업을 처리
     - 장점: 효율적인 CPU 자원 사용
     - 단점: 복잡한 구현

3. **DMA(Direct Memory Access)**
   - DMA 개념
     - CPU의 개입 없이 메모리와 I/O 장치 간의 직접 데이터 전송
   - DMA 작동 방식
     - DMA 컨트롤러가 데이터 전송을 관리
     - CPU는 DMA 시작과 종료만 관리

4. **입출력 버스**
   - 입출력 버스의 역할
     - CPU와 I/O 장치 간의 데이터 전송 통로
   - 주요 입출력 버스
     - PCI (Peripheral Component Interconnect)
     - USB (Universal Serial Bus)
     - SATA (Serial ATA)

**실습:**

1. **폴링을 이용한 I/O 처리 예제**
   - 폴링을 사용하여 간단한 I/O 장치와의 데이터 전송을 시뮬레이션합니다.

```c
#include <stdio.h>
#include <unistd.h>
#include <stdbool.h>

bool device_ready = false;

void check_device() {
    // 장치가 데이터를 전송할 준비가 되었다고 가정
    sleep(2);
    device_ready = true;
}

void polling_io() {
    printf("Polling: Waiting for device to be ready...\n");
    while (!device_ready) {
        check_device();
    }
    printf("Device is ready. Processing data...\n");
}

int main() {
    polling_io();
    return 0;
}
```

2. **인터럽트를 이용한 I/O 처리 예제**
   - 인터럽트를 사용하여 간단한 I/O 장치와의 데이터 전송을 시뮬레이션합니다.

```c
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <stdbool.h>

bool device_ready = false;

void device_interrupt(int signum) {
    device_ready = true;
    printf("Interrupt: Device is ready. Processing data...\n");
}

void setup_interrupt() {
    signal(SIGUSR1, device_interrupt);
}

int main() {
    setup_interrupt();
    printf("Waiting for device interrupt...\n");
    // 시뮬레이션을 위해 2초 후에 인터럽트 발생
    sleep(2);
    raise(SIGUSR1);
    
    while (!device_ready) {
        pause(); // 인터럽트를 기다림
    }
    
    return 0;
}
```

3. **DMA 시뮬레이션**
   - DMA를 사용하여 메모리와 I/O 장치 간의 데이터 전송을 시뮬레이션합니다.

```c
#include <stdio.h>
#include <unistd.h>

void dma_transfer() {
    printf("DMA: Starting data transfer...\n");
    sleep(2); // 데이터 전송 시간 시뮬레이션
    printf("DMA: Data transfer complete.\n");
}

int main() {
    printf("CPU: Initiating DMA transfer...\n");
    dma_transfer();
    printf("CPU: DMA transfer completed.\n");
    return 0;
}
```

**과제:**

1. **폴링과 인터럽트의 성능 비교**
   - 폴링과 인터럽트를 사용하여 동일한 I/O 작업을 수행하고, 성능 차이를 비교합니다.
   - 각 방법의 장단점을 분석하고, 실제 적용 사례를 조사합니다.

2. **DMA 컨트롤러 구현**
   - DMA 컨트롤러를 시뮬레이션하여 CPU 개입 없이 메모리와 I/O 장치 간의 데이터 전송을 관리합니다.
   - DMA 전송 완료 후 CPU에 신호를 보내는 기능을 추가합니다.

```c
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

pthread_mutex_t dma_mutex;
pthread_cond_t dma_cond;
bool dma_complete = false;

void *dma_transfer(void *arg) {
    pthread_mutex_lock(&dma_mutex);
    printf("DMA: Starting data transfer...\n");
    sleep(2); // 데이터 전송 시간 시뮬레이션
    dma_complete = true;
    printf("DMA: Data transfer complete.\n");
    pthread_cond_signal(&dma_cond);
    pthread_mutex_unlock(&dma_mutex);
    return NULL;
}

int main() {
    pthread_t dma_thread;
    pthread_mutex_init(&dma_mutex, NULL);
    pthread_cond_init(&dma_cond, NULL);

    printf("CPU: Initiating DMA transfer...\n");
    pthread_create(&dma_thread, NULL, dma_transfer, NULL);

    pthread_mutex_lock(&dma_mutex);
    while (!dma_complete) {
        pthread_cond_wait(&dma_cond, &dma_mutex);
    }
    pthread_mutex_unlock(&dma_mutex);

    printf("CPU: DMA transfer completed.\n");

    pthread_join(dma_thread, NULL);
    pthread_mutex_destroy(&dma_mutex);
    pthread_cond_destroy(&dma_cond);

    return 0;
}
```

**퀴즈 및 해설:**

1. **폴링과 인터럽트의 차이점은 무엇인가요?**
   - 폴링은 CPU가 주기적으로 I/O 장치를 확인하여 데이터 전송을 처리하는 방식이고, 인터럽트는 I/O 장치가 데이터를 전송할 준비가 되면 CPU에 신호를 보내 작업을 처리하는 방식입니다.

2. **DMA의 장점은 무엇인가요?**
   - DMA는 CPU의 개입 없이 메모리와 I/O 장치 간의 직접 데이터 전송을 가능하게 하여 CPU의 부하를 줄이고 데이터 전송 속도를 높입니다.

3. **입출력 버스의 역할은 무엇인가요?**
   - 입출력 버스는 CPU와 I/O 장치 간의 데이터 전송 통로로서, 데이터, 주소, 제어 신호를 전달하여 장치 간의 통신을 가능하게 합니다.

**해설:**

1. **폴링과 인터럽트의 차이점**은 폴링은 CPU가 주기적으로 I/O 장치를 확인하여 데이터 전송을 처리하는 방식이고, 인터럽트는 I/O 장치가 데이터를 전송할 준비가 되면 CPU에 신호를 보내 작업을 처리하는 방식입니다. 폴링은 간단하지만 CPU 자원을 낭비할 수 있으며, 인터럽트는 효율적이지만 구현이 복잡할 수 있습니다.

2. **DMA의 장점**은 CPU의 개입 없이 메모리와 I/O 장치 간의 직접 데이터 전송을 가능하게 하여 CPU의 부하를 줄이고 데이터 전송 속도를 높이는 것입니다. 이는 대량의 데이터를 빠르게 전송해야 하는 경우에 유용합니다.

3. **입출력 버스의 역할**은 CPU와 I/O 장치 간의 데이터 전송 통로로서, 데이터, 주소, 제어 신호를 전달하여 장치 간의 통신을 가능하게 합니다. 입출력 버스는 다양한 장치가 CPU와 효율적으로 통신할 수 있도록 도와줍니다.

이로써 12주차 강의를 마무리합니다. 다음 주차에는 네트워크 프로그래밍 기초에 대해 학습합니다. 추가 질문이 있거나 특정 주제에 대해 더 알고 싶다면 언제든지 문의해 주세요.