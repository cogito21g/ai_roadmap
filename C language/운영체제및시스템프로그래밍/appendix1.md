### 멀티쓰레딩과 동기화

**강의 목표:** 멀티쓰레딩의 개념과 필요성을 이해하고, Pthreads 라이브러리를 사용하여 스레드를 생성하는 방법을 학습합니다. 또한, 스레드 간의 동기화 기법을 배우고 이를 실습을 통해 적용합니다.

**강의 내용:**

1. **멀티쓰레딩의 개념과 필요성**
   - 멀티쓰레딩이란 무엇인가?
     - 프로세스 내에서 여러 스레드를 병렬로 실행
     - 각 스레드는 독립적인 실행 흐름을 가짐
   - 멀티쓰레딩의 필요성
     - 병렬 처리 및 성능 향상
     - 응답성 향상
     - 리소스 공유 및 효율적인 사용

2. **Pthreads 라이브러리를 사용한 스레드 생성**
   - Pthreads란 무엇인가?
     - POSIX 스레드(POSIX Threads)의 약자
     - 유닉스/리눅스 시스템에서 스레드를 생성하고 관리하기 위한 표준 API
   - 스레드 생성 및 종료
     - `pthread_create` 함수: 스레드를 생성하는 함수
     - `pthread_exit` 함수: 스레드를 종료하는 함수
   - 스레드 조인
     - `pthread_join` 함수: 생성된 스레드가 종료될 때까지 기다리는 함수

3. **스레드 동기화 기법**
   - 뮤텍스(Mutex)
     - 상호 배제(Mutual Exclusion) 객체
     - `pthread_mutex_t` 타입 사용
     - `pthread_mutex_init`, `pthread_mutex_lock`, `pthread_mutex_unlock`, `pthread_mutex_destroy` 함수
   - 세마포어(Semaphore)
     - 신호와 대기 메커니즘
     - `sem_t` 타입 사용
     - `sem_init`, `sem_wait`, `sem_post`, `sem_destroy` 함수
   - 조건 변수(Condition Variable)
     - 특정 조건을 기다리거나 신호를 보내는 메커니즘
     - `pthread_cond_t` 타입 사용
     - `pthread_cond_init`, `pthread_cond_wait`, `pthread_cond_signal`, `pthread_cond_destroy` 함수

**실습:**

1. **Pthreads를 이용한 스레드 생성 및 동기화**
   - 스레드를 생성하여 간단한 작업을 수행하고, 메인 스레드가 생성된 스레드의 종료를 기다리는 프로그램 작성

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 5

void *PrintHello(void *threadid) {
    long tid;
    tid = (long)threadid;
    printf("Hello World! It's me, thread #%ld!\n", tid);
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    pthread_t threads[NUM_THREADS];
    int rc;
    long t;
    for (t = 0; t < NUM_THREADS; t++) {
        rc = pthread_create(&threads[t], NULL, PrintHello, (void *)t);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    for (t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    pthread_exit(NULL);
}
```

2. **뮤텍스를 이용한 동기화**
   - 여러 스레드가 공유 자원을 안전하게 사용할 수 있도록 뮤텍스를 사용하여 동기화

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 5

int counter = 0;
pthread_mutex_t mutex;

void *IncrementCounter(void *threadid) {
    long tid;
    tid = (long)threadid;
    pthread_mutex_lock(&mutex);
    counter++;
    printf("Thread #%ld, Counter: %d\n", tid, counter);
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    pthread_t threads[NUM_THREADS];
    int rc;
    long t;
    pthread_mutex_init(&mutex, NULL);

    for (t = 0; t < NUM_THREADS; t++) {
        rc = pthread_create(&threads[t], NULL, IncrementCounter, (void *)t);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    for (t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    pthread_mutex_destroy(&mutex);
    pthread_exit(NULL);
}
```

**과제:**

1. **세마포어를 이용한 동기화**
   - 세마포어를 사용하여 여러 스레드가 특정 자원을 동시에 사용하지 않도록 동기화하는 프로그램 작성

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>

#define NUM_THREADS 5

int counter = 0;
sem_t semaphore;

void *IncrementCounter(void *threadid) {
    long tid;
    tid = (long)threadid;
    sem_wait(&semaphore);
    counter++;
    printf("Thread #%ld, Counter: %d\n", tid, counter);
    sem_post(&semaphore);
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    pthread_t threads[NUM_THREADS];
    int rc;
    long t;
    sem_init(&semaphore, 0, 1);

    for (t = 0; t < NUM_THREADS; t++) {
        rc = pthread_create(&threads[t], NULL, IncrementCounter, (void *)t);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    for (t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    sem_destroy(&semaphore);
    pthread_exit(NULL);
}
```

2. **조건 변수를 이용한 동기화**
   - 조건 변수를 사용하여 특정 조건이 만족될 때까지 스레드가 대기하고, 조건이 만족되면 신호를 보내는 프로그램 작성

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 5

pthread_mutex_t mutex;
pthread_cond_t cond_var;
int ready = 0;

void *WaitForCondition(void *threadid) {
    long tid;
    tid = (long)threadid;
    pthread_mutex_lock(&mutex);
    while (!ready) {
        pthread_cond_wait(&cond_var, &mutex);
    }
    printf("Thread #%ld is proceeding after condition is met.\n", tid);
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}

void *SetCondition(void *threadid) {
    long tid;
    tid = (long)threadid;
    pthread_mutex_lock(&mutex);
    ready = 1;
    pthread_cond_broadcast(&cond_var);
    printf("Thread #%ld set the condition and signaled all waiting threads.\n", tid);
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    pthread_t threads[NUM_THREADS + 1];
    int rc;
    long t;
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond_var, NULL);

    for (t = 0; t < NUM_THREADS; t++) {
        rc = pthread_create(&threads[t], NULL, WaitForCondition, (void *)t);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    rc = pthread_create(&threads[NUM_THREADS], NULL, SetCondition, (void *)(NUM_THREADS));
    if (rc) {
        printf("ERROR; return code from pthread_create() is %d\n", rc);
        exit(-1);
    }

    for (t = 0; t <= NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond_var);
    pthread_exit(NULL);
}
```

**퀴즈 및 해설:**

1. **스레드의 주요 특징은 무엇인가요?**
   - 스레드는 프로세스 내에서 실행되는 경량 프로세스로, 프로세스의 자원을 공유하며 독립적으로 실행됩니다. 스레드는 병렬 처리를 통해 성능을 향상시키고 응답성을 개선합니다.

2. **뮤텍스와 세마포어의 차이점은 무엇인가요?**
   - 뮤텍스는 상호 배제를 위한 객체로, 하나의 스레드만이 특정 자원에 접근할 수 있도록 합니다. 세마포어는 신호와 대기 메커니즘으로, 여러 스레드가 동시에 자원에 접근할 수 있도록 허용하며, 특정 수의 스레드만 자원에 접근할 수 있도록 제한할 수 있습니다.

3. **조건 변수의 역할은 무엇인가요?**
   - 조건 변수는 특정 조건이 만족될 때까지 스레드가 대기하고, 조건이 만족되면 신호를 보내어 대기 중인 스레드를 깨우는 동기화 메커니즘입니다. 이는 스

레드 간의 협력을 통해 효율적인 동기화를 가능하게 합니다.

**해설:**

1. **스레드의 주요 특징**은 프로세스 내에서 실행되는 경량 프로세스이며, 프로세스의 자원을 공유하지만 독립적으로 실행된다는 것입니다. 스레드는 병렬 처리를 통해 성능을 향상시키고 응답성을 개선할 수 있습니다.

2. **뮤텍스와 세마포어의 차이점**은 뮤텍스가 상호 배제를 위한 객체로 하나의 스레드만 자원에 접근할 수 있도록 하는 반면, 세마포어는 신호와 대기 메커니즘으로 여러 스레드가 자원에 접근할 수 있도록 허용하고, 특정 수의 스레드만 자원에 접근할 수 있도록 제한할 수 있다는 점입니다.

3. **조건 변수의 역할**은 특정 조건이 만족될 때까지 스레드가 대기하고, 조건이 만족되면 신호를 보내어 대기 중인 스레드를 깨우는 것입니다. 이는 스레드 간의 협력을 통해 효율적인 동기화를 가능하게 합니다.

이로써 14주차 강의를 마무리합니다. 다음 주차에는 운영체제와 시스템 프로그래밍 기초에 대해 학습합니다. 추가 질문이 있거나 특정 주제에 대해 더 알고 싶다면 언제든지 문의해 주세요.