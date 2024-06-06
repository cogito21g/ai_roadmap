### 10주차: 동기화 문제와 해결

**강의 목표:** 동기화 문제의 개념과 중요성을 이해하고, 교착상태를 예방, 회피, 탐지하는 다양한 기법을 학습합니다. 또한, 동기화 문제를 해결하기 위해 뮤텍스와 세마포어를 사용하는 방법을 실습합니다.

**강의 내용:**

1. **동기화 문제**
   - 동기화란 무엇인가?
     - 여러 스레드가 공유 자원에 접근할 때 데이터의 일관성을 유지하는 기법
   - 동기화 문제의 예시
     - 경쟁 조건 (Race Condition)
     - 상호 배제 (Mutual Exclusion)
     - 임계 구역 (Critical Section)

2. **교착상태 (Deadlock)**
   - 교착상태란 무엇인가?
     - 두 개 이상의 프로세스가 서로 상대방의 자원을 기다리며 무한 대기 상태에 빠지는 상황
   - 교착상태의 조건
     - 상호 배제 (Mutual Exclusion)
     - 점유 대기 (Hold and Wait)
     - 비선점 (No Preemption)
     - 순환 대기 (Circular Wait)

3. **교착상태 예방, 회피 및 탐지 기법**
   - 교착상태 예방 (Deadlock Prevention)
     - 교착상태 조건 중 하나 이상을 제거
   - 교착상태 회피 (Deadlock Avoidance)
     - 안전 상태를 유지하면서 자원 할당
     - 은행가 알고리즘 (Banker's Algorithm)
   - 교착상태 탐지 (Deadlock Detection)
     - 교착상태 발생 시 이를 탐지하고 복구

4. **뮤텍스와 세마포어를 이용한 동기화 문제 해결**
   - 뮤텍스 (Mutex)
     - 상호 배제 객체
     - `pthread_mutex_t` 타입 사용
     - `pthread_mutex_init`, `pthread_mutex_lock`, `pthread_mutex_unlock`, `pthread_mutex_destroy` 함수
   - 세마포어 (Semaphore)
     - 신호와 대기 메커니즘
     - `sem_t` 타입 사용
     - `sem_init`, `sem_wait`, `sem_post`, `sem_destroy` 함수

**실습:**

1. **뮤텍스를 이용한 동기화 문제 해결**

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 5

int counter = 0;
pthread_mutex_t mutex;

void *increment_counter(void *threadid) {
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
        rc = pthread_create(&threads[t], NULL, increment_counter, (void *)t);
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

2. **세마포어를 이용한 동기화 문제 해결**

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>

#define NUM_THREADS 5

int counter = 0;
sem_t semaphore;

void *increment_counter(void *threadid) {
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
        rc = pthread_create(&threads[t], NULL, increment_counter, (void *)t);
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

**과제:**

1. **교착상태 예방 기법 구현**
   - 상호 배제, 점유 대기, 비선점, 순환 대기 조건 중 하나를 제거하여 교착상태를 예방하는 프로그램을 작성합니다.

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define NUM_THREADS 5

pthread_mutex_t mutex1;
pthread_mutex_t mutex2;

void *thread_func1(void *arg) {
    pthread_mutex_lock(&mutex1);
    sleep(1); // 교착상태 시뮬레이션을 위한 지연
    pthread_mutex_lock(&mutex2);

    printf("Thread 1: locked mutex1 and mutex2\n");

    pthread_mutex_unlock(&mutex2);
    pthread_mutex_unlock(&mutex1);
    pthread_exit(NULL);
}

void *thread_func2(void *arg) {
    pthread_mutex_lock(&mutex2);
    sleep(1); // 교착상태 시뮬레이션을 위한 지연
    pthread_mutex_lock(&mutex1);

    printf("Thread 2: locked mutex2 and mutex1\n");

    pthread_mutex_unlock(&mutex1);
    pthread_mutex_unlock(&mutex2);
    pthread_exit(NULL);
}

int main() {
    pthread_t thread1, thread2;

    pthread_mutex_init(&mutex1, NULL);
    pthread_mutex_init(&mutex2, NULL);

    pthread_create(&thread1, NULL, thread_func1, NULL);
    pthread_create(&thread2, NULL, thread_func2, NULL);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    pthread_mutex_destroy(&mutex1);
    pthread_mutex_destroy(&mutex2);

    return 0;
}
```

2. **은행가 알고리즘 구현**
   - 교착상태를 회피하기 위한 은행가 알고리즘을 구현합니다. 각 프로세스의 자원 요청을 안전 상태를 유지하면서 처리합니다.

```c
#include <stdio.h>
#include <stdbool.h>

#define P 5 // 프로세스 수
#define R 3 // 자원 유형 수

int available[R] = {3, 3, 2}; // 가용 자원
int maximum[P][R] = { {7, 5, 3}, {3, 2, 2}, {9, 0, 2}, {2, 2, 2}, {4, 3, 3} }; // 최대 자원 요구
int allocation[P][R] = { {0, 1, 0}, {2, 0, 0}, {3, 0, 2}, {2, 1, 1}, {0, 0, 2} }; // 현재 할당된 자원
int need[P][R]; // 남은 자원 요구

void calculate_need() {
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < R; j++) {
            need[i][j] = maximum[i][j] - allocation[i][j];
        }
    }
}

bool is_safe() {
    bool finish[P] = {0};
    int work[R];
    for (int i = 0; i < R; i++) {
        work[i] = available[i];
    }

    while (true) {
        bool found = false;
        for (int i = 0; i < P; i++) {
            if (!finish[i]) {
                bool possible = true;
                for (int j = 0; j < R; j++) {
                    if (need[i][j] > work[j]) {
                        possible = false;
                        break;
                    }
                }
                if (possible) {
                    for (int j = 0; j < R; j++) {
                        work[j] += allocation[i][j];
                    }
                    finish[i] = true;
                    found = true;
                }
            }
        }
        if (!found) {
            for (int i = 0; i < P; i++) {
                if (!finish[i]) {
                    return false;
                }
            }
            return true;
        }
    }
}

bool request_resources(int process_id, int request[]) {
    for (int i = 0; i < R; i++) {
        if (request[i] > need[process_id][i] || request[i] > available[i]) {
            return false;
        }
    }

    for (int i = 0; i < R; i++) {
        available[i] -= request[i];
        allocation[process_id][i] += request[i];
        need[process_id][i] -= request[i];
    }

    if (!is_safe()) {
        for (int i = 0; i < R; i++) {


            available[i] += request[i];
            allocation[process_id][i] -= request[i];
            need[process_id][i] += request[i];
        }
        return false;
    }

    return true;
}

int main() {
    calculate_need();

    int process_id = 1;
    int request[R] = {1, 0, 2};

    if (request_resources(process_id, request)) {
        printf("Request granted\n");
    } else {
        printf("Request denied\n");
    }

    return 0;
}
```

**퀴즈 및 해설:**

1. **교착상태의 네 가지 조건은 무엇인가요?**
   - 교착상태의 네 가지 조건은 상호 배제, 점유 대기, 비선점, 순환 대기입니다. 이 조건들이 동시에 충족되면 교착상태가 발생할 수 있습니다.

2. **뮤텍스와 세마포어의 차이점은 무엇인가요?**
   - 뮤텍스는 상호 배제를 위한 객체로, 하나의 스레드만이 특정 자원에 접근할 수 있도록 합니다. 세마포어는 신호와 대기 메커니즘으로, 여러 스레드가 동시에 자원에 접근할 수 있도록 허용하며, 특정 수의 스레드만 자원에 접근할 수 있도록 제한할 수 있습니다.

3. **은행가 알고리즘의 목적은 무엇인가요?**
   - 은행가 알고리즘의 목적은 교착상태를 회피하기 위해 안전 상태를 유지하면서 자원을 할당하는 것입니다. 각 프로세스의 자원 요청을 처리할 때, 시스템이 안전 상태인지 확인하고 자원을 할당합니다.

**해설:**

1. **교착상태의 네 가지 조건**은 상호 배제, 점유 대기, 비선점, 순환 대기입니다. 상호 배제는 자원이 한 번에 하나의 프로세스에 의해 점유될 수 있음을 의미하고, 점유 대기는 프로세스가 자원을 점유한 상태에서 다른 자원을 기다리는 상황을 의미합니다. 비선점은 자원이 프로세스에 의해 점유된 상태에서 다른 프로세스에 의해 선점될 수 없음을 의미하며, 순환 대기는 프로세스 간에 자원 요청이 순환하는 상황을 의미합니다.

2. **뮤텍스와 세마포어의 차이점**은 뮤텍스가 상호 배제를 위한 객체로 하나의 스레드만 자원에 접근할 수 있도록 하는 반면, 세마포어는 신호와 대기 메커니즘으로 여러 스레드가 자원에 접근할 수 있도록 허용하고, 특정 수의 스레드만 자원에 접근할 수 있도록 제한할 수 있다는 점입니다. 뮤텍스는 주로 상호 배제를 보장하기 위해 사용되며, 세마포어는 더 복잡한 동기화 시나리오에서 사용됩니다.

3. **은행가 알고리즘의 목적**은 교착상태를 회피하기 위해 안전 상태를 유지하면서 자원을 할당하는 것입니다. 각 프로세스의 자원 요청을 처리할 때, 시스템이 안전 상태인지 확인하고 자원을 할당합니다. 안전 상태란 모든 프로세스가 교착상태 없이 자원을 할당받아 완료될 수 있는 상태를 의미합니다.

이로써 10주차 강의를 마무리합니다. 다음 주차에는 가상 메모리에 대해 학습합니다. 추가 질문이 있거나 특정 주제에 대해 더 알고 싶다면 언제든지 문의해 주세요.