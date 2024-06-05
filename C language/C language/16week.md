### C 언어 20주차 심화 교육과정 - 16주차: 멀티쓰레딩

#### 16주차: 멀티쓰레딩

**강의 목표:**
16주차의 목표는 멀티쓰레딩의 개념과 중요성을 이해하고, pthread 라이브러리를 사용하여 C 언어에서 멀티쓰레딩을 구현하는 방법을 배우는 것입니다. 이를 통해 프로그램의 성능을 향상시키고 동시성 처리를 구현하는 능력을 기릅니다.

**강의 구성:**

##### 1. 멀티쓰레딩 개념
- **강의 내용:**
  - 멀티쓰레딩의 개념과 필요성
    - 멀티쓰레딩의 정의
    - 멀티쓰레딩의 장점: 성능 향상, 응답성 개선
  - 프로세스와 쓰레드의 차이점
    - 프로세스와 쓰레드의 정의
    - 프로세스와 쓰레드의 비교: 메모리 구조, 자원 공유
  - 쓰레드의 생명주기
    - 생성, 실행, 대기, 종료

##### 2. pthread 라이브러리
- **강의 내용:**
  - pthread 라이브러리 소개
    - pthread 라이브러리의 기능과 주요 함수
  - 쓰레드 생성 및 종료
    - `pthread_create()`, `pthread_exit()`, `pthread_join()` 함수 사용법
  - 쓰레드 동기화
    - 뮤텍스(Mutex)와 세마포어(Semaphore)의 개념
    - `pthread_mutex_t`, `pthread_mutex_lock()`, `pthread_mutex_unlock()` 함수 사용법
- **실습:**
  - 간단한 쓰레드 생성 및 종료 예제 작성
    ```c
    #include <stdio.h>
    #include <stdlib.h>
    #include <pthread.h>

    void* printHello(void* threadid) {
        long tid;
        tid = (long)threadid;
        printf("Hello World! Thread ID: %ld\n", tid);
        pthread_exit(NULL);
    }

    int main() {
        pthread_t threads[5];
        int rc;
        long t;
        for(t = 0; t < 5; t++) {
            printf("In main: creating thread %ld\n", t);
            rc = pthread_create(&threads[t], NULL, printHello, (void *)t);
            if (rc) {
                printf("ERROR; return code from pthread_create() is %d\n", rc);
                exit(-1);
            }
        }
        for(t = 0; t < 5; t++) {
            pthread_join(threads[t], NULL);
        }
        pthread_exit(NULL);
    }
    ```

  - 뮤텍스를 사용한 쓰레드 동기화 예제 작성
    ```c
    #include <stdio.h>
    #include <stdlib.h>
    #include <pthread.h>

    #define NUM_THREADS 5
    int counter = 0;
    pthread_mutex_t counter_mutex;

    void* incrementCounter(void* threadid) {
        long tid;
        tid = (long)threadid;
        pthread_mutex_lock(&counter_mutex);
        counter++;
        printf("Thread %ld, Counter value: %d\n", tid, counter);
        pthread_mutex_unlock(&counter_mutex);
        pthread_exit(NULL);
    }

    int main() {
        pthread_t threads[NUM_THREADS];
        int rc;
        long t;

        pthread_mutex_init(&counter_mutex, NULL);

        for(t = 0; t < NUM_THREADS; t++) {
            printf("In main: creating thread %ld\n", t);
            rc = pthread_create(&threads[t], NULL, incrementCounter, (void *)t);
            if (rc) {
                printf("ERROR; return code from pthread_create() is %d\n", rc);
                exit(-1);
            }
        }

        for(t = 0; t < NUM_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }

        pthread_mutex_destroy(&counter_mutex);
        pthread_exit(NULL);
    }
    ```

##### 3. 쓰레드 동기화 (세마포어)
- **강의 내용:**
  - 세마포어의 개념
    - 세마포어란 무엇인가?
    - 세마포어를 사용하는 이유
  - 세마포어 사용법
    - `sem_init()`, `sem_wait()`, `sem_post()`, `sem_destroy()` 함수 사용법
- **실습:**
  - 세마포어를 사용한 쓰레드 동기화 예제 작성
    ```c
    #include <stdio.h>
    #include <stdlib.h>
    #include <pthread.h>
    #include <semaphore.h>

    #define NUM_THREADS 5
    int counter = 0;
    sem_t counter_sem;

    void* incrementCounter(void* threadid) {
        long tid;
        tid = (long)threadid;
        sem_wait(&counter_sem);
        counter++;
        printf("Thread %ld, Counter value: %d\n", tid, counter);
        sem_post(&counter_sem);
        pthread_exit(NULL);
    }

    int main() {
        pthread_t threads[NUM_THREADS];
        int rc;
        long t;

        sem_init(&counter_sem, 0, 1);

        for(t = 0; t < NUM_THREADS; t++) {
            printf("In main: creating thread %ld\n", t);
            rc = pthread_create(&threads[t], NULL, incrementCounter, (void *)t);
            if (rc) {
                printf("ERROR; return code from pthread_create() is %d\n", rc);
                exit(-1);
            }
        }

        for(t = 0; t < NUM_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }

        sem_destroy(&counter_sem);
        pthread_exit(NULL);
    }
    ```

**과제:**
16주차 과제는 다음과 같습니다.
- pthread 라이브러리를 사용하여 두 개의 쓰레드를 생성하고, 각 쓰레드가 서로 다른 작업을 수행하는 프로그램 작성
- 뮤텍스를 사용하여 공유 자원을 안전하게 접근하는 프로그램 작성
- 세마포어를 사용하여 여러 쓰레드가 동시에 접근하는 자원을 동기화하는 프로그램 작성

**퀴즈 및 해설:**

1. **쓰레드와 프로세스의 차이점은 무엇인가요?**
   - 프로세스는 독립된 메모리 공간을 가지며, 운영 체제에서 독립적으로 실행되는 프로그램입니다. 쓰레드는 프로세스 내에서 실행되는 작은 단위로, 프로세스의 메모리 공간을 공유합니다.

2. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    pthread_t thread;
    pthread_create(&thread, NULL, printHello, (void *)1);
    pthread_join(thread, NULL);
    ```
   - 출력 결과는 `Hello World! Thread ID: 1`입니다. `pthread_create` 함수로 생성된 쓰레드는 `printHello` 함수를 실행하고, `pthread_join` 함수는 쓰레드가 종료될 때까지 대기합니다.

3. **뮤텍스와 세마포어의 차이점은 무엇인가요?**
   - 뮤텍스는 단일 쓰레드가 자원을 안전하게 접근할 수 있도록 하는 동기화 메커니즘입니다. 세마포어는 카운터를 사용하여 여러 쓰레드가 동시에 자원을 접근할 수 있도록 하는 동기화 메커니즘입니다. 뮤텍스는 보통 1개의 키를 가지고, 세마포어는 N개의 키를 가집니다.

4. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    pthread_mutex_lock(&counter_mutex);
    counter++;
    pthread_mutex_unlock(&counter_mutex);
    ```
   - 출력 결과는 특정 값이 아닌, 동작 결과입니다. 이 코드는 뮤텍스를 사용하여 `counter` 변수에 대한 접근을 동기화합니다. `pthread_mutex_lock` 함수는 뮤텍스를 잠그고, `pthread_mutex_unlock` 함수는 뮤텍스를 해제합니다. 이를 통해 `counter` 변수의 값을 안전하게 증가시킬 수 있습니다.

5. **세마포어를 초기화하는 함수는 무엇인가요?**
   - 세마포어를 초기화하는 함수는 `sem_init`입니다. 이 함수는 세마포어를 초기화하고, 초기 카운터 값을 설정합니다.

**해설:**
1. 프로세스는 독립된 메모리 공간을 가지며, 운영 체제에서 독립적으로 실행되는 프로그램입니다. 쓰레드는 프로세스 내에서 실행되는 작은 단위로, 프로세스의 메모리 공간을 공유합니다.
2. `pthread_create` 함수로 생성된 쓰레드는 `printHello` 함수를 실행하고, `pthread_join` 함수는 쓰레드가 종료될 때까지 대기합니다. 출력 결과는 `Hello World! Thread ID: 1`입니다.
3. 뮤텍스는 단일 쓰레드가 자원을 안전하게 접근할 수 있도록 하는 동기화 메커니즘이며, 세마포어는 카운터를 사용하여 여러 쓰레드가 동시에 자원을 접근할 수 있도록 하는 동기화 메커니즘입니다. 뮤텍스는 보통 1개의 키를

 가지고, 세마포어는 N개의 키를 가집니다.
4. 이 코드는 뮤텍스를 사용하여 `counter` 변수에 대한 접근을 동기화합니다. `pthread_mutex_lock` 함수는 뮤텍스를 잠그고, `pthread_mutex_unlock` 함수는 뮤텍스를 해제합니다. 이를 통해 `counter` 변수의 값을 안전하게 증가시킬 수 있습니다.
5. 세마포어를 초기화하는 함수는 `sem_init`입니다. 이 함수는 세마포어를 초기화하고, 초기 카운터 값을 설정합니다.

이 16주차 강의는 학생들이 멀티쓰레딩의 개념과 중요성을 이해하고, pthread 라이브러리를 사용하여 C 언어에서 멀티쓰레딩을 구현하는 능력을 기를 수 있도록 도와줍니다.

---

16주차 이후의 교육과정은 현재 준비되어 있지 않습니다. 추가적으로 궁금한 사항이나 다른 주제에 대해 문의사항이 있으시면 언제든지 알려주세요!