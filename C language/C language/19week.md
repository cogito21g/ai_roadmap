### C 언어 20주차 심화 교육과정 - 19주차: 최적화 및 디버깅

#### 19주차: 최적화 및 디버깅

**강의 목표:**
19주차의 목표는 코드의 성능을 최적화하는 기법과 디버깅 기법을 이해하고, 이를 통해 효율적이고 오류 없는 프로그램을 작성하는 능력을 기르는 것입니다.

**강의 구성:**

##### 1. 코드 최적화 기법
- **강의 내용:**
  - 코드 최적화의 필요성
    - 성능 향상, 자원 절약
  - 컴파일러 최적화
    - 컴파일러 최적화 옵션 (`-O1`, `-O2`, `-O3`)
  - 코드 최적화 기법
    - 루프 최적화, 함수 인라인화, 상수 접미어 사용, 불필요한 계산 제거
- **실습:**
  - 컴파일러 최적화 옵션 사용 예제
    ```sh
    gcc -O2 -o optimized_program program.c
    ```

  - 루프 최적화 예제 작성
    ```c
    #include <stdio.h>
    #include <time.h>

    void optimizedLoop() {
        int arr[1000];
        for (int i = 0; i < 1000; i++) {
            arr[i] = i * i;
        }
    }

    void nonOptimizedLoop() {
        int arr[1000];
        for (int i = 0; i < 1000; i++) {
            arr[i] = i * i;
            int temp = 0;
            for (int j = 0; j < 1000; j++) {
                temp += arr[j];
            }
        }
    }

    int main() {
        clock_t start, end;
        double cpu_time_used;

        start = clock();
        optimizedLoop();
        end = clock();
        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Optimized loop time: %f\n", cpu_time_used);

        start = clock();
        nonOptimizedLoop();
        end = clock();
        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Non-optimized loop time: %f\n", cpu_time_used);

        return 0;
    }
    ```

##### 2. 메모리 누수 탐지
- **강의 내용:**
  - 메모리 누수의 개념과 문제점
    - 메모리 누수란?
    - 메모리 누수의 영향과 해결 방법
  - 메모리 누수 탐지 도구
    - `valgrind` 사용법
- **실습:**
  - `valgrind`를 사용한 메모리 누수 탐지 예제
    ```sh
    gcc -g -o memory_leak_program memory_leak_program.c
    valgrind --leak-check=full ./memory_leak_program
    ```

  - 메모리 누수 예제 작성 및 수정
    ```c
    #include <stdlib.h>
    
    void memoryLeak() {
        int *ptr = (int *)malloc(sizeof(int) * 100);
        // 메모리 누수 발생, free(ptr) 호출 필요
    }

    void noMemoryLeak() {
        int *ptr = (int *)malloc(sizeof(int) * 100);
        free(ptr); // 메모리 해제
    }

    int main() {
        memoryLeak();
        noMemoryLeak();
        return 0;
    }
    ```

##### 3. gdb를 사용한 디버깅
- **강의 내용:**
  - `gdb`의 개념과 필요성
    - `gdb`란 무엇인가?
    - 디버깅의 중요성
  - `gdb`의 주요 명령어
    - `break`, `run`, `next`, `step`, `print`, `backtrace`
  - 디버깅 절차
    - 중단점 설정, 프로그램 실행, 변수 값 확인, 호출 스택 확인
- **실습:**
  - `gdb`를 사용한 디버깅 예제
    ```sh
    gcc -g -o debug_program debug_program.c
    gdb ./debug_program
    ```

  - 디버깅 예제 작성 및 디버깅 과정
    ```c
    #include <stdio.h>

    void buggyFunction() {
        int a = 5;
        int b = 0;
        int c = a / b; // 오류 발생
        printf("c: %d\n", c);
    }

    int main() {
        buggyFunction();
        return 0;
    }
    ```

    ```sh
    # gdb 명령어 예시
    (gdb) break buggyFunction
    (gdb) run
    (gdb) next
    (gdb) print a
    (gdb) backtrace
    ```

**과제:**
19주차 과제는 다음과 같습니다.
- 컴파일러 최적화 옵션을 사용하여 프로그램의 성능을 향상시키는 실습
- `valgrind`를 사용하여 메모리 누수를 탐지하고 수정하는 실습
- `gdb`를 사용하여 주어진 프로그램의 버그를 디버깅하는 실습

**퀴즈 및 해설:**

1. **컴파일러 최적화 옵션 중 `-O2`와 `-O3`의 차이점은 무엇인가요?**
   - `-O2`는 중간 수준의 최적화를 수행하며, 실행 속도와 코드 크기의 균형을 맞춥니다. `-O3`는 더 높은 수준의 최적화를 수행하며, 실행 속도를 최대화하지만 코드 크기가 증가할 수 있습니다.

2. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    void optimizedLoop() {
        int arr[1000];
        for (int i = 0; i < 1000; i++) {
            arr[i] = i * i;
        }
    }
    int main() {
        clock_t start, end;
        double cpu_time_used;
        start = clock();
        optimizedLoop();
        end = clock();
        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Optimized loop time: %f\n", cpu_time_used);
        return 0;
    }
    ```
   - 출력 결과는 `Optimized loop time: x.xxxxx` 형식으로, `optimizedLoop` 함수의 실행 시간을 측정하여 출력합니다. 실제 시간은 시스템 성능에 따라 다릅니다.

3. **메모리 누수란 무엇인가요?**
   - 메모리 누수는 동적으로 할당된 메모리를 사용 후 해제하지 않아, 프로그램이 종료될 때까지 메모리가 반환되지 않는 상황을 의미합니다. 메모리 누수는 시스템 자원을 낭비하고 프로그램의 성능을 저하시킵니다.

4. **다음 코드의 문제점은 무엇인가요?**
    ```c
    void memoryLeak() {
        int *ptr = (int *)malloc(sizeof(int) * 100);
        // 메모리 누수 발생, free(ptr) 호출 필요
    }
    ```
   - `memoryLeak` 함수에서 `malloc`을 통해 할당된 메모리가 `free`를 통해 해제되지 않아 메모리 누수가 발생합니다. `free(ptr)` 호출이 필요합니다.

5. **`gdb`에서 `break` 명령어의 역할은 무엇인가요?**
   - `break` 명령어는 중단점을 설정하여 프로그램 실행을 특정 지점에서 멈추게 합니다. 이를 통해 해당 지점에서 변수 값, 프로그램 상태 등을 확인할 수 있습니다.

**해설:**
1. `-O2`는 중간 수준의 최적화를 수행하며, 실행 속도와 코드 크기의 균형을 맞춥니다. `-O3`는 더 높은 수준의 최적화를 수행하며, 실행 속도를 최대화하지만 코드 크기가 증가할 수 있습니다.
2. 출력 결과는 `Optimized loop time: x.xxxxx` 형식으로, `optimizedLoop` 함수의 실행 시간을 측정하여 출력합니다. 실제 시간은 시스템 성능에 따라 다릅니다.
3. 메모리 누수는 동적으로 할당된 메모리를 사용 후 해제하지 않아, 프로그램이 종료될 때까지 메모리가 반환되지 않는 상황을 의미합니다. 메모리 누수는 시스템 자원을 낭비하고 프로그램의 성능을 저하시킵니다.
4. `memoryLeak` 함수에서 `malloc`을 통해 할당된 메모리가 `free`를 통해 해제되지 않아 메모리 누수가 발생합니다. `free(ptr)` 호출이 필요합니다.
5. `break` 명령어는 중단점을 설정하여 프로그램 실행을 특정 지점에서 멈추게 합니다. 이를 통해 해당 지점에서 변수 값, 프로그램 상태 등을 확인할 수 있습니다.

이 19주차 강의는 학생들이 코드의 성능을 최적화하는 기법과 디버깅 기법을 이해하고, 이를 통해 효율적이고 오류 없는 프로그램을 작성하는 능력을 기를 수 있도록 도와줍니다.