### C 언어 20주차 심화 교육과정 - 17주차: 운영 체제와 시스템 프로그래밍

#### 17주차: 운영 체제와 시스템 프로그래밍

**강의 목표:**
17주차의 목표는 운영 체제의 기본 개념과 시스템 프로그래밍의 기초를 이해하고, 시스템 콜을 통해 프로세스와 메모리 관리의 기초를 배우는 것입니다.

**강의 구성:**

##### 1. 시스템 콜 (System Call)
- **강의 내용:**
  - 시스템 콜의 개념
    - 시스템 콜이란 무엇인가?
    - 사용자 모드와 커널 모드의 차이
  - 주요 시스템 콜
    - 프로세스 관리: `fork()`, `exec()`, `wait()`
    - 파일 입출력: `open()`, `read()`, `write()`, `close()`
    - 메모리 관리: `mmap()`, `munmap()`
- **실습:**
  - `fork()`와 `exec()`를 사용한 프로세스 생성 예제 작성
    ```c
    #include <stdio.h>
    #include <unistd.h>
    #include <sys/wait.h>

    int main() {
        pid_t pid = fork();

        if (pid == 0) {
            // 자식 프로세스
            execlp("/bin/ls", "ls", NULL);
        } else if (pid > 0) {
            // 부모 프로세스
            wait(NULL);
            printf("자식 프로세스가 종료되었습니다.\n");
        } else {
            // fork 실패
            perror("fork 실패");
            return 1;
        }

        return 0;
    }
    ```

  - 파일 입출력 시스템 콜 예제 작성
    ```c
    #include <stdio.h>
    #include <fcntl.h>
    #include <unistd.h>

    int main() {
        int fd;
        char buffer[100];
        
        // 파일 열기
        fd = open("example.txt", O_RDONLY);
        if (fd < 0) {
            perror("open");
            return 1;
        }

        // 파일 읽기
        ssize_t bytesRead = read(fd, buffer, sizeof(buffer) - 1);
        if (bytesRead < 0) {
            perror("read");
            close(fd);
            return 1;
        }
        buffer[bytesRead] = '\0';
        printf("파일에서 읽은 내용: %s\n", buffer);

        // 파일 닫기
        close(fd);
        return 0;
    }
    ```

##### 2. 프로세스 관리
- **강의 내용:**
  - 프로세스의 개념
    - 프로세스와 스레드의 차이
    - 프로세스 제어 블록 (PCB)
  - 프로세스 생성과 종료
    - `fork()`와 `exec()`를 사용한 프로세스 생성
    - `exit()`와 `wait()`를 사용한 프로세스 종료
  - 프로세스 간 통신 (IPC)
    - 파이프 (Pipe), 메시지 큐 (Message Queue), 공유 메모리 (Shared Memory)
- **실습:**
  - 파이프를 사용한 프로세스 간 통신 예제 작성
    ```c
    #include <stdio.h>
    #include <unistd.h>
    #include <string.h>

    int main() {
        int fd[2];
        pid_t pid;
        char write_msg[] = "Hello from parent";
        char read_msg[100];

        // 파이프 생성
        if (pipe(fd) == -1) {
            perror("pipe");
            return 1;
        }

        pid = fork();

        if (pid == 0) {
            // 자식 프로세스
            close(fd[1]); // 쓰기 끝 닫기
            read(fd[0], read_msg, sizeof(read_msg));
            printf("자식이 읽음: %s\n", read_msg);
            close(fd[0]);
        } else if (pid > 0) {
            // 부모 프로세스
            close(fd[0]); // 읽기 끝 닫기
            write(fd[1], write_msg, strlen(write_msg) + 1);
            close(fd[1]);
            wait(NULL);
        } else {
            perror("fork");
            return 1;
        }

        return 0;
    }
    ```

##### 3. 메모리 관리 기초
- **강의 내용:**
  - 메모리 관리의 개념
    - 물리적 메모리와 가상 메모리
    - 메모리 할당과 해제
  - 메모리 관리 기술
    - 페이징 (Paging)과 세그멘테이션 (Segmentation)
    - 동적 메모리 할당: `malloc()`, `free()`
  - 메모리 매핑
    - `mmap()`와 `munmap()` 함수 사용법
- **실습:**
  - `mmap()`를 사용한 메모리 매핑 예제 작성
    ```c
    #include <stdio.h>
    #include <stdlib.h>
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <unistd.h>

    int main() {
        int fd;
        struct stat sb;
        char *mapped;

        // 파일 열기
        fd = open("example.txt", O_RDONLY);
        if (fd == -1) {
            perror("open");
            exit(EXIT_FAILURE);
        }

        // 파일 정보 얻기
        if (fstat(fd, &sb) == -1) {
            perror("fstat");
            close(fd);
            exit(EXIT_FAILURE);
        }

        // 파일을 메모리에 매핑
        mapped = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) {
            perror("mmap");
            close(fd);
            exit(EXIT_FAILURE);
        }

        // 매핑된 메모리 출력
        printf("파일 내용:\n%.*s\n", (int)sb.st_size, mapped);

        // 메모리 매핑 해제
        if (munmap(mapped, sb.st_size) == -1) {
            perror("munmap");
        }

        close(fd);
        return 0;
    }
    ```

**과제:**
17주차 과제는 다음과 같습니다.
- `fork()`와 `exec()`를 사용하여 자식 프로세스를 생성하고, 자식 프로세스가 다른 프로그램을 실행하는 프로그램 작성
- 파이프를 사용하여 부모 프로세스와 자식 프로세스 간의 데이터 통신을 구현하는 프로그램 작성
- `mmap()`와 `munmap()`를 사용하여 파일을 메모리에 매핑하고, 매핑된 메모리를 출력하는 프로그램 작성

**퀴즈 및 해설:**

1. **시스템 콜이란 무엇인가요?**
   - 시스템 콜은 운영 체제 커널에서 제공하는 함수로, 사용자 모드에서 커널 모드로 전환하여 시스템 자원을 사용할 수 있게 합니다. 예: 파일 입출력, 프로세스 관리.

2. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    pid_t pid = fork();
    if (pid == 0) {
        printf("Child process\n");
    } else if (pid > 0) {
        printf("Parent process\n");
    }
    ```
   - 출력 결과는 `Parent process`와 `Child process`입니다. `fork()`는 새로운 프로세스를 생성하며, 부모 프로세스와 자식 프로세스에서 각각 다른 메시지를 출력합니다.

3. **파이프를 사용하는 이유는 무엇인가요?**
   - 파이프는 프로세스 간 통신(IPC)을 위한 메커니즘으로, 하나의 프로세스가 데이터를 쓰고 다른 프로세스가 데이터를 읽을 수 있게 합니다. 이를 통해 프로세스 간에 데이터를 교환할 수 있습니다.

4. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    int fd = open("example.txt", O_RDONLY);
    char buffer[10];
    read(fd, buffer, 10);
    printf("%s\n", buffer);
    close(fd);
    ```
   - 출력 결과는 `example.txt` 파일의 처음 10바이트입니다. `open()` 함수로 파일을 열고, `read()` 함수로 데이터를 읽어와 `buffer`에 저장한 후 출력합니다.

5. **`mmap()` 함수의 역할은 무엇인가요?**
   - `mmap()` 함수는 파일이나 디바이스를 메모리에 매핑하여, 메모리를 통해 파일이나 디바이스에 접근할 수 있게 합니다. 이를 통해 파일의 내용을 메모리처럼 다룰 수 있습니다.

**해설:**
1. 시스템 콜은 운영 체제 커널에서 제공하는 함수로, 사용자 모드에서 커널 모드로 전환하여 시스템 자원을 사용할 수 있게 합니다. 예: 파일 입출력, 프로세스 관리.
2. `fork()`는 새로운 프로세스를 생성하며, 부모 프로세스와 자식 프로세스에서 각각 다른 메시지를 출력합니다. 따라서 출력 결과는 `Parent process`와 `Child process`입니다.
3. 파이프는 프로세스 간 통신(IPC)을 위한

 메커니즘으로, 하나의 프로세스가 데이터를 쓰고 다른 프로세스가 데이터를 읽을 수 있게 합니다. 이를 통해 프로세스 간에 데이터를 교환할 수 있습니다.
4. `open()` 함수로 파일을 열고, `read()` 함수로 데이터를 읽어와 `buffer`에 저장한 후 출력합니다. 따라서 출력 결과는 `example.txt` 파일의 처음 10바이트입니다.
5. `mmap()` 함수는 파일이나 디바이스를 메모리에 매핑하여, 메모리를 통해 파일이나 디바이스에 접근할 수 있게 합니다. 이를 통해 파일의 내용을 메모리처럼 다룰 수 있습니다.

이 17주차 강의는 학생들이 운영 체제의 기본 개념과 시스템 프로그래밍의 기초를 이해하고, 시스템 콜을 통해 프로세스와 메모리 관리의 기초를 배우는 능력을 기를 수 있도록 도와줍니다.