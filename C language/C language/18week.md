### C 언어 20주차 심화 교육과정 - 18주차: 고급 시스템 프로그래밍

#### 18주차: 고급 시스템 프로그래밍

**강의 목표:**
18주차의 목표는 고급 시스템 프로그래밍 개념을 이해하고, 신호 처리, 프로세스 간 통신(IPC), 파일 시스템 관리 등의 기술을 익히는 것입니다.

**강의 구성:**

##### 1. 신호 (Signal) 처리
- **강의 내용:**
  - 신호의 개념
    - 신호란 무엇인가?
    - 신호의 종류와 역할
  - 신호 처리 함수
    - `signal()`, `sigaction()`
  - 주요 신호와 예제
    - `SIGINT`, `SIGTERM`, `SIGKILL`, `SIGSTOP`
- **실습:**
  - 신호 처리 예제 작성
    ```c
    #include <stdio.h>
    #include <signal.h>
    #include <unistd.h>

    void handle_sigint(int sig) {
        printf("Caught signal %d\n", sig);
    }

    int main() {
        signal(SIGINT, handle_sigint);
        while (1) {
            printf("Running...\n");
            sleep(1);
        }
        return 0;
    }
    ```

##### 2. 프로세스 간 통신 (IPC)
- **강의 내용:**
  - IPC의 개념
    - IPC란 무엇인가?
    - IPC의 필요성과 종류
  - 파이프 (Pipe)
    - `pipe()`, `fork()`, `read()`, `write()`
  - 메시지 큐 (Message Queue)
    - `msgget()`, `msgsnd()`, `msgrcv()`
  - 공유 메모리 (Shared Memory)
    - `shmget()`, `shmat()`, `shmdt()`, `shmctl()`
- **실습:**
  - 메시지 큐를 사용한 프로세스 간 통신 예제 작성
    ```c
    #include <stdio.h>
    #include <stdlib.h>
    #include <sys/ipc.h>
    #include <sys/msg.h>

    struct msg_buffer {
        long msg_type;
        char msg_text[100];
    };

    int main() {
        key_t key;
        int msgid;
        struct msg_buffer message;

        // 키 생성
        key = ftok("progfile", 65);

        // 메시지 큐 생성
        msgid = msgget(key, 0666 | IPC_CREAT);
        message.msg_type = 1;

        printf("Write Data: ");
        fgets(message.msg_text, sizeof(message.msg_text), stdin);

        // 메시지 전송
        msgsnd(msgid, &message, sizeof(message), 0);

        printf("Data sent is : %s \n", message.msg_text);

        return 0;
    }
    ```

  - 공유 메모리를 사용한 프로세스 간 통신 예제 작성
    ```c
    #include <stdio.h>
    #include <stdlib.h>
    #include <sys/ipc.h>
    #include <sys/shm.h>
    #include <string.h>

    int main() {
        key_t key = ftok("shmfile", 65);
        int shmid = shmget(key, 1024, 0666 | IPC_CREAT);
        char *str = (char *) shmat(shmid, (void *) 0, 0);

        printf("Write Data: ");
        fgets(str, 1024, stdin);

        printf("Data written in memory: %s\n", str);

        shmdt(str);

        return 0;
    }
    ```

##### 3. 파일 시스템 관리
- **강의 내용:**
  - 파일 시스템의 개념
    - 파일 시스템이란 무엇인가?
    - 파일 시스템의 구조와 역할
  - 디렉터리 관리
    - `opendir()`, `readdir()`, `closedir()`
  - 파일 속성 관리
    - `stat()`, `chmod()`, `chown()`
- **실습:**
  - 디렉터리 탐색 예제 작성
    ```c
    #include <stdio.h>
    #include <dirent.h>

    int main() {
        struct dirent *de;
        DIR *dr = opendir(".");

        if (dr == NULL) {
            printf("Could not open current directory");
            return 0;
        }

        while ((de = readdir(dr)) != NULL)
            printf("%s\n", de->d_name);

        closedir(dr);
        return 0;
    }
    ```

  - 파일 속성 변경 예제 작성
    ```c
    #include <stdio.h>
    #include <sys/stat.h>

    int main() {
        struct stat st;
        stat("example.txt", &st);

        printf("File size: %ld bytes\n", st.st_size);
        printf("File permissions: %o\n", st.st_mode & 0777);

        chmod("example.txt", 0644);
        printf("Permissions changed to 0644\n");

        chown("example.txt", 1000, 1000);
        printf("Owner changed to UID 1000 and GID 1000\n");

        return 0;
    }
    ```

**과제:**
18주차 과제는 다음과 같습니다.
- 신호 처리 함수를 작성하여, `SIGINT` 신호를 처리하는 프로그램 작성
- 메시지 큐를 사용하여 부모 프로세스와 자식 프로세스 간의 데이터 통신을 구현하는 프로그램 작성
- 공유 메모리를 사용하여 두 프로세스 간의 데이터 통신을 구현하는 프로그램 작성
- 디렉터리를 탐색하여 파일 목록을 출력하는 프로그램 작성
- 파일 속성을 변경하는 프로그램 작성

**퀴즈 및 해설:**

1. **신호란 무엇인가요?**
   - 신호는 운영 체제가 프로세스에 보내는 비동기적 통보로, 특정 이벤트가 발생했음을 알립니다. 예: `SIGINT`는 인터럽트 신호로, Ctrl+C 입력 시 발생합니다.

2. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    struct msg_buffer {
        long msg_type;
        char msg_text[100];
    };
    key_t key = ftok("progfile", 65);
    int msgid = msgget(key, 0666 | IPC_CREAT);
    struct msg_buffer message;
    message.msg_type = 1;
    strcpy(message.msg_text, "Hello, World!");
    msgsnd(msgid, &message, sizeof(message), 0);
    ```
   - 출력 결과는 `Data sent is : Hello, World!`입니다. 메시지 큐를 통해 "Hello, World!" 메시지가 전송됩니다.

3. **파이프와 메시지 큐의 차이점은 무엇인가요?**
   - 파이프는 단방향 통신을 위한 간단한 IPC 메커니즘으로, 부모와 자식 프로세스 간의 데이터 통신에 주로 사용됩니다. 메시지 큐는 양방향 통신을 지원하며, 메시지를 큐에 넣고 꺼내는 방식으로 동작합니다. 더 복잡한 통신 시나리오에 유용합니다.

4. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    DIR *dr = opendir(".");
    struct dirent *de;
    while ((de = readdir(dr)) != NULL)
        printf("%s\n", de->d_name);
    closedir(dr);
    ```
   - 출력 결과는 현재 디렉터리의 파일 및 디렉터리 목록입니다. `opendir` 함수로 디렉터리를 열고, `readdir` 함수로 항목을 읽어와 출력합니다.

5. **`mmap()` 함수의 역할은 무엇인가요?**
   - `mmap()` 함수는 파일이나 디바이스를 메모리에 매핑하여, 메모리를 통해 파일이나 디바이스에 접근할 수 있게 합니다. 이를 통해 파일의 내용을 메모리처럼 다룰 수 있습니다.

**해설:**
1. 신호는 운영 체제가 프로세스에 보내는 비동기적 통보로, 특정 이벤트가 발생했음을 알립니다. 예: `SIGINT`는 인터럽트 신호로, Ctrl+C 입력 시 발생합니다.
2. 메시지 큐를 통해 "Hello, World!" 메시지가 전송됩니다. 따라서 출력 결과는 `Data sent is : Hello, World!`입니다.
3. 파이프는 단방향 통신을 위한 간단한 IPC 메커니즘으로, 부모와 자식 프로세스 간의 데이터 통신에 주로 사용됩니다. 메시지 큐는 양방향 통신을 지원하며, 메시지를 큐에 넣고 꺼내는 방식으로 동작합니다. 더 복잡한 통신 시나리오에 유용합니다.
4. `opendir` 함수로 디렉터리를 열고, `readdir` 함수로 항목을 읽어와 출력하므로, 출력 결과는 현재 디렉터리의 파일 및 디렉터리 목록입니다.
5. `mmap()` 함수는 파일이나 디바이스를 메모리에 매핑하여, 메모리를 통해 파일이나 디바이스에 접근할 수 있게 합니다. 이를 통해 파일의 내용을 메모리

처럼 다룰 수 있습니다.

이 18주차 강의는 학생들이 고급 시스템 프로그래밍 개념을 이해하고, 신호 처리, 프로세스 간 통신(IPC), 파일 시스템 관리 등의 기술을 익히는 능력을 기를 수 있도록 도와줍니다.