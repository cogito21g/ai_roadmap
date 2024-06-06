### 고급 시스템 프로그래밍

**강의 목표:** 고급 시스템 프로그래밍의 주요 개념과 기술을 학습하고, 신호 처리, IPC(Inter-Process Communication), 파일 시스템 인터페이스 등의 고급 주제를 다룹니다.

**강의 내용:**

1. **신호 처리 (Signal Handling)**
   - 신호란 무엇인가?
     - 프로세스 간에 비동기적으로 메시지를 전달하는 메커니즘
   - 주요 신호 종류
     - SIGINT, SIGTERM, SIGKILL, SIGSEGV 등
   - 신호 처리기 (Signal Handler)
     - `signal()`, `sigaction()` 함수
     - 신호 처리기 등록 및 사용

2. **프로세스 간 통신 (IPC)**
   - IPC의 필요성
     - 여러 프로세스 간의 데이터 교환
   - IPC 기법
     - 파이프 (Pipes)
     - 명명된 파이프 (Named Pipes or FIFOs)
     - 메시지 큐 (Message Queues)
     - 공유 메모리 (Shared Memory)
     - 소켓 (Sockets)

3. **파일 시스템 인터페이스**
   - 파일 시스템의 구조
     - 파일, 디렉토리, inode
   - 파일 시스템 인터페이스 함수
     - `open()`, `read()`, `write()`, `close()`
     - `opendir()`, `readdir()`, `closedir()`
   - 고급 파일 시스템 함수
     - `stat()`, `fstat()`, `lstat()`
     - 파일 권한 변경 (`chmod`, `fchmod`)
     - 파일 소유권 변경 (`chown`, `fchown`)

**실습:**

1. **신호 처리기 예제**

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>

void handle_signal(int signal) {
    if (signal == SIGINT) {
        printf("Received SIGINT (Ctrl+C). Exiting...\n");
        exit(0);
    }
}

int main() {
    signal(SIGINT, handle_signal);
    while (1) {
        printf("Running... Press Ctrl+C to exit.\n");
        sleep(1);
    }
    return 0;
}
```

2. **파이프를 이용한 프로세스 간 통신**

**파이프 예제:**

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define BUFFER_SIZE 1024

int main() {
    int pipefd[2];
    pid_t pid;
    char buffer[BUFFER_SIZE];

    if (pipe(pipefd) == -1) {
        perror("pipe");
        exit(EXIT_FAILURE);
    }

    pid = fork();
    if (pid == -1) {
        perror("fork");
        exit(EXIT_FAILURE);
    }

    if (pid == 0) {
        // 자식 프로세스
        close(pipefd[1]); // 쓰기 종단 닫기
        read(pipefd[0], buffer, BUFFER_SIZE);
        printf("Child Process received: %s\n", buffer);
        close(pipefd[0]);
    } else {
        // 부모 프로세스
        close(pipefd[0]); // 읽기 종단 닫기
        write(pipefd[1], "Hello from parent", 18);
        close(pipefd[1]);
        wait(NULL); // 자식 프로세스 종료 대기
    }

    return 0;
}
```

3. **파일 시스템 인터페이스 예제**

```c
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

void print_file_info(const char *filename) {
    struct stat file_stat;

    if (stat(filename, &file_stat) == -1) {
        perror("stat");
        exit(EXIT_FAILURE);
    }

    printf("File: %s\n", filename);
    printf("Size: %ld bytes\n", file_stat.st_size);
    printf("Permissions: %o\n", file_stat.st_mode & 0777);
    printf("Last accessed: %s", ctime(&file_stat.st_atime));
}

int main() {
    int fd;
    const char *filename = "example.txt";
    char buffer[100] = "Hello, World!\n";

    fd = open(filename, O_WRONLY | O_CREAT, 0644);
    if (fd == -1) {
        perror("open");
        exit(EXIT_FAILURE);
    }

    if (write(fd, buffer, sizeof(buffer)) == -1) {
        perror("write");
        close(fd);
        exit(EXIT_FAILURE);
    }

    close(fd);
    print_file_info(filename);

    return 0;
}
```

**과제:**

1. **메시지 큐를 사용한 프로세스 간 통신**
   - 메시지 큐를 사용하여 두 프로세스 간의 메시지 전송을 구현합니다.
   - 프로세스1은 메시지를 보내고, 프로세스2는 메시지를 수신합니다.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/msg.h>

#define KEY 1234
#define MSG_SIZE 100

struct message {
    long msg_type;
    char msg_text[MSG_SIZE];
};

void send_message() {
    int msgid;
    struct message msg;

    msgid = msgget(KEY, 0666 | IPC_CREAT);
    if (msgid == -1) {
        perror("msgget");
        exit(EXIT_FAILURE);
    }

    msg.msg_type = 1;
    strcpy(msg.msg_text, "Hello from Process 1");
    if (msgsnd(msgid, &msg, sizeof(msg), 0) == -1) {
        perror("msgsnd");
        exit(EXIT_FAILURE);
    }

    printf("Message sent: %s\n", msg.msg_text);
}

void receive_message() {
    int msgid;
    struct message msg;

    msgid = msgget(KEY, 0666 | IPC_CREAT);
    if (msgid == -1) {
        perror("msgget");
        exit(EXIT_FAILURE);
    }

    if (msgrcv(msgid, &msg, sizeof(msg), 1, 0) == -1) {
        perror("msgrcv");
        exit(EXIT_FAILURE);
    }

    printf("Message received: %s\n", msg.msg_text);
    msgctl(msgid, IPC_RMID, NULL); // 메시지 큐 제거
}

int main() {
    if (fork() == 0) {
        // 자식 프로세스
        receive_message();
    } else {
        // 부모 프로세스
        send_message();
    }
    return 0;
}
```

2. **공유 메모리를 사용한 프로세스 간 통신**
   - 공유 메모리를 사용하여 두 프로세스 간의 데이터를 공유합니다.
   - 프로세스1은 데이터를 쓰고, 프로세스2는 데이터를 읽습니다.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#define SHM_KEY 5678
#define SHM_SIZE 1024

void write_shared_memory() {
    int shmid;
    char *shm_ptr;

    shmid = shmget(SHM_KEY, SHM_SIZE, 0666 | IPC_CREAT);
    if (shmid == -1) {
        perror("shmget");
        exit(EXIT_FAILURE);
    }

    shm_ptr = (char *)shmat(shmid, NULL, 0);
    if (shm_ptr == (char *)(-1)) {
        perror("shmat");
        exit(EXIT_FAILURE);
    }

    strcpy(shm_ptr, "Hello from Process 1");
    printf("Data written to shared memory: %s\n", shm_ptr);
    shmdt(shm_ptr);
}

void read_shared_memory() {
    int shmid;
    char *shm_ptr;

    shmid = shmget(SHM_KEY, SHM_SIZE, 0666);
    if (shmid == -1) {
        perror("shmget");
        exit(EXIT_FAILURE);
    }

    shm_ptr = (char *)shmat(shmid, NULL, 0);
    if (shm_ptr == (char *)(-1)) {
        perror("shmat");
        exit(EXIT_FAILURE);
    }

    printf("Data read from shared memory: %s\n", shm_ptr);
    shmdt(shm_ptr);
    shmctl(shmid, IPC_RMID, NULL); // 공유 메모리 제거
}

int main() {
    if (fork() == 0) {
        // 자식 프로세스
        sleep(1); // 부모 프로세스가 먼저 쓰도록 대기
        read_shared_memory();
    } else {
        // 부모 프로세스
        write_shared_memory();
    }
    return 0;
}
```

**퀴즈 및 해설:**

1. **신호와 신호 처리기의 역할은 무엇인가요?**
   - 신호는 프로세스 간에 비동기적으로 메시지를 전달하는 메커니즘입니다. 신호 처리기는 특정 신호가 발생했을 때 실행되는 함수로, 신호를 처리하고 적절한 동작을 수행합니다.

2. **IPC의 필요성과 주요 기법은 무엇인가요?**
   - IPC는 여러 프로세스 간의 데이터 교환을 위해 필요합니다. 주요 IPC 기법

으로는 파이프, 명명된 파이프, 메시지 큐, 공유 메모리, 소켓 등이 있습니다.

3. **파일 시스템 인터페이스의 주요 함수는 무엇인가요?**
   - 주요 함수로는 파일을 열기(`open`), 읽기(`read`), 쓰기(`write`), 닫기(`close`), 디렉토리 열기(`opendir`), 읽기(`readdir`), 닫기(`closedir`), 파일 상태 정보 가져오기(`stat`, `fstat`, `lstat`) 등이 있습니다.

**해설:**

1. **신호와 신호 처리기**는 프로세스 간에 비동기적으로 메시지를 전달하는 메커니즘입니다. 예를 들어, `SIGINT`는 Ctrl+C를 눌렀을 때 발생하는 신호이며, 이를 처리하기 위해 신호 처리기를 등록할 수 있습니다. 신호 처리기는 신호가 발생했을 때 실행되는 함수로, 신호를 처리하고 적절한 동작을 수행합니다.

2. **IPC(Inter-Process Communication)**는 여러 프로세스 간의 데이터 교환을 위해 필요합니다. IPC는 프로세스들이 데이터를 공유하고, 협력 작업을 수행할 수 있도록 합니다. 주요 IPC 기법으로는 파이프, 명명된 파이프, 메시지 큐, 공유 메모리, 소켓 등이 있습니다. 각 기법은 특정 상황에 적합한 데이터 교환 방식을 제공합니다.

3. **파일 시스템 인터페이스**는 파일 및 디렉토리를 조작하는 데 사용되는 함수들의 집합입니다. 주요 함수로는 파일을 열기(`open`), 읽기(`read`), 쓰기(`write`), 닫기(`close`), 디렉토리 열기(`opendir`), 읽기(`readdir`), 닫기(`closedir`), 파일 상태 정보 가져오기(`stat`, `fstat`, `lstat`) 등이 있습니다. 이러한 함수들은 파일 시스템과 상호작용하여 파일과 디렉토리를 관리하는 데 사용됩니다.

이로써 16주차 강의를 마무리합니다. 다음 주차에는 코드 최적화 및 디버깅에 대해 학습합니다. 추가 질문이 있거나 특정 주제에 대해 더 알고 싶다면 언제든지 문의해 주세요.