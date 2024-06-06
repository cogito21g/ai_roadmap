### 20주차: 고급 시스템 프로그래밍 기법

**강의 목표:** 고급 시스템 프로그래밍 기법을 이해하고, 시그널 처리와 프로세스 간 통신(IPC)의 다양한 방법을 학습합니다. 또한, 공유 메모리와 메시지 큐를 사용한 IPC 기법을 실습합니다.

**강의 내용:**

1. **시그널 처리**
   - 시그널이란 무엇인가?
     - 프로세스 간 비동기적 통신 방법
   - 주요 시그널 종류
     - SIGINT, SIGTERM, SIGKILL, SIGALRM 등
   - 시그널 처리기 (Signal Handler)
     - `signal()`, `sigaction()` 함수
   - 시그널 발생 및 처리 예제

2. **프로세스 간 통신 (IPC)**
   - IPC의 필요성
     - 여러 프로세스 간 데이터 교환 및 동기화
   - IPC 기법
     - 파이프 (Pipes)
     - 명명된 파이프 (Named Pipes or FIFOs)
     - 메시지 큐 (Message Queues)
     - 공유 메모리 (Shared Memory)
   - IPC 기법 비교 및 사용 사례

3. **공유 메모리와 메시지 큐**
   - 공유 메모리
     - `shmget()`, `shmat()`, `shmdt()`, `shmctl()` 함수
   - 메시지 큐
     - `msgget()`, `msgsnd()`, `msgrcv()`, `msgctl()` 함수
   - 공유 메모리와 메시지 큐를 사용한 IPC 예제

**실습:**

1. **시그널 처리 예제**

**시그널 처리기 예제:**

```c
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

void handle_sigint(int sig) {
    printf("Caught signal %d (SIGINT)\n", sig);
    exit(0);
}

int main() {
    signal(SIGINT, handle_sigint);
    while (1) {
        printf("Running... Press Ctrl+C to stop.\n");
        sleep(1);
    }
    return 0;
}
```

2. **파이프를 사용한 프로세스 간 통신 예제**

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

3. **공유 메모리를 사용한 프로세스 간 통신 예제**

**공유 메모리 예제:**

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#define SHM_KEY 1234
#define SHM_SIZE 1024

int main() {
    int shmid;
    char *shm_ptr;

    // 공유 메모리 생성
    shmid = shmget(SHM_KEY, SHM_SIZE, 0666|IPC_CREAT);
    if (shmid == -1) {
        perror("shmget");
        exit(EXIT_FAILURE);
    }

    // 공유 메모리 연결
    shm_ptr = (char*) shmat(shmid, NULL, 0);
    if (shm_ptr == (char*) -1) {
        perror("shmat");
        exit(EXIT_FAILURE);
    }

    // 공유 메모리에 데이터 쓰기
    strcpy(shm_ptr, "Hello, Shared Memory!");
    printf("Data written to shared memory: %s\n", shm_ptr);

    // 공유 메모리 분리
    shmdt(shm_ptr);

    // 공유 메모리 제거
    shmctl(shmid, IPC_RMID, NULL);

    return 0;
}
```

4. **메시지 큐를 사용한 프로세스 간 통신 예제**

**메시지 큐 예제:**

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/msg.h>

#define MSG_KEY 5678
#define MSG_SIZE 100

struct message {
    long msg_type;
    char msg_text[MSG_SIZE];
};

void send_message() {
    int msgid;
    struct message msg;

    msgid = msgget(MSG_KEY, 0666 | IPC_CREAT);
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

    msgid = msgget(MSG_KEY, 0666 | IPC_CREAT);
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
        sleep(1); // 부모 프로세스가 먼저 메시지를 보내도록 대기
        receive_message();
    } else {
        // 부모 프로세스
        send_message();
    }
    return 0;
}
```

**과제:**

1. **시그널 처리 및 시그널 마스크 예제 구현**
   - 특정 시그널을 무시하고, 다른 시그널을 처리하는 프로그램 작성
   - `sigprocmask` 함수를 사용하여 시그널 마스크 설정

2. **고급 IPC 기법 구현**
   - 파이프와 메시지 큐를 조합하여 복잡한 데이터 교환 시나리오를 구현
   - 여러 프로세스 간의 동기화와 데이터 교환을 처리

**퀴즈 및 해설:**

1. **시그널의 주요 용도는 무엇인가요?**
   - 시그널은 프로세스 간 비동기적 통신을 위해 사용되며, 특정 이벤트가 발생했을 때 프로세스에 알리는 역할을 합니다.

2. **공유 메모리와 메시지 큐의 차이점은 무엇인가요?**
   - 공유 메모리는 여러 프로세스가 동일한 메모리 공간을 공유하여 데이터를 주고받는 방법이고, 메시지 큐는 메시지 큐 구조를 통해 데이터를 주고받는 방법입니다. 공유 메모리는 빠르지만 동기화 문제가 발생할 수 있고, 메시지 큐는 동기화가 쉬운 반면 오버헤드가 발생할 수 있습니다.

3. **`sigaction` 함수와 `signal` 함수의 차이점은 무엇인가요?**
   - `sigaction` 함수는 시그널 처리기를 설정할 때 더 많은 옵션을 제공하며, 보다 안전하고 유연하게 시그널을 처리할 수 있습니다. `signal` 함수는 간단한 시그널 처리기를 설정할 때 사용됩니다.

**해설:**

1. **시그널의 주요 용도**는 프로세스 간 비동기적 통신을 위해 사용되며, 특정 이벤트가 발생했을 때 프로세스에 알리는 역할을 합니다. 예를 들어, `SIGINT`는 사용자가 Ctrl+C를 눌렀을 때 프로세스에 전달되는 시그널입니다.

2. **공유 메모리와 메시지 큐의 차이점**은 공유 메모리는 여러 프로세스가 동일한 메모리 공간을 공유하여 데이터를 주고받는 방법이고, 메시지 큐는 메시지 큐 구조를 통해 데이터를 주고받는 방법입니다. 공유 메모리는 빠르지만 동기화 문제가 발생할 수 있고, 메시지 큐는 동기화가 쉬운 반면 오버헤드가 발생할 수 있습니다.

3. **`sigaction` 함수와 `signal` 함수의 차이점**은 `sigaction` 함수는 시그널 처리기를 설정할 때 더 많은 옵션을 제공하며, 보다 안전하고 유연하게

 시그널을 처리할 수 있다는 점입니다. `signal` 함수는 간단한 시그널 처리기를 설정할 때 사용되지만, `sigaction` 함수는 시그널 처리기와 관련된 추가 정보를 설정할 수 있어 더 강력한 기능을 제공합니다.

이로써 20주차 강의를 마무리합니다. 전체 과정이 끝났으니 프로젝트를 진행하고 최종 발표를 준비하세요. 추가 질문이 있거나 특정 주제에 대해 더 알고 싶다면 언제든지 문의해 주세요.