### 운영체제와 시스템 프로그래밍 기초

**강의 목표:** 운영체제의 주요 기능과 시스템 프로그래밍의 기본 개념을 이해하고, 시스템 콜을 통해 운영체제의 기능을 활용하는 방법을 학습합니다. 프로세스 관리와 메모리 관리의 기본 개념도 다룹니다.

**강의 내용:**

1. **운영체제의 주요 기능**
   - 운영체제란 무엇인가?
     - 컴퓨터 시스템의 자원을 관리하고, 사용자와 하드웨어 간의 인터페이스 역할을 하는 소프트웨어
   - 운영체제의 주요 기능
     - 프로세스 관리, 메모리 관리, 파일 시스템 관리, 입출력 시스템 관리, 보안 및 보호

2. **시스템 프로그래밍 개요**
   - 시스템 프로그래밍이란 무엇인가?
     - 운영체제의 기능을 활용하여 하드웨어와 직접 상호작용하는 프로그램 작성
   - 시스템 콜 (System Call)
     - 운영체제의 커널 기능을 호출하는 인터페이스
     - 주요 시스템 콜: `fork`, `exec`, `wait`, `exit`, `open`, `read`, `write`, `close`

3. **프로세스 관리**
   - 프로세스란 무엇인가?
     - 실행 중인 프로그램의 인스턴스
   - 프로세스 상태
     - 생성, 준비, 실행, 대기, 종료
   - 프로세스 제어 블록 (PCB)
     - 프로세스의 상태 정보를 저장하는 데이터 구조

4. **메모리 관리**
   - 메모리 관리란 무엇인가?
     - 프로세스가 메모리를 효율적으로 사용할 수 있도록 관리
   - 메모리 할당 기법
     - 고정 분할, 가변 분할
   - 가상 메모리
     - 페이징과 세그먼테이션

**실습:**

1. **프로세스 생성 및 관리**
   - `fork`, `exec`, `wait`, `exit` 시스템 콜을 사용하여 프로세스를 생성하고 관리하는 프로그램 작성

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();

    if (pid == 0) {
        // 자식 프로세스
        printf("Child Process: PID = %d\n", getpid());
        execlp("/bin/ls", "ls", NULL);
    } else if (pid > 0) {
        // 부모 프로세스
        printf("Parent Process: PID = %d, Child PID = %d\n", getpid(), pid);
        wait(NULL);  // 자식 프로세스가 종료될 때까지 대기
        printf("Child Complete\n");
    } else {
        // fork 실패
        perror("fork");
    }

    return 0;
}
```

2. **파일 입출력**
   - `open`, `read`, `write`, `close` 시스템 콜을 사용하여 파일을 처리하는 프로그램 작성

```c
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

#define BUFFER_SIZE 1024

int main() {
    int fd;
    char buffer[BUFFER_SIZE];

    // 파일 열기
    fd = open("testfile.txt", O_RDONLY);
    if (fd == -1) {
        perror("open");
        return 1;
    }

    // 파일 읽기
    ssize_t bytes_read = read(fd, buffer, sizeof(buffer) - 1);
    if (bytes_read == -1) {
        perror("read");
        close(fd);
        return 1;
    }
    buffer[bytes_read] = '\0';

    // 파일 내용 출력
    printf("File content:\n%s\n", buffer);

    // 파일 닫기
    close(fd);
    return 0;
}
```

**과제:**

1. **프로세스 제어 블록 (PCB) 구현**
   - 간단한 PCB 구조를 구현하고, 프로세스의 상태 정보를 저장하고 관리하는 프로그램 작성

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

typedef enum { NEW, READY, RUNNING, WAITING, TERMINATED } ProcessState;

typedef struct {
    int pid;
    ProcessState state;
    char *name;
} ProcessControlBlock;

void print_pcb(ProcessControlBlock *pcb) {
    printf("Process ID: %d\n", pcb->pid);
    printf("Process Name: %s\n", pcb->name);
    printf("Process State: ");
    switch (pcb->state) {
        case NEW: printf("NEW\n"); break;
        case READY: printf("READY\n"); break;
        case RUNNING: printf("RUNNING\n"); break;
        case WAITING: printf("WAITING\n"); break;
        case TERMINATED: printf("TERMINATED\n"); break;
    }
}

int main() {
    ProcessControlBlock pcb = {1, NEW, "TestProcess"};

    print_pcb(&pcb);

    pcb.state = READY;
    print_pcb(&pcb);

    pcb.state = RUNNING;
    print_pcb(&pcb);

    pcb.state = TERMINATED;
    print_pcb(&pcb);

    return 0;
}
```

2. **가상 메모리 시뮬레이션**
   - 페이징 기법을 사용하여 가상 주소를 물리 주소로 변환하는 간단한 시뮬레이션 프로그램 작성

```c
#include <stdio.h>
#include <stdlib.h>

#define PAGE_SIZE 4096
#define NUM_PAGES 10

typedef struct {
    int frame_number;
    int valid;
} PageTableEntry;

PageTableEntry page_table[NUM_PAGES];

void initialize_table() {
    for (int i = 0; i < NUM_PAGES; i++) {
        page_table[i].frame_number = -1;
        page_table[i].valid = 0;
    }
}

int translate_address(int virtual_address) {
    int page_number = virtual_address / PAGE_SIZE;
    int offset = virtual_address % PAGE_SIZE;

    if (page_table[page_number].valid) {
        return page_table[page_number].frame_number * PAGE_SIZE + offset;
    } else {
        printf("Page fault: Page %d is not in memory\n", page_number);
        return -1;
    }
}

int main() {
    initialize_table();
    page_table[2].frame_number = 5;
    page_table[2].valid = 1;

    int virtual_address = 8192;  // 2 * PAGE_SIZE
    int physical_address = translate_address(virtual_address);
    if (physical_address != -1) {
        printf("Virtual address %d is translated to physical address %d\n", virtual_address, physical_address);
    }

    return 0;
}
```

**퀴즈 및 해설:**

1. **운영체제의 주요 기능은 무엇인가요?**
   - 운영체제의 주요 기능은 프로세스 관리, 메모리 관리, 파일 시스템 관리, 입출력 시스템 관리, 보안 및 보호입니다.

2. **시스템 콜이란 무엇인가요?**
   - 시스템 콜은 운영체제의 커널 기능을 호출하는 인터페이스입니다. 시스템 콜을 통해 응용 프로그램은 운영체제의 기능을 사용할 수 있습니다.

3. **프로세스 제어 블록(PCB)의 역할은 무엇인가요?**
   - 프로세스 제어 블록(PCB)은 프로세스의 상태 정보를 저장하는 데이터 구조입니다. PCB는 프로세스 ID, 프로세스 상태, 레지스터 정보, 메모리 정보 등을 포함하며, 운영체제가 프로세스를 관리하는 데 사용됩니다.

**해설:**

1. **운영체제의 주요 기능**은 프로세스 관리, 메모리 관리, 파일 시스템 관리, 입출력 시스템 관리, 보안 및 보호입니다. 운영체제는 이러한 기능을 통해 시스템 자원을 효율적으로 관리하고 사용자와 응용 프로그램이 시스템을 안전하고 효율적으로 사용할 수 있도록 지원합니다.

2. **시스템 콜**은 운영체제의 커널 기능을 호출하는 인터페이스입니다. 응용 프로그램은 시스템 콜을 통해 운영체제의 기능을 사용할 수 있습니다. 예를 들어, 파일을 열거나, 프로세스를 생성하거나, 메모리를 할당하는 등의 작업을 수행할 수 있습니다.

3. **프로세스 제어 블록(PCB)**은 프로세스의 상태 정보를 저장하는 데이터 구조입니다. PCB는 프로세스의 ID, 상태, 레지스터 값, 메모리 할당 정보 등을 포함합니다. 운영체제는 PCB를 사용하여 프로세스를 생성, 스케줄링, 종료하는 등의 작업을 관리합니다.

이로써 15주차 강의를 마무리합니다. 다음 주차에는 고급 시스템 프로그래밍에 대해 학습합니다. 추가 질문이 있거나 특정 주제에 대해 더 알고 싶다면 언제든지 문의해 주세요.