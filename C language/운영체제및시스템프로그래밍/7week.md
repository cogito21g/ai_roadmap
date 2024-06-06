### 7주차: 시스템 호출 (System Calls)

**강의 목표:** 시스템 호출의 개념과 역할을 이해하고, 주요 시스템 호출을 사용하여 파일 처리 및 프로세스 관리를 학습합니다. 사용자 모드와 커널 모드 간의 전환을 이해합니다.

**강의 내용:**

1. **시스템 호출의 개념과 역할**
   - 시스템 호출이란 무엇인가?
     - 운영체제 커널의 서비스를 호출하는 인터페이스
   - 시스템 호출의 역할
     - 응용 프로그램이 운영체제의 기능을 사용할 수 있게 함
     - 하드웨어 자원에 대한 안전한 접근 제공

2. **주요 시스템 호출**
   - 파일 시스템 관련 시스템 호출
     - `open`, `read`, `write`, `close`
   - 프로세스 관리 시스템 호출
     - `fork`, `exec`, `wait`, `exit`
   - 기타 유용한 시스템 호출
     - `ioctl`, `getpid`, `getppid`, `kill`

3. **사용자 모드와 커널 모드 간의 전환**
   - 사용자 모드와 커널 모드의 개념
     - 사용자 모드: 응용 프로그램 실행 모드
     - 커널 모드: 운영체제 커널 실행 모드
   - 시스템 호출을 통한 모드 전환
     - 시스템 호출 인터페이스와 컨텍스트 스위칭

**실습:**

1. **파일 처리 시스템 호출 사용 예제**
   - `open`, `read`, `write`, `close` 시스템 호출을 사용하여 파일을 처리하는 프로그램 작성

```c
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

int main() {
    int fd;
    char buffer[100];
    
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

2. **프로세스 관리 시스템 호출 사용 예제**
   - `fork`, `exec`, `wait` 시스템 호출을 사용하여 프로세스를 생성하고 관리하는 프로그램 작성

```c
#include <stdio.h>
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

**과제:**

1. **파일 복사 프로그램 작성**
   - `open`, `read`, `write`, `close` 시스템 호출을 사용하여 파일을 복사하는 프로그램 작성
   - 소스 파일을 읽고, 대상 파일에 데이터를 쓰는 기능 구현

```c
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <source file> <destination file>\n", argv[0]);
        return 1;
    }

    int src_fd = open(argv[1], O_RDONLY);
    if (src_fd == -1) {
        perror("open source file");
        return 1;
    }

    int dest_fd = open(argv[2], O_WRONLY | O_CREAT, 0644);
    if (dest_fd == -1) {
        perror("open destination file");
        close(src_fd);
        return 1;
    }

    char buffer[1024];
    ssize_t bytes_read, bytes_written;

    while ((bytes_read = read(src_fd, buffer, sizeof(buffer))) > 0) {
        bytes_written = write(dest_fd, buffer, bytes_read);
        if (bytes_written != bytes_read) {
            perror("write");
            close(src_fd);
            close(dest_fd);
            return 1;
        }
    }

    if (bytes_read == -1) {
        perror("read");
    }

    close(src_fd);
    close(dest_fd);
    return 0;
}
```

2. **프로세스 트리 생성 프로그램 작성**
   - `fork` 시스템 호출을 사용하여 부모 프로세스가 자식 프로세스를 생성하고, 자식 프로세스가 다시 자식 프로세스를 생성하여 트리 형태의 프로세스를 만듭니다.
   - 각 프로세스가 자신의 PID와 부모 PID를 출력하도록 구현

```c
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

void create_process_tree(int level) {
    if (level > 0) {
        pid_t pid = fork();

        if (pid == 0) {
            // 자식 프로세스
            printf("Child Process: Level = %d, PID = %d, Parent PID = %d\n", level, getpid(), getppid());
            create_process_tree(level - 1);
        } else if (pid > 0) {
            // 부모 프로세스
            wait(NULL);  // 자식 프로세스가 종료될 때까지 대기
        } else {
            // fork 실패
            perror("fork");
        }
    }
}

int main() {
    int levels = 3; // 프로세스 트리의 깊이 설정
    printf("Parent Process: PID = %d\n", getpid());
    create_process_tree(levels);
    return 0;
}
```

**퀴즈 및 해설:**

1. **시스템 호출의 역할은 무엇인가요?**
   - 시스템 호출은 응용 프로그램이 운영체제의 기능을 사용할 수 있게 하는 인터페이스입니다. 이를 통해 응용 프로그램은 파일 열기, 읽기, 쓰기, 프로세스 생성 등의 작업을 안전하게 수행할 수 있습니다.

2. **사용자 모드와 커널 모드의 차이점은 무엇인가요?**
   - 사용자 모드는 응용 프로그램이 실행되는 모드로, 제한된 권한을 가지며 하드웨어 접근이 제한됩니다. 커널 모드는 운영체제가 실행되는 모드로, 모든 권한을 가지며 하드웨어에 직접 접근할 수 있습니다. 시스템 호출을 통해 사용자 모드에서 커널 모드로 전환됩니다.

3. **`fork`와 `exec` 시스템 호출의 차이점은 무엇인가요?**
   - `fork` 시스템 호출은 새로운 프로세스를 생성하며, 부모 프로세스의 복사본을 생성합니다. `exec` 시스템 호출은 현재 프로세스를 새로운 프로그램으로 덮어쓰며, 새로운 프로그램을 실행합니다. `fork`와 `exec`는 종종 함께 사용되어 새로운 프로세스를 생성하고, 해당 프로세스에서 새로운 프로그램을 실행합니다.

**해설:**

1. **시스템 호출의 역할**은 응용 프로그램이 운영체제의 기능을 사용할 수 있게 하는 인터페이스입니다. 시스템 호출을 통해 응용 프로그램은 파일 열기, 읽기, 쓰기, 프로세스 생성 등의 작업을 안전하게 수행할 수 있습니다. 이는 운영체제가 하드웨어 자원에 대한 직접 접근을 제어하고 보호할 수 있도록 합니다.

2. **사용자 모드와 커널 모드**는 운영체제의 두 가지 실행 모드입니다. 사용자 모드는 응용 프로그램이 실행되는 모드로, 제한된 권한을 가지며 하드웨어 접근이 제한됩니다. 커널 모드는 운영체제가 실행되는 모드로, 모든 권한을 가지며 하드웨어에 직접 접근할 수 있습니다. 시스템 호출을 통해 사용자 모드에서 커널 모드로 전환되며, 운영체제의 기능을 사용할 수 있습니다.

3. **`fork`와 `exec` 시스템 호출의 차이점**은 `fork`가 새로운 프로세스를 생성하여 부모 프로세스의 복사본을 만드는 반면, `exec`는 현재 프로세스를 새로운 프로그램으로 덮어쓰며 새로운 프로그램을 실행한다는 점입니다. `fork`와 `exec`는 종종 함께 사용되어 새로운 프로세스를 생성하고, 해당 프로세스에서 새로운 프로그램을 실행하는 데 사용됩니다.

이로써 7주차 강의를 마무리합니다. 다음 주차에는 장치 드라이버 기초에 대해 학습합니다. 추가 질문이 있거나 특정 주제에 대해 더 알고 싶다면 언제든지 문의해 주세요.