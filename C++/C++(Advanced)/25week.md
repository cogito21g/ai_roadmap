### 25주차: 시스템 프로그래밍

#### 강의 목표
- 파일 시스템 접근 및 관리 이해 및 사용
- 프로세스 제어 이해 및 사용
- 고급 I/O 관리 이해 및 사용

#### 강의 내용

##### 1. 파일 시스템 접근 및 관리
- **파일 읽기 및 쓰기**

```cpp
#include <iostream>
#include <fstream>
using namespace std;

int main() {
    // 파일 쓰기
    ofstream outFile("example.txt");
    if (outFile.is_open()) {
        outFile << "Hello, World!" << endl;
        outFile.close();
    } else {
        cout << "Unable to open file for writing" << endl;
    }

    // 파일 읽기
    ifstream inFile("example.txt");
    if (inFile.is_open()) {
        string line;
        while (getline(inFile, line)) {
            cout << line << endl;
        }
        inFile.close();
    } else {
        cout << "Unable to open file for reading" << endl;
    }

    return 0;
}
```

- **파일 상태 확인**

```cpp
#include <iostream>
#include <sys/stat.h>
using namespace std;

int main() {
    struct stat fileInfo;

    if (stat("example.txt", &fileInfo) == 0) {
        cout << "File Size: " << fileInfo.st_size << " bytes" << endl;
        cout << "Permissions: " << ((fileInfo.st_mode & S_IRUSR) ? "r" : "-")
             << ((fileInfo.st_mode & S_IWUSR) ? "w" : "-")
             << ((fileInfo.st_mode & S_IXUSR) ? "x" : "-") << endl;
        cout << "Last Accessed: " << ctime(&fileInfo.st_atime);
    } else {
        cout << "Unable to get file info" << endl;
    }

    return 0;
}
```

##### 2. 프로세스 제어
- **프로세스 생성**

```cpp
#include <iostream>
#include <unistd.h>
using namespace std;

int main() {
    pid_t pid = fork();

    if (pid < 0) {
        cout << "Fork failed" << endl;
    } else if (pid == 0) {
        cout << "Child process" << endl;
        execlp("/bin/ls", "ls", NULL);
    } else {
        cout << "Parent process" << endl;
        wait(NULL);
    }

    return 0;
}
```

- **프로세스 간 통신 (IPC)**

```cpp
#include <iostream>
#include <sys/ipc.h>
#include <sys/msg.h>
using namespace std;

struct message_buffer {
    long message_type;
    char message_text[100];
};

int main() {
    key_t key = ftok("progfile", 65);
    int msgid = msgget(key, 0666 | IPC_CREAT);
    message_buffer message;

    // 메시지 송신
    message.message_type = 1;
    strcpy(message.message_text, "Hello, World!");
    msgsnd(msgid, &message, sizeof(message), 0);
    cout << "Message sent: " << message.message_text << endl;

    // 메시지 수신
    msgrcv(msgid, &message, sizeof(message), 1, 0);
    cout << "Message received: " << message.message_text << endl;

    msgctl(msgid, IPC_RMID, NULL);

    return 0;
}
```

##### 3. 고급 I/O 관리
- **비동기 I/O (aio)**

```cpp
#include <iostream>
#include <fcntl.h>
#include <aio.h>
#include <unistd.h>
using namespace std;

int main() {
    const char* filename = "example.txt";
    const char* message = "Hello, asynchronous I/O!";
    int fd = open(filename, O_WRONLY | O_CREAT, 0644);
    if (fd == -1) {
        perror("open");
        return 1;
    }

    struct aiocb aio;
    memset(&aio, 0, sizeof(aio));
    aio.aio_fildes = fd;
    aio.aio_buf = message;
    aio.aio_nbytes = strlen(message);

    if (aio_write(&aio) == -1) {
        perror("aio_write");
        close(fd);
        return 1;
    }

    while (aio_error(&aio) == EINPROGRESS) {
        cout << "Writing..." << endl;
        usleep(100000);
    }

    if (aio_return(&aio) == -1) {
        perror("aio_return");
        close(fd);
        return 1;
    }

    cout << "Write complete" << endl;
    close(fd);
    return 0;
}
```

- **다중 I/O (select, poll)**

```cpp
#include <iostream>
#include <sys/select.h>
#include <unistd.h>
using namespace std;

int main() {
    fd_set readfds;
    struct timeval tv;
    int retval;

    FD_ZERO(&readfds);
    FD_SET(STDIN_FILENO, &readfds);

    tv.tv_sec = 5;
    tv.tv_usec = 0;

    retval = select(STDIN_FILENO + 1, &readfds, NULL, NULL, &tv);

    if (retval == -1) {
        perror("select()");
    } else if (retval) {
        cout << "Data is available now." << endl;
        char buf[1024];
        read(STDIN_FILENO, buf, sizeof(buf));
        cout << "You entered: " << buf << endl;
    } else {
        cout << "No data within five seconds." << endl;
    }

    return 0;
}
```

#### 과제

1. **파일 읽기 및 쓰기 프로그램 작성**
   - 텍스트 파일을 생성하고, 텍스트 파일에서 데이터를 읽어와 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <fstream>
using namespace std;

int main() {
    ofstream outFile("example.txt");
    if (outFile.is_open()) {
        outFile << "Hello, File I/O!" << endl;
        outFile.close();
    } else {
        cout << "Unable to open file for writing" << endl;
    }

    ifstream inFile("example.txt");
    if (inFile.is_open()) {
        string line;
        while (getline(inFile, line)) {
            cout << line << endl;
        }
        inFile.close();
    } else {
        cout << "Unable to open file for reading" << endl;
    }

    return 0;
}
```

2. **프로세스 생성 및 실행 프로그램 작성**
   - 새로운 프로세스를 생성하여, 자식 프로세스에서 "Hello, World!"를 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <unistd.h>
using namespace std;

int main() {
    pid_t pid = fork();

    if (pid < 0) {
        cout << "Fork failed" << endl;
    } else if (pid == 0) {
        cout << "Hello, World! from Child process" << endl;
    } else {
        cout << "Parent process" << endl;
        wait(NULL);
    }

    return 0;
}
```

3. **프로세스 간 통신 프로그램 작성**
   - 메시지 큐를 사용하여 두 프로세스 간에 메시지를 주고받는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <sys/ipc.h>
#include <sys/msg.h>
using namespace std;

struct message_buffer {
    long message_type;
    char message_text[100];
};

int main() {
    key_t key = ftok("progfile", 65);
    int msgid = msgget(key, 0666 | IPC_CREAT);
    message_buffer message;

    message.message_type = 1;
    strcpy(message.message_text, "Hello, IPC!");
    msgsnd(msgid, &message, sizeof(message), 0);
    cout << "Message sent: " << message.message_text << endl;

    msgrcv(msgid, &message, sizeof(message), 1, 0);
    cout << "Message received: " << message.message_text << endl;

    msgctl(msgid, IPC_RMID, NULL);

    return 0;
}
```

4. **비동기 I/O 프로그램 작성**
   - aio를 사용하여 비동기적으로 파일에 데이터를 쓰는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <fcntl.h>
#include <aio.h>
#include <unistd.h>
using namespace std;

int main() {
    const char* filename = "example.txt";
    const char* message = "Hello, asynchronous I/O!";
    int fd = open(filename, O_WRONLY | O_CREAT, 0644);
    if (fd == -1) {
        perror("open");
        return 1;
    }

    struct aiocb aio;
    memset(&aio, 0, sizeof(aio));
    aio.aio_fildes = fd;
    aio.aio_buf = message;
    aio.aio_nbytes = strlen(message);

    if (aio_write(&aio) == -1) {
        perror("aio_write");
        close(fd);
        return 1;
    }

    while (aio_error(&aio) == EINPROGRESS) {
        cout << "Writing..." << endl;
        usleep(100000);
    }

    if (aio_return(&aio) == -1) {
        perror("aio_return");
        close(fd);
        return 1;
    }

    cout << "Write

 complete" << endl;
    close(fd);
    return 0;
}
```

#### 퀴즈

1. **파일 시스템 접근에 대한 설명 중 맞는 것은?**
   - A) 파일 읽기/쓰기는 항상 비동기적으로 수행된다.
   - B) 파일 상태는 `stat` 함수를 사용하여 확인할 수 있다.
   - C) 파일은 열지 않고도 데이터를 읽을 수 있다.
   - D) 파일 시스템은 메모리 관리와 무관하다.

2. **프로세스 제어에 대한 설명 중 맞는 것은?**
   - A) 프로세스는 부모 프로세스 없이 생성될 수 있다.
   - B) `fork` 함수는 새로운 프로세스를 생성한다.
   - C) 자식 프로세스는 부모 프로세스와 동일한 PID를 가진다.
   - D) 프로세스 간 통신은 불가능하다.

3. **비동기 I/O의 장점 중 맞는 것은?**
   - A) 비동기 I/O는 항상 데이터 손실을 초래한다.
   - B) 비동기 I/O는 시스템 성능을 저하시킨다.
   - C) 비동기 I/O는 I/O 작업 중에도 다른 작업을 수행할 수 있게 한다.
   - D) 비동기 I/O는 동기 I/O보다 항상 느리다.

4. **프로세스 간 통신(IPC)에 대한 설명 중 맞는 것은?**
   - A) IPC는 동일한 프로세스 내에서만 사용된다.
   - B) 메시지 큐는 프로세스 간에 데이터를 주고받을 수 있는 방법이다.
   - C) IPC는 프로세스 간 데이터 공유를 방지한다.
   - D) IPC는 항상 비동기적으로 동작한다.

#### 퀴즈 해설

1. **파일 시스템 접근에 대한 설명 중 맞는 것은?**
   - **정답: B) 파일 상태는 `stat` 함수를 사용하여 확인할 수 있다.**
     - 해설: `stat` 함수는 파일의 상태 정보를 확인하는 데 사용됩니다.

2. **프로세스 제어에 대한 설명 중 맞는 것은?**
   - **정답: B) `fork` 함수는 새로운 프로세스를 생성한다.**
     - 해설: `fork` 함수는 새로운 프로세스를 생성하여 자식 프로세스를 만듭니다.

3. **비동기 I/O의 장점 중 맞는 것은?**
   - **정답: C) 비동기 I/O는 I/O 작업 중에도 다른 작업을 수행할 수 있게 한다.**
     - 해설: 비동기 I/O는 I/O 작업이 완료될 때까지 기다리지 않고 다른 작업을 수행할 수 있게 합니다.

4. **프로세스 간 통신(IPC)에 대한 설명 중 맞는 것은?**
   - **정답: B) 메시지 큐는 프로세스 간에 데이터를 주고받을 수 있는 방법이다.**
     - 해설: 메시지 큐는 프로세스 간에 데이터를 주고받기 위해 사용되는 IPC 메커니즘 중 하나입니다.

다음 주차 강의 내용을 요청하시면, 26주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.