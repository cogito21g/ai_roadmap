### C 언어 20주차 심화 교육과정 - 15주차: 네트워크 프로그래밍 기초

#### 15주차: 네트워크 프로그래밍 기초

**강의 목표:**
15주차의 목표는 네트워크 프로그래밍의 기초 개념을 이해하고, 소켓 프로그래밍을 통해 간단한 클라이언트-서버 프로그램을 작성하는 것입니다. 이를 통해 네트워크 상에서 데이터 통신을 수행하는 능력을 기릅니다.

**강의 구성:**

##### 1. 소켓 프로그래밍 소개
- **강의 내용:**
  - 네트워크 프로그래밍의 개념
    - 네트워크의 기본 원리와 데이터 통신
  - 소켓의 개념
    - 소켓이란 무엇인가?
    - 소켓의 역할과 종류 (TCP, UDP)
  - 네트워크 프로그래밍의 주요 용어
    - IP 주소, 포트, 프로토콜, 클라이언트, 서버
- **실습:**
  - 소켓 프로그래밍 기본 개념 이해를 위한 예제 코드 분석

##### 2. TCP/IP 기초
- **강의 내용:**
  - TCP/IP 프로토콜 개요
    - TCP와 UDP의 차이점
    - TCP/IP 프로토콜 스택
  - TCP 연결 과정
    - 3-way handshake 과정
    - 연결 해제 과정
- **실습:**
  - TCP/IP 연결과 관련된 예제 코드 분석

##### 3. 간단한 클라이언트-서버 프로그램
- **강의 내용:**
  - 소켓 API 소개
    - `socket()`, `bind()`, `listen()`, `accept()`, `connect()`, `send()`, `recv()`, `close()`
  - 서버 프로그램 작성
    - 소켓 생성 및 설정
    - 클라이언트 연결 대기
    - 데이터 송수신
  - 클라이언트 프로그램 작성
    - 소켓 생성 및 서버 연결
    - 데이터 송수신
- **실습:**
  - 간단한 클라이언트-서버 프로그램 작성 및 실행

**서버 프로그램 예제:**
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 8080
#define BUFFER_SIZE 1024

int main() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[BUFFER_SIZE] = {0};
    const char *hello = "Hello from server";

    // 소켓 파일 디스크립터 생성
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // 소켓 옵션 설정
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        close(server_fd);
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    // 소켓에 주소 바인딩
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    // 연결 대기
    if (listen(server_fd, 3) < 0) {
        perror("listen");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    // 클라이언트 연결 수락
    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
        perror("accept");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    // 데이터 송수신
    read(new_socket, buffer, BUFFER_SIZE);
    printf("Message from client: %s\n", buffer);
    send(new_socket, hello, strlen(hello), 0);
    printf("Hello message sent\n");

    close(new_socket);
    close(server_fd);
    return 0;
}
```

**클라이언트 프로그램 예제:**
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 8080
#define BUFFER_SIZE 1024

int main() {
    int sock = 0;
    struct sockaddr_in serv_addr;
    char buffer[BUFFER_SIZE] = {0};
    const char *hello = "Hello from client";

    // 소켓 생성
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("\n Socket creation error \n");
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    // 서버 주소 변환
    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        printf("\nInvalid address/ Address not supported \n");
        return -1;
    }

    // 서버에 연결
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        printf("\nConnection Failed \n");
        return -1;
    }

    // 데이터 송수신
    send(sock, hello, strlen(hello), 0);
    printf("Hello message sent\n");
    read(sock, buffer, BUFFER_SIZE);
    printf("Message from server: %s\n", buffer);

    close(sock);
    return 0;
}
```

**과제:**
15주차 과제는 다음과 같습니다.
- 간단한 클라이언트-서버 프로그램을 작성하여, 클라이언트가 서버에 메시지를 보내고 서버가 응답을 보내는 프로그램 작성
- TCP와 UDP 소켓 프로그래밍의 차이점을 이해하고, UDP 소켓 프로그래밍을 통해 간단한 데이터 송수신 프로그램 작성
- 클라이언트-서버 프로그램을 확장하여, 여러 클라이언트와 동시에 통신할 수 있는 서버 프로그램 작성

**퀴즈 및 해설:**

1. **소켓이란 무엇인가요?**
   - 소켓은 네트워크 상에서 데이터를 송수신하기 위한 양 끝단을 나타내는 추상화된 개념입니다. 네트워크 프로그래밍에서 소켓은 IP 주소와 포트 번호를 통해 통신합니다.

2. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    const char *hello = "Hello from server";
    send(new_socket, hello, strlen(hello), 0);
    printf("Hello message sent\n");
    ```
   - 출력 결과는 `Hello message sent`입니다. 서버가 클라이언트에게 메시지를 전송한 후 해당 메시지를 출력합니다.

3. **TCP와 UDP의 차이점은 무엇인가요?**
   - TCP는 연결 지향형 프로토콜로, 데이터의 신뢰성과 순서를 보장합니다. 3-way handshake를 통해 연결을 설정합니다. UDP는 비연결형 프로토콜로, 데이터그램을 전송하며 신뢰성과 순서를 보장하지 않습니다. TCP는 신뢰성이 높고, UDP는 속도가 빠릅니다.

4. **다음 코드의 역할은 무엇인가요?**
    ```c
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }
    ```
   - 이 코드는 서버 소켓을 특정 IP 주소와 포트에 바인딩합니다. 바인딩 실패 시 에러 메시지를 출력하고 프로그램을 종료합니다.

5. **클라이언트가 서버에 연결하려면 어떤 함수를 사용하나요?**
   - 클라이언트는 `connect()` 함수를 사용하여 서버에 연결합니다. 이 함수는 서버의 IP 주소와 포트 번호를 사용하여 연결을 시도합니다.

**해설:**
1. 소켓은 네트워크 상에서 데이터를 송수신하기 위한 양 끝단을 나타내는 추상화된 개념입니다. 네트워크 프로그래밍에서 소켓은 IP 주소와 포트 번호를 통해 통신합니다.
2. `send()` 함수는 데이터를 클라이언트에게 전송하고, `printf()` 함수는 메시지를 출력합니다. 따라서 출력 결과는 `Hello message sent`입니다.
3. TCP는 연결 지향형 프로토콜로, 데이터의 신뢰성과 순서를 보장합니다. UDP는 비연결형 프로토콜로, 데이터그램을 전송하며 신뢰성과 순서를 보장하지 않습니다. TCP는 신뢰성이 높고, UDP는 속도가 빠릅니다.
4. 이 코드는 서버 소켓을 특정 IP 주소와 포트에 바인딩합니다. 바인딩 실패 시 에러 메시지를 출력하고 프로그램을 종료합니다.
5. 클라이언트는 `connect()` 함수를 사용하여 서버에 연결합니다. 이 함수는 서버의 IP 주소와 포트 번호를 사용하여 연결을 시도합니다.

이 15주차 강의는 학생들이 네트워크 프로그래밍의 기초 개념을 이해하고, 소켓 프로그래밍을 통해 간단한 클라이언트-서버 프로그램을

 작성하는 능력을 기를 수 있도록 도와줍니다.