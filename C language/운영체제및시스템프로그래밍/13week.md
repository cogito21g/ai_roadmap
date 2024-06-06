### 13주차: 네트워크 프로그래밍 기초

**강의 목표:** 네트워크 프로그래밍의 기초 개념과 TCP/IP 프로토콜의 구조를 이해하고, 소켓 프로그래밍을 통해 간단한 클라이언트-서버 애플리케이션을 작성합니다.

**강의 내용:**

1. **네트워크 프로그래밍 개요**
   - 네트워크 프로그래밍이란 무엇인가?
     - 네트워크를 통해 데이터 전송을 수행하는 프로그래밍
   - TCP/IP 프로토콜 스택
     - 전송 계층 (TCP, UDP)
     - 네트워크 계층 (IP)
     - 링크 계층 (Ethernet 등)

2. **소켓 프로그래밍**
   - 소켓이란 무엇인가?
     - 네트워크 통신의 끝점 (Endpoint)
   - 소켓의 종류
     - 스트림 소켓 (TCP)
     - 데이터그램 소켓 (UDP)

3. **TCP 소켓 프로그래밍**
   - 클라이언트 소켓
     - `socket()`, `connect()`, `send()`, `recv()`, `close()`
   - 서버 소켓
     - `socket()`, `bind()`, `listen()`, `accept()`, `send()`, `recv()`, `close()`

4. **UDP 소켓 프로그래밍**
   - 클라이언트 소켓
     - `socket()`, `sendto()`, `recvfrom()`, `close()`
   - 서버 소켓
     - `socket()`, `bind()`, `sendto()`, `recvfrom()`, `close()`

**실습:**

1. **TCP 클라이언트-서버 프로그램 작성**

**TCP 서버 예제:**

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
    char *hello = "Hello from server";

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 3) < 0) {
        perror("listen");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen)) < 0) {
        perror("accept");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    read(new_socket, buffer, BUFFER_SIZE);
    printf("Message from client: %s\n", buffer);
    send(new_socket, hello, strlen(hello), 0);
    printf("Hello message sent\n");

    close(new_socket);
    close(server_fd);
    return 0;
}
```

**TCP 클라이언트 예제:**

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
    char *hello = "Hello from client";

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("\n Socket creation error \n");
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        printf("\nInvalid address/ Address not supported \n");
        return -1;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        printf("\nConnection Failed \n");
        return -1;
    }

    send(sock, hello, strlen(hello), 0);
    printf("Hello message sent\n");
    read(sock, buffer, BUFFER_SIZE);
    printf("Message from server: %s\n", buffer);

    close(sock);
    return 0;
}
```

2. **UDP 클라이언트-서버 프로그램 작성**

**UDP 서버 예제:**

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 8080
#define BUFFER_SIZE 1024

int main() {
    int sockfd;
    char buffer[BUFFER_SIZE];
    char *hello = "Hello from server";
    struct sockaddr_in servaddr, cliaddr;

    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    memset(&servaddr, 0, sizeof(servaddr));
    memset(&cliaddr, 0, sizeof(cliaddr));

    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = INADDR_ANY;
    servaddr.sin_port = htons(PORT);

    if (bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
        perror("bind failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    int len, n;
    len = sizeof(cliaddr);
    n = recvfrom(sockfd, (char *)buffer, BUFFER_SIZE, MSG_WAITALL, (struct sockaddr *)&cliaddr, &len);
    buffer[n] = '\0';
    printf("Message from client: %s\n", buffer);
    sendto(sockfd, (const char *)hello, strlen(hello), MSG_CONFIRM, (const struct sockaddr *)&cliaddr, len);
    printf("Hello message sent.\n");

    close(sockfd);
    return 0;
}
```

**UDP 클라이언트 예제:**

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 8080
#define BUFFER_SIZE 1024

int main() {
    int sockfd;
    char buffer[BUFFER_SIZE];
    char *hello = "Hello from client";
    struct sockaddr_in servaddr;

    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    memset(&servaddr, 0, sizeof(servaddr));

    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(PORT);
    servaddr.sin_addr.s_addr = INADDR_ANY;

    int n, len;
    sendto(sockfd, (const char *)hello, strlen(hello), MSG_CONFIRM, (const struct sockaddr *)&servaddr, sizeof(servaddr));
    printf("Hello message sent.\n");

    n = recvfrom(sockfd, (char *)buffer, BUFFER_SIZE, MSG_WAITALL, (struct sockaddr *)&servaddr, &len);
    buffer[n] = '\0';
    printf("Message from server: %s\n", buffer);

    close(sockfd);
    return 0;
}
```

**과제:**

1. **간단한 채팅 프로그램 구현**
   - TCP 소켓을 이용하여 클라이언트와 서버 간의 채팅 프로그램을 구현합니다.
   - 클라이언트와 서버는 메시지를 주고받을 수 있어야 합니다.

2. **파일 전송 프로그램 구현**
   - UDP 소켓을 이용하여 클라이언트에서 서버로 파일을 전송하는 프로그램을 구현합니다.
   - 서버는 클라이언트로부터 파일을 받아 저장합니다.

**퀴즈 및 해설:**

1. **TCP와 UDP의 차이점은 무엇인가요?**
   - TCP는 연결 지향 프로토콜로, 신뢰성 있는 데이터 전송을 보장합니다. 데이터가 순서대로 전달되고, 오류가 발생하면 재전송합니다.
   - UDP는 비연결 지향 프로토콜로, 빠른 데이터 전송을 보장하지만, 신뢰성은 보장하지 않습니다. 데이터가 순서대로 전달되지 않을 수 있으며, 오류가 발생해도 재전송하지 않습니다.

2. **소켓이란 무엇인가요?**
   - 소켓은 네트워크 통신의 끝점을 의미합니다. 네트워크를 통해 데이터를 주고받기 위해 소켓을 사용합니다.

3. **`bind()` 함수의 역할은 무엇인가요?**
   - `bind()` 함수는 소켓을 특정 IP 주소와 포트 번호에 연결하는 역할을 합니다. 서버 소켓에서 주로 사용됩니다.

**해설:**

1. **TCP와 UDP의 차이점**은 TCP는 연결 지향 프로토콜로, 신뢰성 있는 데이터 전송을 보장하는 반면, UDP는 비연결 지

향 프로토콜로, 빠른 데이터 전송을 보장하지만 신뢰성은 보장하지 않습니다. TCP는 데이터가 순서대로 전달되고, 오류가 발생하면 재전송하지만, UDP는 데이터가 순서대로 전달되지 않을 수 있으며, 오류가 발생해도 재전송하지 않습니다.

2. **소켓**은 네트워크 통신의 끝점을 의미합니다. 네트워크를 통해 데이터를 주고받기 위해 소켓을 사용합니다. 소켓은 IP 주소와 포트 번호를 기반으로 통신합니다.

3. **`bind()` 함수**는 소켓을 특정 IP 주소와 포트 번호에 연결하는 역할을 합니다. 서버 소켓에서 주로 사용되며, 클라이언트 소켓은 보통 연결할 필요가 없습니다.

이로써 13주차 강의를 마무리합니다. 다음 주차에는 멀티쓰레딩과 동기화에 대해 학습합니다. 추가 질문이 있거나 특정 주제에 대해 더 알고 싶다면 언제든지 문의해 주세요.