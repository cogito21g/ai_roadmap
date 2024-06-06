### 14주차: 고급 네트워크 프로그래밍

**강의 목표:** 고급 네트워크 프로그래밍 기법을 이해하고, 네트워크 프로그래밍의 효율성과 안정성을 향상시키는 방법을 학습합니다. 또한, 비동기 I/O와 멀티플렉싱 기법을 통해 성능을 최적화하는 방법을 배웁니다.

**강의 내용:**

1. **비동기 I/O (Asynchronous I/O)**
   - 비동기 I/O란 무엇인가?
     - I/O 작업을 비동기적으로 처리하여 블로킹 없이 작업을 수행
   - 비동기 I/O의 장점
     - 응답 시간 단축, 자원 사용 효율성 증가
   - 비동기 I/O API
     - `select()`, `poll()`, `epoll()`

2. **멀티플렉싱 (Multiplexing)**
   - 멀티플렉싱이란 무엇인가?
     - 하나의 스레드에서 여러 I/O 작업을 처리하는 기법
   - 멀티플렉싱의 구현
     - `select()`, `poll()`, `epoll()`을 사용한 멀티플렉싱 구현

3. **고성능 네트워크 프로그래밍 기법**
   - 비동기 소켓 프로그래밍
     - 논블로킹 소켓 설정 (`fcntl`)
   - I/O 멀티플렉싱
     - `select()`, `poll()`, `epoll()`을 사용한 효율적인 네트워크 프로그래밍
   - 이벤트 기반 프로그래밍
     - `libevent`와 같은 라이브러리를 사용하여 이벤트 기반 네트워크 프로그래밍 구현

4. **보안 네트워크 프로그래밍**
   - SSL/TLS 개념
     - SSL/TLS를 사용한 보안 통신
   - OpenSSL 라이브러리 사용
     - OpenSSL을 사용하여 보안 소켓 구현

**실습:**

1. **비동기 소켓 프로그래밍 예제**

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/epoll.h>

#define PORT 8080
#define MAX_EVENTS 10

int set_nonblocking(int sockfd) {
    int flags = fcntl(sockfd, F_GETFL, 0);
    if (flags == -1) {
        perror("fcntl");
        return -1;
    }
    if (fcntl(sockfd, F_SETFL, flags | O_NONBLOCK) == -1) {
        perror("fcntl");
        return -1;
    }
    return 0;
}

int main() {
    int server_fd, new_socket, epoll_fd;
    struct sockaddr_in address;
    struct epoll_event event, events[MAX_EVENTS];
    int addrlen = sizeof(address);

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    set_nonblocking(server_fd);

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

    epoll_fd = epoll_create1(0);
    if (epoll_fd == -1) {
        perror("epoll_create1");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    event.data.fd = server_fd;
    event.events = EPOLLIN | EPOLLET;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, server_fd, &event) == -1) {
        perror("epoll_ctl");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    while (1) {
        int n = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);
        for (int i = 0; i < n; i++) {
            if (events[i].events & EPOLLERR || events[i].events & EPOLLHUP || !(events[i].events & EPOLLIN)) {
                close(events[i].data.fd);
                continue;
            } else if (events[i].data.fd == server_fd) {
                while (1) {
                    new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen);
                    if (new_socket == -1) {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) {
                            break;
                        } else {
                            perror("accept");
                            break;
                        }
                    }
                    set_nonblocking(new_socket);
                    event.data.fd = new_socket;
                    event.events = EPOLLIN | EPOLLET;
                    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, new_socket, &event);
                }
                continue;
            } else {
                while (1) {
                    char buffer[1024];
                    ssize_t count = read(events[i].data.fd, buffer, sizeof(buffer));
                    if (count == -1) {
                        if (errno != EAGAIN) {
                            perror("read");
                            close(events[i].data.fd);
                        }
                        break;
                    } else if (count == 0) {
                        close(events[i].data.fd);
                        break;
                    }
                    write(events[i].data.fd, buffer, count);
                }
            }
        }
    }

    close(server_fd);
    return 0;
}
```

2. **SSL/TLS를 사용한 보안 네트워크 프로그래밍 예제**

```c
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 4433

void initialize_ssl() {
    SSL_load_error_strings();
    OpenSSL_add_ssl_algorithms();
}

SSL_CTX *create_context() {
    const SSL_METHOD *method;
    SSL_CTX *ctx;

    method = SSLv23_server_method();
    ctx = SSL_CTX_new(method);
    if (!ctx) {
        perror("Unable to create SSL context");
        ERR_print_errors_fp(stderr);
        exit(EXIT_FAILURE);
    }
    return ctx;
}

void configure_context(SSL_CTX *ctx) {
    SSL_CTX_set_ecdh_auto(ctx, 1);

    if (SSL_CTX_use_certificate_file(ctx, "cert.pem", SSL_FILETYPE_PEM) <= 0) {
        ERR_print_errors_fp(stderr);
        exit(EXIT_FAILURE);
    }

    if (SSL_CTX_use_PrivateKey_file(ctx, "key.pem", SSL_FILETYPE_PEM) <= 0) {
        ERR_print_errors_fp(stderr);
        exit(EXIT_FAILURE);
    }
}

int main() {
    int sock;
    struct sockaddr_in addr;
    SSL_CTX *ctx;

    initialize_ssl();
    ctx = create_context();
    configure_context(ctx);

    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("Unable to create socket");
        exit(EXIT_FAILURE);
    }

    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("Unable to bind");
        exit(EXIT_FAILURE);
    }

    if (listen(sock, 1) < 0) {
        perror("Unable to listen");
        exit(EXIT_FAILURE);
    }

    while (1) {
        struct sockaddr_in addr;
        uint len = sizeof(addr);
        SSL *ssl;

        int client = accept(sock, (struct sockaddr *)&addr, &len);
        if (client < 0) {
            perror("Unable to accept");
            exit(EXIT_FAILURE);
        }

        ssl = SSL_new(ctx);
        SSL_set_fd(ssl, client);

        if (SSL_accept(ssl) <= 0) {
            ERR_print_errors_fp(stderr);
        } else {
            char buf[1024] = {0};
            SSL_read(ssl, buf, sizeof(buf) - 1);
            printf("Received: %s\n", buf);
            SSL_write(ssl, "Hello, Secure World!\n", 22);
        }

        SSL_shutdown(ssl);
        SSL_free(ssl);
        close(client);
    }

    close(sock);
    SSL_CTX_free(ctx);
    EVP_cleanup();
    return 0;
}
```

**과제:**

1. **비동기 네트워크 서버 구현**
   - 비동기 I/O를 사용하여 다중 클라이언트를 처리할 수 있는 네트워크 서버를 구현합니다.
   - `select()`, `poll()`, `epoll()` 중 하나를 선택하여 구현합니다.

2. **SSL/TLS를 사용한 보안 네트워크 클라이언트 구현**
   - OpenSSL을 사용하여 SSL/TLS 기반의 보안 네트워크 클라이언트를 구현합니다.
   - 서버와 클라이언트 간의 안전한 데이터 전송을 테스트합니다.

**퀴즈 및 해설:**

1. **비동기 I/O의 주요 장점은 무엇인가요?**
   - 비동기 I/O의 주요 장점은 응답 시간 단축과 자원 사용 효율성 증가입니다. 비동기 I/O를

 사용하면 I/O 작업이 완료될 때까지 기다릴 필요 없이 다른 작업을 수행할 수 있습니다.

2. **`epoll`의 장점은 무엇인가요?**
   - `epoll`은 많은 파일 디스크립터를 효율적으로 관리할 수 있으며, 높은 성능을 제공합니다. `epoll`은 대규모 네트워크 서버에서 자주 사용됩니다.

3. **SSL/TLS의 주요 목적은 무엇인가요?**
   - SSL/TLS의 주요 목적은 네트워크를 통한 데이터 전송의 보안을 제공하는 것입니다. SSL/TLS는 데이터 암호화, 데이터 무결성, 인증 등을 통해 안전한 통신을 보장합니다.

**해설:**

1. **비동기 I/O의 주요 장점**은 응답 시간 단축과 자원 사용 효율성 증가입니다. 비동기 I/O를 사용하면 I/O 작업이 완료될 때까지 기다릴 필요 없이 다른 작업을 수행할 수 있어 시스템의 전체 성능을 향상시킬 수 있습니다.

2. **`epoll`의 장점**은 많은 파일 디스크립터를 효율적으로 관리할 수 있으며, 높은 성능을 제공한다는 점입니다. `epoll`은 이벤트 기반 I/O 멀티플렉싱 메커니즘으로, 대규모 네트워크 서버에서 자주 사용됩니다. 이는 특히 대규모 연결을 처리할 때 유리합니다.

3. **SSL/TLS의 주요 목적**은 네트워크를 통한 데이터 전송의 보안을 제공하는 것입니다. SSL/TLS는 데이터 암호화, 데이터 무결성, 인증 등을 통해 안전한 통신을 보장합니다. 이를 통해 데이터가 도청되거나 변조되지 않도록 보호할 수 있습니다.

이로써 14주차 강의를 마무리합니다. 다음 주차에는 운영체제와 시스템 프로그래밍에 대해 학습합니다. 추가 질문이 있거나 특정 주제에 대해 더 알고 싶다면 언제든지 문의해 주세요.