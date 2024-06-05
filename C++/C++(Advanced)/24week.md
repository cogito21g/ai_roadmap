### 24주차: 네트워크 프로그래밍

#### 강의 목표
- 소켓 프로그래밍의 기본 개념 이해 및 구현
- Boost.Asio를 이용한 네트워크 프로그래밍 이해 및 구현
- HTTP 서버 구현을 통한 실습

#### 강의 내용

##### 1. 소켓 프로그래밍 기초
- **TCP 소켓 서버**

```cpp
#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>

using namespace std;

int main() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[1024] = {0};
    const char* hello = "Hello from server";

    // 소켓 파일 디스크립터 생성
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // 포트 재사용 옵션 설정
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(8080);

    // 소켓에 주소 바인딩
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    // 연결 대기 상태로 설정
    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    // 클라이언트 연결 수락
    if ((new_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen)) < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }

    // 데이터 수신
    read(new_socket, buffer, 1024);
    cout << buffer << endl;

    // 데이터 전송
    send(new_socket, hello, strlen(hello), 0);
    cout << "Hello message sent" << endl;

    close(new_socket);
    close(server_fd);

    return 0;
}
```

- **TCP 소켓 클라이언트**

```cpp
#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

using namespace std;

int main() {
    int sock = 0;
    struct sockaddr_in serv_addr;
    const char* hello = "Hello from client";
    char buffer[1024] = {0};

    // 소켓 파일 디스크립터 생성
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        cout << "\n Socket creation error \n";
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(8080);

    // 주소 변환
    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        cout << "\nInvalid address/ Address not supported \n";
        return -1;
    }

    // 서버에 연결
    if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        cout << "\nConnection Failed \n";
        return -1;
    }

    // 데이터 전송
    send(sock, hello, strlen(hello), 0);
    cout << "Hello message sent" << endl;

    // 데이터 수신
    read(sock, buffer, 1024);
    cout << buffer << endl;

    close(sock);

    return 0;
}
```

##### 2. Boost.Asio를 이용한 네트워크 프로그래밍
- **Boost.Asio TCP 서버**

```cpp
#include <iostream>
#include <boost/asio.hpp>

using boost::asio::ip::tcp;
using namespace std;

void session(tcp::socket sock) {
    try {
        for (;;) {
            char data[1024];

            boost::system::error_code error;
            size_t length = sock.read_some(boost::asio::buffer(data), error);
            if (error == boost::asio::error::eof)
                break;  // Connection closed cleanly by peer.
            else if (error)
                throw boost::system::system_error(error);  // Some other error.

            boost::asio::write(sock, boost::asio::buffer(data, length));
        }
    } catch (exception& e) {
        cerr << "Exception in thread: " << e.what() << "\n";
    }
}

int main() {
    try {
        boost::asio::io_context io_context;

        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 8080));

        for (;;) {
            tcp::socket socket(io_context);
            acceptor.accept(socket);
            thread(session, move(socket)).detach();
        }
    } catch (exception& e) {
        cerr << "Exception: " << e.what() << "\n";
    }

    return 0;
}
```

- **Boost.Asio TCP 클라이언트**

```cpp
#include <iostream>
#include <boost/asio.hpp>

using boost::asio::ip::tcp;
using namespace std;

int main() {
    try {
        boost::asio::io_context io_context;

        tcp::resolver resolver(io_context);
        tcp::resolver::results_type endpoints = resolver.resolve("127.0.0.1", "8080");

        tcp::socket socket(io_context);
        boost::asio::connect(socket, endpoints);

        const string msg = "Hello from client";
        boost::asio::write(socket, boost::asio::buffer(msg));

        char reply[1024];
        size_t reply_length = boost::asio::read(socket, boost::asio::buffer(reply, msg.size()));
        cout << "Reply is: ";
        cout.write(reply, reply_length);
        cout << "\n";
    } catch (exception& e) {
        cerr << "Exception: " << e.what() << "\n";
    }

    return 0;
}
```

##### 3. HTTP 서버 구현
- **Simple HTTP 서버**

```cpp
#include <iostream>
#include <boost/asio.hpp>

using boost::asio::ip::tcp;
using namespace std;

void handle_request(tcp::socket& socket) {
    boost::system::error_code ec;
    char data[1024];
    size_t length = socket.read_some(boost::asio::buffer(data), ec);

    if (!ec) {
        string request(data, length);
        cout << "Request: " << request << endl;

        string response = "HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\nHello, World!";
        boost::asio::write(socket, boost::asio::buffer(response), ec);
    }
}

int main() {
    try {
        boost::asio::io_context io_context;
        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 8080));

        for (;;) {
            tcp::socket socket(io_context);
            acceptor.accept(socket);
            handle_request(socket);
        }
    } catch (exception& e) {
        cerr << "Exception: " << e.what() << "\n";
    }

    return 0;
}
```

#### 과제

1. **TCP 소켓 서버와 클라이언트 구현**
   - TCP 소켓 서버와 클라이언트를 구현하고, 서버가 클라이언트로부터 메시지를 받아 다시 클라이언트에게 반환하도록 하세요.

```cpp
// TCP 서버 코드

#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>

using namespace std;

int main() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[1024] = {0};

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(8080);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    if ((new_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen)) < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }

    read(new_socket, buffer, 1024);
    cout << buffer << endl;

    send(new_socket, buffer, strlen(buffer), 0);

    close(new_socket);
    close(server_fd);

    return 0;
}

// TCP 클라이언트 코드

#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd

.h>
#include <cstring>

using namespace std;

int main() {
    int sock = 0;
    struct sockaddr_in serv_addr;
    const char* hello = "Hello from client";
    char buffer[1024] = {0};

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        cout << "\n Socket creation error \n";
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(8080);

    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        cout << "\nInvalid address/ Address not supported \n";
        return -1;
    }

    if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        cout << "\nConnection Failed \n";
        return -1;
    }

    send(sock, hello, strlen(hello), 0);
    cout << "Hello message sent" << endl;

    read(sock, buffer, 1024);
    cout << buffer << endl;

    close(sock);

    return 0;
}
```

2. **Boost.Asio를 이용한 비동기 TCP 서버와 클라이언트 구현**
   - Boost.Asio를 사용하여 비동기 TCP 서버와 클라이언트를 구현하세요.

```cpp
// 비동기 TCP 서버 코드

#include <iostream>
#include <boost/asio.hpp>

using boost::asio::ip::tcp;
using namespace std;

void handle_session(tcp::socket socket) {
    try {
        char data[1024];
        for (;;) {
            boost::system::error_code error;
            size_t length = socket.read_some(boost::asio::buffer(data), error);
            if (error == boost::asio::error::eof)
                break;
            else if (error)
                throw boost::system::system_error(error);

            boost::asio::write(socket, boost::asio::buffer(data, length));
        }
    } catch (exception& e) {
        cerr << "Exception: " << e.what() << "\n";
    }
}

int main() {
    try {
        boost::asio::io_context io_context;
        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 8080));

        for (;;) {
            tcp::socket socket(io_context);
            acceptor.accept(socket);
            thread(handle_session, move(socket)).detach();
        }
    } catch (exception& e) {
        cerr << "Exception: " << e.what() << "\n";
    }

    return 0;
}

// 비동기 TCP 클라이언트 코드

#include <iostream>
#include <boost/asio.hpp>

using boost::asio::ip::tcp;
using namespace std;

int main() {
    try {
        boost::asio::io_context io_context;

        tcp::resolver resolver(io_context);
        tcp::resolver::results_type endpoints = resolver.resolve("127.0.0.1", "8080");

        tcp::socket socket(io_context);
        boost::asio::connect(socket, endpoints);

        const string msg = "Hello from client";
        boost::asio::write(socket, boost::asio::buffer(msg));

        char reply[1024];
        size_t reply_length = boost::asio::read(socket, boost::asio::buffer(reply, msg.size()));
        cout << "Reply is: ";
        cout.write(reply, reply_length);
        cout << "\n";
    } catch (exception& e) {
        cerr << "Exception: " << e.what() << "\n";
    }

    return 0;
}
```

3. **HTTP 서버 구현**
   - Boost.Asio를 사용하여 간단한 HTTP 서버를 구현하세요.

```cpp
#include <iostream>
#include <boost/asio.hpp>

using boost::asio::ip::tcp;
using namespace std;

void handle_request(tcp::socket& socket) {
    boost::system::error_code ec;
    char data[1024];
    size_t length = socket.read_some(boost::asio::buffer(data), ec);

    if (!ec) {
        string request(data, length);
        cout << "Request: " << request << endl;

        string response = "HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\nHello, World!";
        boost::asio::write(socket, boost::asio::buffer(response), ec);
    }
}

int main() {
    try {
        boost::asio::io_context io_context;
        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 8080));

        for (;;) {
            tcp::socket socket(io_context);
            acceptor.accept(socket);
            handle_request(socket);
        }
    } catch (exception& e) {
        cerr << "Exception: " << e.what() << "\n";
    }

    return 0;
}
```

#### 퀴즈

1. **TCP 소켓 프로그래밍에 대한 설명 중 맞는 것은?**
   - A) TCP는 비연결형 프로토콜이다.
   - B) TCP는 데이터의 신뢰성을 보장한다.
   - C) TCP 소켓은 항상 비동기적으로 동작한다.
   - D) TCP 소켓은 브로드캐스트를 지원한다.

2. **Boost.Asio의 주요 기능 중 맞는 것은?**
   - A) Boost.Asio는 파일 입출력을 최적화한다.
   - B) Boost.Asio는 네트워크 통신을 비동기적으로 처리한다.
   - C) Boost.Asio는 그래픽 렌더링을 지원한다.
   - D) Boost.Asio는 데이터베이스 관리 시스템을 제공한다.

3. **HTTP 서버에 대한 설명 중 맞는 것은?**
   - A) HTTP 서버는 항상 동기적으로 요청을 처리한다.
   - B) HTTP 서버는 웹 브라우저와 통신하기 위해 사용된다.
   - C) HTTP 서버는 UDP 프로토콜을 사용한다.
   - D) HTTP 서버는 네트워크 계층에서 동작한다.

4. **Boost.Asio의 io_context에 대한 설명 중 맞는 것은?**
   - A) io_context는 네트워크 통신을 관리하는 객체이다.
   - B) io_context는 데이터베이스 연결을 관리한다.
   - C) io_context는 파일 시스템을 관리한다.
   - D) io_context는 멀티스레딩을 지원하지 않는다.

#### 퀴즈 해설

1. **TCP 소켓 프로그래밍에 대한 설명 중 맞는 것은?**
   - **정답: B) TCP는 데이터의 신뢰성을 보장한다.**
     - 해설: TCP는 데이터 전송의 신뢰성을 보장하는 연결형 프로토콜입니다.

2. **Boost.Asio의 주요 기능 중 맞는 것은?**
   - **정답: B) Boost.Asio는 네트워크 통신을 비동기적으로 처리한다.**
     - 해설: Boost.Asio는 네트워크 통신을 비동기적으로 처리하는 라이브러리입니다.

3. **HTTP 서버에 대한 설명 중 맞는 것은?**
   - **정답: B) HTTP 서버는 웹 브라우저와 통신하기 위해 사용된다.**
     - 해설: HTTP 서버는 웹 브라우저와 통신하기 위해 사용되는 서버입니다.

4. **Boost.Asio의 io_context에 대한 설명 중 맞는 것은?**
   - **정답: A) io_context는 네트워크 통신을 관리하는 객체이다.**
     - 해설: io_context는 네트워크 통신을 관리하는 Boost.Asio의 핵심 객체입니다.

다음 주차 강의 내용을 요청하시면, 25주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.