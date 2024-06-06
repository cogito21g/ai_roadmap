### 15주차: 보안과 암호화

**강의 목표:** 보안의 중요성을 이해하고, 다양한 암호화 기법을 학습합니다. 또한, 네트워크와 시스템 보안의 기본 개념을 익히고, 이를 활용하여 안전한 프로그램을 작성하는 방법을 배웁니다.

**강의 내용:**

1. **보안의 중요성**
   - 정보 보안의 기본 원칙
     - 기밀성(Confidentiality), 무결성(Integrity), 가용성(Availability)
   - 보안 위협과 공격 유형
     - 악성 소프트웨어, 피싱, 중간자 공격, 서비스 거부 공격

2. **암호화 기법**
   - 대칭키 암호화
     - AES, DES, 3DES
   - 비대칭키 암호화
     - RSA, ECC
   - 해시 함수
     - MD5, SHA-1, SHA-256
   - 암호화와 복호화의 차이점

3. **네트워크 보안**
   - SSL/TLS 프로토콜
     - SSL/TLS의 개념과 작동 방식
   - HTTPS
     - HTTPS의 중요성과 SSL/TLS를 이용한 구현

4. **시스템 보안**
   - 인증과 권한 부여
     - 인증의 종류와 방법 (예: 비밀번호, OTP, 생체 인식)
     - 권한 부여와 접근 제어 (ACL, RBAC)
   - 로그 모니터링과 침입 탐지
     - 로그 파일의 중요성
     - 침입 탐지 시스템(IDS)의 개념과 종류

**실습:**

1. **대칭키 암호화 예제 (AES)**

```c
#include <stdio.h>
#include <string.h>
#include <openssl/aes.h>

void encrypt_decrypt_aes(const unsigned char *key, const unsigned char *text) {
    AES_KEY encryptKey, decryptKey;
    unsigned char encrypted_text[128];
    unsigned char decrypted_text[128];

    AES_set_encrypt_key(key, 128, &encryptKey);
    AES_encrypt(text, encrypted_text, &encryptKey);

    AES_set_decrypt_key(key, 128, &decryptKey);
    AES_decrypt(encrypted_text, decrypted_text, &decryptKey);

    printf("Original Text: %s\n", text);
    printf("Encrypted Text: %s\n", encrypted_text);
    printf("Decrypted Text: %s\n", decrypted_text);
}

int main() {
    const unsigned char key[16] = "thisisakey123456";
    const unsigned char text[16] = "Hello, World!";
    encrypt_decrypt_aes(key, text);
    return 0;
}
```

2. **비대칭키 암호화 예제 (RSA)**

```c
#include <stdio.h>
#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <openssl/err.h>

void generate_rsa_keys() {
    int ret;
    RSA *r = NULL;
    BIGNUM *bne = NULL;
    BIO *bp_public = NULL, *bp_private = NULL;

    int bits = 2048;
    unsigned long e = RSA_F4;

    bne = BN_new();
    ret = BN_set_word(bne, e);
    if (ret != 1) {
        printf("Error: BN_set_word\n");
        goto free_all;
    }

    r = RSA_new();
    ret = RSA_generate_key_ex(r, bits, bne, NULL);
    if (ret != 1) {
        printf("Error: RSA_generate_key_ex\n");
        goto free_all;
    }

    bp_public = BIO_new_file("public.pem", "w+");
    ret = PEM_write_bio_RSAPublicKey(bp_public, r);
    if (ret != 1) {
        printf("Error: PEM_write_bio_RSAPublicKey\n");
        goto free_all;
    }

    bp_private = BIO_new_file("private.pem", "w+");
    ret = PEM_write_bio_RSAPrivateKey(bp_private, r, NULL, NULL, 0, NULL, NULL);
    if (ret != 1) {
        printf("Error: PEM_write_bio_RSAPrivateKey\n");
        goto free_all;
    }

free_all:
    BIO_free_all(bp_public);
    BIO_free_all(bp_private);
    RSA_free(r);
    BN_free(bne);
}

int main() {
    generate_rsa_keys();
    printf("RSA keys generated.\n");
    return 0;
}
```

3. **SSL/TLS 서버 예제**

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

1. **해시 함수 사용**
   - 주어진 문자열에 대해 MD5와 SHA-256 해시를 생성하는 프로그램 작성

```c
#include <stdio.h>
#include <openssl/md5.h>
#include <openssl/sha.h>

void print_hash(unsigned char *hash, int length) {
    for (int i = 0; i < length; i++) {
        printf("%02x", hash[i]);
    }
    printf("\n");
}

int main() {
    const char *str = "Hello, World!";
    unsigned char md5_hash[MD5_DIGEST_LENGTH];
    unsigned char sha256_hash[SHA256_DIGEST_LENGTH];

    MD5((unsigned char *)str, strlen(str), md5_hash);
    SHA256((unsigned char *)str, strlen(str), sha256_hash);

    printf("MD5: ");
    print_hash(md5_hash, MD5_DIGEST_LENGTH);

    printf("SHA-256: ");
    print_hash(sha256_hash, SHA256_DIGEST_LENGTH);

    return 0;
}
```

2. **HTTPS 클라이언트 구현**
   - OpenSSL을 사용하여 HTTPS 서버에 연결하고 데이터를 송수신하는 클라이언트를 구현

```c
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 443
#define HOST "www.example.com"

void initialize_ssl() {
    SSL_load_error_strings();
    OpenSSL_add_ssl_algorithms();
}

SSL_CTX *create_context() {
    const SSL_METHOD *method;
    SSL_CTX *ctx;

    method = SSLv23_client_method();
    ctx = SSL_CTX_new(method);
    if (!ctx) {
        perror("Unable to create SSL context");
        ERR_print_errors_fp(stderr);
        exit(EXIT_FAILURE);
    }
    return ctx;
}

int main() {
    int sock;
    struct sockaddr_in addr;
    SSL_CTX *ctx;
    SSL *ssl;

    initialize_ssl();
    ctx = create_context();

    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("Unable to create socket");
        exit(EXIT_FAILURE);
   

 }

    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    inet_pton(AF_INET, HOST, &addr.sin_addr);

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("Unable to connect");
        exit(EXIT_FAILURE);
    }

    ssl = SSL_new(ctx);
    SSL_set_fd(ssl, sock);

    if (SSL_connect(ssl) <= 0) {
        ERR_print_errors_fp(stderr);
    } else {
        SSL_write(ssl, "GET / HTTP/1.1\r\nHost: " HOST "\r\n\r\n", strlen("GET / HTTP/1.1\r\nHost: " HOST "\r\n\r\n"));
        char buf[1024] = {0};
        SSL_read(ssl, buf, sizeof(buf) - 1);
        printf("Received:\n%s\n", buf);
    }

    SSL_shutdown(ssl);
    SSL_free(ssl);
    close(sock);
    SSL_CTX_free(ctx);
    EVP_cleanup();
    return 0;
}
```

**퀴즈 및 해설:**

1. **대칭키 암호화와 비대칭키 암호화의 차이점은 무엇인가요?**
   - 대칭키 암호화는 암호화와 복호화에 동일한 키를 사용하는 반면, 비대칭키 암호화는 공개키와 비밀키라는 두 개의 키를 사용합니다. 대칭키 암호화는 빠르지만 키 관리가 어렵고, 비대칭키 암호화는 느리지만 키 관리가 용이합니다.

2. **해시 함수의 주요 용도는 무엇인가요?**
   - 해시 함수는 데이터의 고유한 지문을 생성하는 데 사용됩니다. 주로 데이터 무결성 검증, 암호화 저장, 디지털 서명 등에 사용됩니다.

3. **SSL/TLS의 주요 목적은 무엇인가요?**
   - SSL/TLS의 주요 목적은 네트워크를 통한 데이터 전송의 보안을 제공하는 것입니다. SSL/TLS는 데이터 암호화, 데이터 무결성, 인증 등을 통해 안전한 통신을 보장합니다.

**해설:**

1. **대칭키 암호화와 비대칭키 암호화의 차이점**은 대칭키 암호화는 암호화와 복호화에 동일한 키를 사용하는 반면, 비대칭키 암호화는 공개키와 비밀키라는 두 개의 키를 사용한다는 점입니다. 대칭키 암호화는 빠르지만 키를 안전하게 공유해야 하는 문제가 있으며, 비대칭키 암호화는 느리지만 공개키를 안전하게 배포할 수 있어 키 관리가 용이합니다.

2. **해시 함수의 주요 용도**는 데이터의 고유한 지문을 생성하는 것입니다. 해시 함수는 주로 데이터 무결성 검증, 암호화 저장, 디지털 서명 등에 사용됩니다. 해시 값은 입력 데이터의 고유한 서명을 제공하므로, 데이터가 변경되었는지 여부를 쉽게 확인할 수 있습니다.

3. **SSL/TLS의 주요 목적**은 네트워크를 통한 데이터 전송의 보안을 제공하는 것입니다. SSL/TLS는 데이터 암호화, 데이터 무결성, 인증 등을 통해 안전한 통신을 보장합니다. 이를 통해 데이터가 도청되거나 변조되지 않도록 보호할 수 있습니다.

이로써 15주차 강의를 마무리합니다. 다음 주차에는 최적화 및 디버깅에 대해 학습합니다. 추가 질문이 있거나 특정 주제에 대해 더 알고 싶다면 언제든지 문의해 주세요.