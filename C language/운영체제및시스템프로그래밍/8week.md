### 8주차: 장치 드라이버 기초

**강의 목표:** 장치 드라이버의 역할과 종류를 이해하고, 간단한 문자 장치 드라이버를 작성합니다. 또한, 장치 드라이버의 빌드 및 로드 과정을 학습합니다.

**강의 내용:**

1. **장치 드라이버의 역할과 종류**
   - 장치 드라이버란 무엇인가?
     - 하드웨어 장치와 운영체제 커널 간의 인터페이스
     - 하드웨어 장치의 제어 및 데이터 전송 관리
   - 장치 드라이버의 종류
     - 문자 장치 드라이버 (Character Device Driver)
     - 블록 장치 드라이버 (Block Device Driver)
     - 네트워크 장치 드라이버 (Network Device Driver)

2. **문자 장치 드라이버와 블록 장치 드라이버**
   - 문자 장치 드라이버
     - 데이터가 바이트 단위로 전송되는 장치
     - 예: 터미널, 직렬 포트
   - 블록 장치 드라이버
     - 데이터가 블록 단위로 전송되는 장치
     - 예: 하드 디스크, USB 드라이브

3. **장치 드라이버 작성 및 빌드**
   - 장치 드라이버 작성
     - 주요 함수: `init`, `exit`, `open`, `release`, `read`, `write`
   - 장치 드라이버 빌드
     - 커널 모듈 컴파일
   - 장치 드라이버 로드 및 언로드
     - `insmod`와 `rmmod` 명령어

**실습:**

1. **간단한 문자 장치 드라이버 작성**
   - 문자 장치 드라이버를 작성하여 데이터를 읽고 쓰는 기능을 구현합니다.

```c
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/uaccess.h>

#define DEVICE_NAME "mychardev"
#define BUFFER_SIZE 1024

static int major_number;
static char message[BUFFER_SIZE] = {0};
static short message_size;

static int dev_open(struct inode *inodep, struct file *filep) {
    printk(KERN_INFO "mychardev: Device opened\n");
    return 0;
}

static int dev_release(struct inode *inodep, struct file *filep) {
    printk(KERN_INFO "mychardev: Device closed\n");
    return 0;
}

static ssize_t dev_read(struct file *filep, char *buffer, size_t len, loff_t *offset) {
    int error_count = 0;
    error_count = copy_to_user(buffer, message, message_size);

    if (error_count == 0) {
        printk(KERN_INFO "mychardev: Sent %d characters to the user\n", message_size);
        return (message_size = 0);
    } else {
        printk(KERN_INFO "mychardev: Failed to send %d characters to the user\n", error_count);
        return -EFAULT;
    }
}

static ssize_t dev_write(struct file *filep, const char *buffer, size_t len, loff_t *offset) {
    copy_from_user(message, buffer, len);
    message_size = len;
    printk(KERN_INFO "mychardev: Received %zu characters from the user\n", len);
    return len;
}

static struct file_operations fops = {
    .open = dev_open,
    .read = dev_read,
    .write = dev_write,
    .release = dev_release,
};

static int __init mychardev_init(void) {
    major_number = register_chrdev(0, DEVICE_NAME, &fops);
    if (major_number < 0) {
        printk(KERN_ALERT "mychardev failed to register a major number\n");
        return major_number;
    }
    printk(KERN_INFO "mychardev: registered correctly with major number %d\n", major_number);
    return 0;
}

static void __exit mychardev_exit(void) {
    unregister_chrdev(major_number, DEVICE_NAME);
    printk(KERN_INFO "mychardev: Goodbye from the LKM!\n");
}

module_init(mychardev_init);
module_exit(mychardev_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("A simple Linux char driver");
MODULE_VERSION("0.1");
```

2. **장치 드라이버 빌드 및 로드**
   - Makefile 작성 및 커널 모듈 컴파일

```Makefile
obj-m += mychardev.o

all:
    make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
    make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
```

   - 커널 모듈 로드 및 언로드

```sh
sudo insmod mychardev.ko
sudo rmmod mychardev
dmesg | tail
```

**과제:**

1. **확장된 문자 장치 드라이버 작성**
   - 문자 장치 드라이버를 확장하여 다양한 입출력 기능을 추가합니다.
   - 여러 클라이언트의 동시 접근을 지원하도록 드라이버를 개선합니다.

2. **블록 장치 드라이버 작성**
   - 간단한 블록 장치 드라이버를 작성하여 데이터 블록을 읽고 쓰는 기능을 구현합니다.
   - 파일 시스템을 통해 드라이버를 테스트하고, 디스크 유틸리티를 사용하여 드라이버의 성능을 분석합니다.

**퀴즈 및 해설:**

1. **장치 드라이버의 역할은 무엇인가요?**
   - 장치 드라이버는 하드웨어 장치와 운영체제 커널 간의 인터페이스로, 하드웨어 장치를 제어하고 데이터 전송을 관리합니다.

2. **문자 장치 드라이버와 블록 장치 드라이버의 차이점은 무엇인가요?**
   - 문자 장치 드라이버는 데이터가 바이트 단위로 전송되는 장치를 제어하며, 블록 장치 드라이버는 데이터가 블록 단위로 전송되는 장치를 제어합니다.

3. **커널 모듈을 빌드하고 로드하는 명령어는 무엇인가요?**
   - 커널 모듈을 빌드하기 위해 `make` 명령어를 사용하며, 로드하기 위해 `insmod` 명령어를, 언로드하기 위해 `rmmod` 명령어를 사용합니다.

**해설:**

1. **장치 드라이버의 역할**은 하드웨어 장치와 운영체제 커널 간의 인터페이스로, 하드웨어 장치를 제어하고 데이터 전송을 관리하는 것입니다. 이를 통해 응용 프로그램이 하드웨어 장치를 사용할 수 있도록 합니다.

2. **문자 장치 드라이버와 블록 장치 드라이버의 차이점**은 데이터 전송 단위입니다. 문자 장치 드라이버는 데이터가 바이트 단위로 전송되는 장치를 제어하며, 블록 장치 드라이버는 데이터가 블록 단위로 전송되는 장치를 제어합니다. 예를 들어, 터미널은 문자 장치이며, 하드 디스크는 블록 장치입니다.

3. **커널 모듈을 빌드하고 로드하는 명령어**는 `make` 명령어를 사용하여 커널 모듈을 빌드하고, `insmod` 명령어를 사용하여 모듈을 로드하며, `rmmod` 명령어를 사용하여 모듈을 언로드합니다. 빌드 과정에서는 Makefile을 작성하여 커널 모듈을 컴파일합니다.

이로써 8주차 강의를 마무리합니다. 다음 주차에는 스케줄링 알고리즘에 대해 학습합니다. 추가 질문이 있거나 특정 주제에 대해 더 알고 싶다면 언제든지 문의해 주세요.