### 5주차: 파일 시스템의 개념과 구조

**강의 목표:** 파일 시스템의 역할과 구조를 이해하고, 파일 디스크립터와 inode의 개념을 학습합니다. 또한, 다양한 파일 시스템의 종류와 그 특징을 비교합니다.

**강의 내용:**

1. **파일 시스템의 역할**
   - 파일 시스템이란 무엇인가?
     - 데이터를 저장하고 관리하는 시스템
   - 파일 시스템의 주요 기능
     - 파일 및 디렉토리 관리
     - 파일 읽기/쓰기
     - 파일 접근 제어
     - 파일 보호 및 보안

2. **디스크 구조와 파일 시스템 인터페이스**
   - 디스크 구조
     - 섹터, 트랙, 실린더
     - 디스크의 논리적 구조 (파티션, 파일 시스템)
   - 파일 시스템 인터페이스
     - 파일 이름과 경로
     - 파일 속성 (크기, 유형, 권한 등)
     - 파일 조작 명령 (열기, 닫기, 읽기, 쓰기 등)

3. **파일 디스크립터와 inode**
   - 파일 디스크립터
     - 파일 식별자
     - 파일 열기 시 생성되는 정수형 식별자
   - inode (Index Node)
     - 파일의 메타데이터 저장소
     - 파일 크기, 소유자, 권한, 링크 수, 데이터 블록 위치 등 정보 포함

4. **파일 시스템의 종류와 비교**
   - FAT (File Allocation Table)
     - 간단한 파일 시스템 구조
     - 제한된 기능 및 성능
   - NTFS (New Technology File System)
     - 고급 기능 제공 (보안, 압축, 트랜잭션 등)
   - ext3/ext4 (Extended File System)
     - 저널링 파일 시스템
     - 데이터 무결성 보장
   - ZFS (Zettabyte File System)
     - 고성능, 확장성, 데이터 무결성 보장
   - Btrfs (B-tree File System)
     - 스냅샷, 압축, 데이터 무결성 보장

**실습:**

1. **파일 시스템의 기본 구조 설계 및 구현**
   - 간단한 파일 시스템 구조를 설계하고 구현합니다. 파일을 생성하고, 파일의 메타데이터를 관리하는 기능을 포함합니다.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FILES 100
#define MAX_FILENAME 20

typedef struct {
    char name[MAX_FILENAME];
    int size;
    int start_block;
} File;

File file_system[MAX_FILES];

void initialize_file_system() {
    for (int i = 0; i < MAX_FILES; i++) {
        file_system[i].name[0] = '\0';
        file_system[i].size = 0;
        file_system[i].start_block = -1;
    }
}

void create_file(const char *name, int size, int start_block) {
    for (int i = 0; i < MAX_FILES; i++) {
        if (file_system[i].name[0] == '\0') {
            strcpy(file_system[i].name, name);
            file_system[i].size = size;
            file_system[i].start_block = start_block;
            printf("File '%s' created: Size = %d, Start Block = %d\n", name, size, start_block);
            break;
        }
    }
}

void list_files() {
    printf("File System Content:\n");
    for (int i = 0; i < MAX_FILES; i++) {
        if (file_system[i].name[0] != '\0') {
            printf("File: %s, Size: %d, Start Block: %d\n", file_system[i].name, file_system[i].size, file_system[i].start_block);
        }
    }
}

int main() {
    initialize_file_system();
    create_file("file1.txt", 100, 0);
    create_file("file2.txt", 200, 1);
    list_files();
    return 0;
}
```

**과제:**

1. **파일 시스템 확장**
   - 파일 시스템을 확장하여 파일 삭제 기능을 추가합니다. 파일 삭제 시 해당 파일의 메타데이터를 초기화합니다.
   - 디렉토리 구조를 추가하여 파일을 디렉토리 내에 생성하고 관리할 수 있도록 합니다.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FILES 100
#define MAX_FILENAME 20

typedef struct {
    char name[MAX_FILENAME];
    int size;
    int start_block;
} File;

File file_system[MAX_FILES];

void initialize_file_system() {
    for (int i = 0; i < MAX_FILES; i++) {
        file_system[i].name[0] = '\0';
        file_system[i].size = 0;
        file_system[i].start_block = -1;
    }
}

void create_file(const char *name, int size, int start_block) {
    for (int i = 0; i < MAX_FILES; i++) {
        if (file_system[i].name[0] == '\0') {
            strcpy(file_system[i].name, name);
            file_system[i].size = size;
            file_system[i].start_block = start_block;
            printf("File '%s' created: Size = %d, Start Block = %d\n", name, size, start_block);
            break;
        }
    }
}

void delete_file(const char *name) {
    for (int i = 0; i < MAX_FILES; i++) {
        if (strcmp(file_system[i].name, name) == 0) {
            file_system[i].name[0] = '\0';
            file_system[i].size = 0;
            file_system[i].start_block = -1;
            printf("File '%s' deleted.\n", name);
            break;
        }
    }
}

void list_files() {
    printf("File System Content:\n");
    for (int i = 0; i < MAX_FILES; i++) {
        if (file_system[i].name[0] != '\0') {
            printf("File: %s, Size: %d, Start Block: %d\n", file_system[i].name, file_system[i].size, file_system[i].start_block);
        }
    }
}

int main() {
    initialize_file_system();
    create_file("file1.txt", 100, 0);
    create_file("file2.txt", 200, 1);
    list_files();
    delete_file("file1.txt");
    list_files();
    return 0;
}
```

**퀴즈 및 해설:**

1. **파일 시스템의 주요 기능은 무엇인가요?**
   - 파일 시스템의 주요 기능은 파일 및 디렉토리 관리, 파일 읽기/쓰기, 파일 접근 제어, 파일 보호 및 보안입니다.

2. **파일 디스크립터와 inode의 차이점은 무엇인가요?**
   - 파일 디스크립터는 파일이 열릴 때 생성되는 정수형 식별자이며, inode는 파일의 메타데이터를 저장하는 구조체입니다. 파일 디스크립터는 파일 조작을 위한 핸들로 사용되고, inode는 파일의 속성 및 위치 정보를 포함합니다.

3. **ext4 파일 시스템의 주요 특징은 무엇인가요?**
   - ext4 파일 시스템은 저널링 파일 시스템으로, 데이터 무결성을 보장하며, 대용량 파일과 디렉토리를 효율적으로 관리할 수 있습니다. 또한, 향상된 성능과 신뢰성을 제공합니다.

**해설:**

1. **파일 시스템의 주요 기능**은 파일 및 디렉토리 관리, 파일 읽기/쓰기, 파일 접근 제어, 파일 보호 및 보안입니다. 이러한 기능을 통해 파일 시스템은 데이터를 효율적으로 저장하고 관리하며, 사용자와 응용 프로그램이 파일을 안전하게 사용할 수 있도록 합니다.

2. **파일 디스크립터와 inode의 차이점**은 파일 디스크립터가 파일이 열릴 때 생성되는 정수형 식별자라는 점과 inode가 파일의 메타데이터를 저장하는 구조체라는 점입니다. 파일 디스크립터는 파일 조작을 위한 핸들로 사용되고, inode는 파일의 속성 및 위치 정보를 포함합니다.

3. **ext4 파일 시스템의 주요 특징**은 저널링을 통해 데이터 무결성을 보장하며, 대용량 파일과 디렉토리를 효율적으로 관리할 수 있다는 점입니다. 또한, 향상된 성능과 신뢰성을 제공하여 다양한 환경에서 안정적으로 사용될 수 있습니다.

이로써 5주차 강의를 마무리합니다. 다음 주차에는 파일 시스템 구현에 대해 학습합니다. 추가 질문이 있거나 특정 주제에 대해 더 알고 싶다면 언제든지 문의해 주세요.