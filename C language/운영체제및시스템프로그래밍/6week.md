### 6주차: 파일 시스템 구현

**강의 목표:** 파일 시스템 구현의 세부 사항을 학습하고, 블록 할당 및 관리를 이해합니다. 또한, 디렉토리 구조를 설계하고 파일 읽기/쓰기 기능을 구현합니다.

**강의 내용:**

1. **파일 시스템 구현의 세부 사항**
   - 파일 시스템 설계 고려사항
     - 데이터 구조 (디렉토리, 파일 테이블, 블록 맵 등)
     - 메타데이터 관리 (파일 속성, 접근 권한 등)
     - 파일 시스템의 크기 제한 및 성능 최적화
   - 파일 시스템 초기화 및 설정
     - 파일 시스템 초기화 과정
     - 파일 시스템 포맷팅

2. **블록 할당 및 관리**
   - 블록 할당 전략
     - 연속 할당
     - 연결 할당
     - 인덱스 할당
   - 프리 블록 리스트 (Free Block List)
     - 사용 가능한 블록의 관리
   - 블록 할당 및 해제 함수

3. **디렉토리 구조와 관리**
   - 디렉토리 구조의 개념
     - 파일 및 디렉토리의 계층적 구조
   - 디렉토리 엔트리
     - 파일 이름, inode 번호, 파일 유형 등
   - 디렉토리 조작 명령 (생성, 삭제, 열기, 닫기 등)

4. **파일 읽기/쓰기 구현**
   - 파일 열기 및 닫기
     - 파일 디스크립터 관리
   - 파일 읽기 및 쓰기
     - 읽기/쓰기 함수 구현
   - 데이터 블록 접근 및 관리

**실습:**

1. **간단한 파일 시스템 구현**
   - 파일 생성, 삭제, 읽기, 쓰기 기능을 포함한 간단한 파일 시스템을 구현합니다.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FILES 100
#define MAX_FILENAME 20
#define MAX_BLOCKS 1000
#define BLOCK_SIZE 512

typedef struct {
    char name[MAX_FILENAME];
    int size;
    int start_block;
} File;

File file_system[MAX_FILES];
int disk[MAX_BLOCKS][BLOCK_SIZE];
int free_blocks[MAX_BLOCKS];

void initialize_file_system() {
    for (int i = 0; i < MAX_FILES; i++) {
        file_system[i].name[0] = '\0';
        file_system[i].size = 0;
        file_system[i].start_block = -1;
    }
    for (int i = 0; i < MAX_BLOCKS; i++) {
        free_blocks[i] = 1; // 1 indicates free block
    }
}

int find_free_block() {
    for (int i = 0; i < MAX_BLOCKS; i++) {
        if (free_blocks[i] == 1) {
            free_blocks[i] = 0;
            return i;
        }
    }
    return -1; // No free block found
}

void create_file(const char *name, int size) {
    int start_block = find_free_block();
    if (start_block == -1) {
        printf("No free blocks available\n");
        return;
    }
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
            free_blocks[file_system[i].start_block] = 1;
            file_system[i].name[0] = '\0';
            file_system[i].size = 0;
            file_system[i].start_block = -1;
            printf("File '%s' deleted.\n", name);
            break;
        }
    }
}

void write_file(const char *name, const char *data) {
    for (int i = 0; i < MAX_FILES; i++) {
        if (strcmp(file_system[i].name, name) == 0) {
            int block = file_system[i].start_block;
            strcpy((char *)disk[block], data);
            printf("Data written to file '%s': %s\n", name, data);
            break;
        }
    }
}

void read_file(const char *name) {
    for (int i = 0; i < MAX_FILES; i++) {
        if (strcmp(file_system[i].name, name) == 0) {
            int block = file_system[i].start_block;
            printf("Data read from file '%s': %s\n", name, (char *)disk[block]);
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
    create_file("file1.txt", 100);
    create_file("file2.txt", 200);
    write_file("file1.txt", "Hello, World!");
    read_file("file1.txt");
    list_files();
    delete_file("file1.txt");
    list_files();
    return 0;
}
```

**과제:**

1. **디렉토리 구조 구현**
   - 파일 시스템을 확장하여 디렉토리 구조를 추가합니다. 디렉토리 내에 파일을 생성하고 관리할 수 있도록 합니다.
   - 디렉토리 생성, 삭제, 파일 이동 등의 기능을 구현합니다.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FILES 100
#define MAX_FILENAME 20
#define MAX_BLOCKS 1000
#define BLOCK_SIZE 512

typedef struct {
    char name[MAX_FILENAME];
    int size;
    int start_block;
    int is_directory;
    struct Directory *parent;
    struct Directory *child;
} File;

typedef struct Directory {
    char name[MAX_FILENAME];
    struct Directory *parent;
    struct Directory *child;
    File files[MAX_FILES];
} Directory;

Directory root;
int disk[MAX_BLOCKS][BLOCK_SIZE];
int free_blocks[MAX_BLOCKS];

void initialize_file_system() {
    strcpy(root.name, "/");
    root.parent = NULL;
    root.child = NULL;
    for (int i = 0; i < MAX_FILES; i++) {
        root.files[i].name[0] = '\0';
        root.files[i].size = 0;
        root.files[i].start_block = -1;
        root.files[i].is_directory = 0;
        root.files[i].parent = &root;
        root.files[i].child = NULL;
    }
    for (int i = 0; i < MAX_BLOCKS; i++) {
        free_blocks[i] = 1; // 1 indicates free block
    }
}

int find_free_block() {
    for (int i = 0; i < MAX_BLOCKS; i++) {
        if (free_blocks[i] == 1) {
            free_blocks[i] = 0;
            return i;
        }
    }
    return -1; // No free block found
}

void create_file(const char *name, int size, Directory *dir) {
    int start_block = find_free_block();
    if (start_block == -1) {
        printf("No free blocks available\n");
        return;
    }
    for (int i = 0; i < MAX_FILES; i++) {
        if (dir->files[i].name[0] == '\0') {
            strcpy(dir->files[i].name, name);
            dir->files[i].size = size;
            dir->files[i].start_block = start_block;
            dir->files[i].is_directory = 0;
            printf("File '%s' created in directory '%s': Size = %d, Start Block = %d\n", name, dir->name, size, start_block);
            break;
        }
    }
}

void create_directory(const char *name, Directory *parent) {
    for (int i = 0; i < MAX_FILES; i++) {
        if (parent->files[i].name[0] == '\0') {
            strcpy(parent->files[i].name, name);
            parent->files[i].size = 0;
            parent->files[i].start_block = -1;
            parent->files[i].is_directory = 1;
            parent->files[i].child = malloc(sizeof(Directory));
            strcpy(parent->files[i].child->name, name);
            parent->files[i].child->parent = parent;
            parent->files[i].child->child = NULL;
            for (int j = 0; j < MAX_FILES; j++) {
                parent->files[i].child->files[j].name[0] = '\0';
                parent->files[i].child->files[j].size = 0;
                parent->files[i].child->

files[j].start_block = -1;
                parent->files[i].child->files[j].is_directory = 0;
                parent->files[i].child->files[j].parent = parent->files[i].child;
                parent->files[i].child->files[j].child = NULL;
            }
            printf("Directory '%s' created in directory '%s'\n", name, parent->name);
            break;
        }
    }
}

void delete_file(const char *name, Directory *dir) {
    for (int i = 0; i < MAX_FILES; i++) {
        if (strcmp(dir->files[i].name, name) == 0) {
            free_blocks[dir->files[i].start_block] = 1;
            dir->files[i].name[0] = '\0';
            dir->files[i].size = 0;
            dir->files[i].start_block = -1;
            dir->files[i].is_directory = 0;
            printf("File '%s' deleted from directory '%s'.\n", name, dir->name);
            break;
        }
    }
}

void write_file(const char *name, const char *data, Directory *dir) {
    for (int i = 0; i < MAX_FILES; i++) {
        if (strcmp(dir->files[i].name, name) == 0) {
            int block = dir->files[i].start_block;
            strcpy((char *)disk[block], data);
            printf("Data written to file '%s' in directory '%s': %s\n", name, dir->name, data);
            break;
        }
    }
}

void read_file(const char *name, Directory *dir) {
    for (int i = 0; i < MAX_FILES; i++) {
        if (strcmp(dir->files[i].name, name) == 0) {
            int block = dir->files[i].start_block;
            printf("Data read from file '%s' in directory '%s': %s\n", name, dir->name, (char *)disk[block]);
            break;
        }
    }
}

void list_files(Directory *dir) {
    printf("Directory '%s' Content:\n", dir->name);
    for (int i = 0; i < MAX_FILES; i++) {
        if (dir->files[i].name[0] != '\0') {
            if (dir->files[i].is_directory) {
                printf("Directory: %s\n", dir->files[i].name);
            } else {
                printf("File: %s, Size: %d, Start Block: %d\n", dir->files[i].name, dir->files[i].size, dir->files[i].start_block);
            }
        }
    }
}

int main() {
    initialize_file_system();
    create_directory("docs", &root);
    create_file("file1.txt", 100, root.files[0].child);
    create_file("file2.txt", 200, &root);
    write_file("file1.txt", "Hello, World!", root.files[0].child);
    read_file("file1.txt", root.files[0].child);
    list_files(&root);
    list_files(root.files[0].child);
    delete_file("file1.txt", root.files[0].child);
    list_files(root.files[0].child);
    return 0;
}
```

**퀴즈 및 해설:**

1. **블록 할당 전략에는 어떤 것들이 있나요?**
   - 블록 할당 전략에는 연속 할당, 연결 할당, 인덱스 할당 등이 있습니다. 연속 할당은 연속된 블록에 파일을 저장하고, 연결 할당은 블록을 연결하여 파일을 저장하며, 인덱스 할당은 인덱스를 통해 블록을 관리합니다.

2. **파일 디스크립터와 inode의 차이점은 무엇인가요?**
   - 파일 디스크립터는 파일이 열릴 때 생성되는 정수형 식별자이며, inode는 파일의 메타데이터를 저장하는 구조체입니다. 파일 디스크립터는 파일 조작을 위한 핸들로 사용되고, inode는 파일의 속성 및 위치 정보를 포함합니다.

3. **파일 시스템 초기화 과정은 무엇인가요?**
   - 파일 시스템 초기화 과정은 파일 시스템의 데이터 구조를 설정하고, 필요한 메타데이터를 초기화하는 과정입니다. 이는 파일 시스템을 사용할 수 있도록 준비하는 과정으로, 주로 포맷팅과 관련됩니다.

**해설:**

1. **블록 할당 전략**은 파일을 디스크에 저장할 때 사용하는 방법으로, 연속 할당, 연결 할당, 인덱스 할당이 있습니다. 연속 할당은 파일을 연속된 블록에 저장하여 빠른 접근이 가능하지만, 디스크 단편화 문제가 발생할 수 있습니다. 연결 할당은 블록을 연결하여 파일을 저장하며, 단편화 문제를 해결할 수 있지만, 접근 속도가 느려질 수 있습니다. 인덱스 할당은 인덱스를 통해 블록을 관리하여 효율적인 접근과 단편화 문제를 해결합니다.

2. **파일 디스크립터와 inode**는 파일 시스템에서 중요한 역할을 합니다. 파일 디스크립터는 파일이 열릴 때 생성되는 정수형 식별자로, 파일 조작을 위한 핸들로 사용됩니다. inode는 파일의 메타데이터를 저장하는 구조체로, 파일의 속성 및 위치 정보를 포함합니다. inode는 파일 시스템에서 파일의 정보를 관리하는 데 중요한 역할을 합니다.

3. **파일 시스템 초기화 과정**은 파일 시스템을 사용할 수 있도록 준비하는 과정입니다. 이는 파일 시스템의 데이터 구조를 설정하고, 필요한 메타데이터를 초기화하는 과정으로, 주로 포맷팅과 관련됩니다. 파일 시스템 초기화는 파일 시스템을 생성하고, 데이터를 저장하고 관리할 수 있는 환경을 만드는 중요한 단계입니다.

이로써 6주차 강의를 마무리합니다. 다음 주차에는 시스템 호출(System Calls)에 대해 학습합니다. 추가 질문이 있거나 특정 주제에 대해 더 알고 싶다면 언제든지 문의해 주세요.