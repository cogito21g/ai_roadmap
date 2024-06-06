### 17주차: 고급 파일 시스템

**강의 목표:** 저널링 파일 시스템과 RAID의 개념을 이해하고, 고급 파일 시스템(ZFS, Btrfs)의 구조와 기능을 학습합니다. 또한, 저널링 파일 시스템의 시뮬레이션을 통해 실습을 진행합니다.

**강의 내용:**

1. **저널링 파일 시스템**
   - 저널링 파일 시스템이란?
     - 파일 시스템의 일관성을 유지하기 위해 변경 내용을 기록하는 시스템
   - 저널링의 종류
     - 메타데이터 저널링, 전체 저널링
   - 저널링의 장점과 단점
     - 장점: 데이터 일관성 유지, 빠른 복구
     - 단점: 성능 오버헤드

2. **RAID (Redundant Array of Independent Disks)**
   - RAID 개념
     - 여러 개의 물리적 디스크를 하나의 논리적 디스크로 구성
   - RAID 레벨
     - RAID 0: 스트라이핑
     - RAID 1: 미러링
     - RAID 5: 패리티를 이용한 스트라이핑
     - RAID 6: 이중 패리티를 이용한 스트라이핑
     - RAID 10: 스트라이핑과 미러링 결합
   - RAID의 장점과 단점
     - 장점: 데이터 보호, 성능 향상
     - 단점: 비용 증가, 복잡성

3. **고급 파일 시스템**
   - ZFS (Zettabyte File System)
     - ZFS의 주요 기능: 스냅샷, 데이터 무결성 검증, 압축, RAID-Z
   - Btrfs (B-Tree File System)
     - Btrfs의 주요 기능: 스냅샷, 서브볼륨, 데이터 무결성 검증, 인라인 디듀플리케이션
   - ZFS와 Btrfs의 비교
     - 장단점 비교, 사용 사례

**실습:**

1. **저널링 파일 시스템 시뮬레이션**

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LOG_SIZE 100

typedef struct {
    int id;
    char operation[10];
    char data[100];
} JournalEntry;

JournalEntry log[MAX_LOG_SIZE];
int log_count = 0;

void write_to_journal(int id, const char* operation, const char* data) {
    if (log_count >= MAX_LOG_SIZE) {
        printf("Journal is full!\n");
        return;
    }
    log[log_count].id = id;
    strcpy(log[log_count].operation, operation);
    strcpy(log[log_count].data, data);
    log_count++;
}

void commit() {
    printf("Committing journal entries to disk...\n");
    for (int i = 0; i < log_count; i++) {
        printf("ID: %d, Operation: %s, Data: %s\n", log[i].id, log[i].operation, log[i].data);
    }
    log_count = 0;
}

int main() {
    write_to_journal(1, "WRITE", "Hello, World!");
    write_to_journal(2, "WRITE", "Journal entry 2");
    write_to_journal(3, "DELETE", "Journal entry 3");
    commit();
    return 0;
}
```

2. **RAID 0 시뮬레이션**

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DISK_COUNT 2
#define DISK_SIZE 10

char disks[DISK_COUNT][DISK_SIZE];

void raid0_write(int block, const char* data) {
    int disk = block % DISK_COUNT;
    int offset = block / DISK_COUNT;
    strcpy(&disks[disk][offset * (DISK_SIZE / DISK_COUNT)], data);
}

void raid0_read(int block, char* buffer) {
    int disk = block % DISK_COUNT;
    int offset = block / DISK_COUNT;
    strcpy(buffer, &disks[disk][offset * (DISK_SIZE / DISK_COUNT)]);
}

int main() {
    memset(disks, 0, sizeof(disks));
    raid0_write(0, "Hello");
    raid0_write(1, "World");

    char buffer[DISK_SIZE / DISK_COUNT];
    raid0_read(0, buffer);
    printf("Block 0: %s\n", buffer);
    raid0_read(1, buffer);
    printf("Block 1: %s\n", buffer);

    return 0;
}
```

3. **Btrfs 서브볼륨 생성 및 스냅샷**

```sh
# Btrfs 파일 시스템 생성
mkfs.btrfs /dev/sdb1

# 마운트
mount /dev/sdb1 /mnt

# 서브볼륨 생성
btrfs subvolume create /mnt/subvol

# 파일 작성
echo "Hello, World!" > /mnt/subvol/hello.txt

# 스냅샷 생성
btrfs subvolume snapshot /mnt/subvol /mnt/snapshot

# 스냅샷 확인
btrfs subvolume list /mnt
```

**과제:**

1. **RAID 5 시뮬레이션**
   - RAID 5의 패리티를 사용한 스트라이핑을 시뮬레이션합니다.
   - 데이터를 여러 디스크에 분산 저장하고 패리티를 계산하여 데이터 복구 기능을 구현합니다.

2. **고급 파일 시스템 비교 분석**
   - ZFS와 Btrfs의 주요 기능과 성능을 비교 분석합니다.
   - 각 파일 시스템의 장단점과 사용 사례를 조사하고, 특정 상황에 적합한 파일 시스템을 선택하는 이유를 설명합니다.

**퀴즈 및 해설:**

1. **저널링 파일 시스템의 장점은 무엇인가요?**
   - 저널링 파일 시스템의 장점은 데이터 일관성 유지와 빠른 복구입니다. 시스템 충돌이나 전원 장애 발생 시 저널 로그를 사용하여 파일 시스템을 복구할 수 있습니다.

2. **RAID 1과 RAID 5의 차이점은 무엇인가요?**
   - RAID 1은 미러링을 통해 데이터를 복제하여 고가용성을 제공하지만, 저장 용량의 절반만 사용할 수 있습니다. RAID 5는 스트라이핑과 패리티를 사용하여 데이터를 분산 저장하고, 디스크 장애 시 데이터 복구가 가능합니다.

3. **ZFS와 Btrfs의 주요 기능은 무엇인가요?**
   - ZFS의 주요 기능은 스냅샷, 데이터 무결성 검증, 압축, RAID-Z입니다. Btrfs의 주요 기능은 스냅샷, 서브볼륨, 데이터 무결성 검증, 인라인 디듀플리케이션입니다. 두 파일 시스템 모두 고급 기능을 제공하여 데이터 관리와 보호를 효율적으로 수행합니다.

**해설:**

1. **저널링 파일 시스템의 장점**은 데이터 일관성 유지와 빠른 복구입니다. 저널링 파일 시스템은 파일 시스템 변경 사항을 저널에 기록하여, 시스템 충돌이나 전원 장애 발생 시 저널 로그를 사용하여 파일 시스템을 복구할 수 있습니다.

2. **RAID 1과 RAID 5의 차이점**은 RAID 1은 미러링을 통해 데이터를 복제하여 고가용성을 제공하지만, 저장 용량의 절반만 사용할 수 있다는 점입니다. RAID 5는 스트라이핑과 패리티를 사용하여 데이터를 분산 저장하고, 디스크 장애 시 패리티 정보를 사용하여 데이터를 복구할 수 있습니다.

3. **ZFS와 Btrfs의 주요 기능**은 ZFS는 스냅샷, 데이터 무결성 검증, 압축, RAID-Z 등의 기능을 제공하며, Btrfs는 스냅샷, 서브볼륨, 데이터 무결성 검증, 인라인 디듀플리케이션 등의 기능을 제공합니다. 두 파일 시스템 모두 고급 기능을 제공하여 데이터 관리와 보호를 효율적으로 수행할 수 있습니다.

이로써 17주차 강의를 마무리합니다. 다음 주차에는 고급 시스템 프로그래밍 - 프로세스 간 통신(IPC)에 대해 학습합니다. 추가 질문이 있거나 특정 주제에 대해 더 알고 싶다면 언제든지 문의해 주세요.