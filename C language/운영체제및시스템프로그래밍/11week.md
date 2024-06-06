### 11주차: 가상 메모리

**강의 목표:** 가상 메모리의 개념과 작동 원리를 이해하고, 다양한 페이지 교체 알고리즘을 학습합니다. 이를 통해 가상 메모리의 효율성을 높이는 방법을 익히고, 가상 메모리 관리의 중요성을 이해합니다.

**강의 내용:**

1. **가상 메모리 개념**
   - 가상 메모리란 무엇인가?
     - 프로세스가 실제 물리 메모리보다 큰 주소 공간을 가질 수 있도록 하는 메모리 관리 기법
   - 가상 주소와 물리 주소
     - 주소 변환 과정 (페이지 테이블)
   - 페이지와 프레임
     - 고정 크기의 블록으로 나누어 관리
   - 페이지 테이블
     - 페이지 번호와 프레임 번호 매핑

2. **페이지 교체 알고리즘**
   - 페이지 부재(Page Fault)와 페이지 교체
     - 페이지 부재 발생 시 처리 과정
   - FIFO (First-In, First-Out)
     - 먼저 들어온 페이지를 먼저 교체
   - LRU (Least Recently Used)
     - 가장 오랫동안 사용되지 않은 페이지를 교체
   - Optimal
     - 앞으로 가장 오랫동안 사용되지 않을 페이지를 교체 (이론적 알고리즘)
   - 추가 알고리즘
     - LFU (Least Frequently Used)
     - MFU (Most Frequently Used)

3. **페이지 테이블 구조**
   - 단일 레벨 페이지 테이블
   - 다중 레벨 페이지 테이블
   - 역 페이지 테이블 (Inverted Page Table)

**실습:**

1. **간단한 페이지 테이블 시뮬레이션**
   - 가상 주소를 물리 주소로 변환하는 페이지 테이블 시뮬레이션

```c
#include <stdio.h>
#include <stdlib.h>

#define TABLE_SIZE 10

typedef struct {
    int frame_number;
    int valid;
} PageTableEntry;

PageTableEntry page_table[TABLE_SIZE];

void initialize_table() {
    for (int i = 0; i < TABLE_SIZE; i++) {
        page_table[i].frame_number = -1;
        page_table[i].valid = 0;
    }
}

int main() {
    initialize_table();
    page_table[2].frame_number = 5;
    page_table[2].valid = 1;

    int page_number = 2;
    if (page_table[page_number].valid) {
        printf("Page %d is in frame %d\n", page_number, page_table[page_number].frame_number);
    } else {
        printf("Page %d is not in memory\n", page_number);
    }
    return 0;
}
```

2. **FIFO 페이지 교체 알고리즘 시뮬레이션**

```c
#include <stdio.h>
#include <stdlib.h>

#define FRAME_SIZE 3
#define NUM_PAGES 10

int pages[NUM_PAGES] = {0, 1, 2, 3, 0, 1, 4, 0, 1, 2};
int frames[FRAME_SIZE];
int front = 0;

void initialize_frames() {
    for (int i = 0; i < FRAME_SIZE; i++) {
        frames[i] = -1;
    }
}

void fifo_page_replacement() {
    int page_faults = 0;
    for (int i = 0; i < NUM_PAGES; i++) {
        int found = 0;
        for (int j = 0; j < FRAME_SIZE; j++) {
            if (frames[j] == pages[i]) {
                found = 1;
                break;
            }
        }
        if (!found) {
            frames[front] = pages[i];
            front = (front + 1) % FRAME_SIZE;
            page_faults++;
        }
        printf("Frames: ");
        for (int j = 0; j < FRAME_SIZE; j++) {
            if (frames[j] == -1) {
                printf("[ ] ");
            } else {
                printf("[%d] ", frames[j]);
            }
        }
        printf("\n");
    }
    printf("Total Page Faults: %d\n", page_faults);
}

int main() {
    initialize_frames();
    fifo_page_replacement();
    return 0;
}
```

**과제:**

1. **LRU 페이지 교체 알고리즘 구현**
   - LRU 알고리즘을 사용하여 페이지 교체를 시뮬레이션합니다. 각 페이지 접근 시마다 페이지의 최근 사용 시간을 갱신하여 가장 오래된 페이지를 교체합니다.

```c
#include <stdio.h>
#include <stdlib.h>

#define FRAME_SIZE 3
#define NUM_PAGES 10

int pages[NUM_PAGES] = {0, 1, 2, 3, 0, 1, 4, 0, 1, 2};
int frames[FRAME_SIZE];
int time[FRAME_SIZE];
int clk = 0;

void initialize_frames() {
    for (int i = 0; i < FRAME_SIZE; i++) {
        frames[i] = -1;
        time[i] = -1;
    }
}

int find_least_recently_used() {
    int min_time = time[0];
    int min_index = 0;
    for (int i = 1; i < FRAME_SIZE; i++) {
        if (time[i] < min_time) {
            min_time = time[i];
            min_index = i;
        }
    }
    return min_index;
}

void lru_page_replacement() {
    int page_faults = 0;
    for (int i = 0; i < NUM_PAGES; i++) {
        int found = 0;
        for (int j = 0; j < FRAME_SIZE; j++) {
            if (frames[j] == pages[i]) {
                found = 1;
                time[j] = clk++;
                break;
            }
        }
        if (!found) {
            int lru_index = find_least_recently_used();
            frames[lru_index] = pages[i];
            time[lru_index] = clk++;
            page_faults++;
        }
        printf("Frames: ");
        for (int j = 0; j < FRAME_SIZE; j++) {
            if (frames[j] == -1) {
                printf("[ ] ");
            } else {
                printf("[%d] ", frames[j]);
            }
        }
        printf("\n");
    }
    printf("Total Page Faults: %d\n", page_faults);
}

int main() {
    initialize_frames();
    lru_page_replacement();
    return 0;
}
```

2. **Optimal 페이지 교체 알고리즘 구현**
   - Optimal 알고리즘을 사용하여 페이지 교체를 시뮬레이션합니다. 각 페이지 접근 시 앞으로 가장 오랫동안 사용되지 않을 페이지를 교체합니다.

```c
#include <stdio.h>
#include <stdlib.h>

#define FRAME_SIZE 3
#define NUM_PAGES 10

int pages[NUM_PAGES] = {0, 1, 2, 3, 0, 1, 4, 0, 1, 2};
int frames[FRAME_SIZE];

void initialize_frames() {
    for (int i = 0; i < FRAME_SIZE; i++) {
        frames[i] = -1;
    }
}

int find_optimal(int current_index) {
    int furthest_index = current_index;
    int frame_index = -1;
    for (int i = 0; i < FRAME_SIZE; i++) {
        int found = 0;
        for (int j = current_index + 1; j < NUM_PAGES; j++) {
            if (frames[i] == pages[j]) {
                if (j > furthest_index) {
                    furthest_index = j;
                    frame_index = i;
                }
                found = 1;
                break;
            }
        }
        if (!found) {
            return i;
        }
    }
    if (frame_index == -1) {
        return 0;
    }
    return frame_index;
}

void optimal_page_replacement() {
    int page_faults = 0;
    for (int i = 0; i < NUM_PAGES; i++) {
        int found = 0;
        for (int j = 0; j < FRAME_SIZE; j++) {
            if (frames[j] == pages[i]) {
                found = 1;
                break;
            }
        }
        if (!found) {
            int optimal_index = find_optimal(i);
            frames[optimal_index] = pages[i];
            page_faults++;
        }
        printf("Frames: ");
        for (int j = 0; j < FRAME_SIZE; j++) {
            if (frames[j] == -1) {
                printf("[ ] ");
            } else {
                printf("[%d] ", frames[j]);
            }
        }
        printf("\n");
    }
    printf("Total Page Faults: %d\n", page_faults);
}

int main() {
    initialize_frames();
    optimal_page_replacement();
    return 0;
}
```

**퀴즈 및 해설:**

1. **가상 메모리란 무엇인가요?**
   - 가상 메모리는 프로세스가 실제 물리 메모리보다 큰 주소 공간을 가질 수 있도록 하는 메모리 관리 기법입니다. 가상 주소를 물리 주소로 변환하여 실행됩니다

.

2. **페이지와 프레임의 차이점은 무엇인가요?**
   - 페이지는 가상 메모리의 고정 크기 블록이며, 프레임은 물리 메모리의 고정 크기 블록입니다. 페이지는 페이지 테이블을 통해 프레임에 매핑됩니다.

3. **FIFO 페이지 교체 알고리즘의 단점은 무엇인가요?**
   - FIFO 페이지 교체 알고리즘은 가장 먼저 들어온 페이지를 먼저 교체합니다. 이는 페이지가 자주 사용되는 경우에도 교체될 수 있는 단점이 있습니다.

**해설:**

1. **가상 메모리**는 프로세스가 실제 물리 메모리보다 큰 주소 공간을 가질 수 있도록 하는 메모리 관리 기법입니다. 이는 가상 주소를 물리 주소로 변환하여 프로세스가 실행될 수 있도록 합니다. 가상 메모리를 사용하면 프로세스가 물리 메모리의 제약을 받지 않고 실행될 수 있습니다.

2. **페이지와 프레임**은 가상 메모리와 물리 메모리의 고정 크기 블록입니다. 페이지는 가상 메모리의 블록이고, 프레임은 물리 메모리의 블록입니다. 페이지는 페이지 테이블을 통해 프레임에 매핑됩니다. 즉, 가상 메모리의 페이지는 물리 메모리의 프레임에 저장됩니다.

3. **FIFO 페이지 교체 알고리즘의 단점**은 가장 먼저 들어온 페이지를 먼저 교체하기 때문에, 자주 사용되는 페이지가 교체될 수 있다는 점입니다. 이는 페이지 부재가 자주 발생할 수 있는 문제를 초래할 수 있습니다. 이러한 문제를 해결하기 위해 LRU 알고리즘이나 다른 페이지 교체 알고리즘이 사용됩니다.

이로써 11주차 강의를 마무리합니다. 다음 주차에는 입출력 시스템에 대해 학습합니다. 추가 질문이 있거나 특정 주제에 대해 더 알고 싶다면 언제든지 문의해 주세요.