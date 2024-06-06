### 9주차: 스케줄링 알고리즘

**강의 목표:** CPU 스케줄링의 개념과 다양한 스케줄링 알고리즘을 이해하고, 각 알고리즘의 장단점을 학습합니다. 이를 통해 다양한 상황에서 적절한 스케줄링 알고리즘을 선택하고 구현할 수 있는 능력을 배양합니다.

**강의 내용:**

1. **CPU 스케줄링 개념**
   - 스케줄링이란 무엇인가?
     - CPU 자원을 여러 프로세스에 할당하는 방법
   - 스케줄링 기준
     - CPU 이용률, 처리율, 대기 시간, 응답 시간, 공정성

2. **스케줄링 알고리즘**
   - **First-Come, First-Served (FCFS)**
     - 가장 먼저 도착한 프로세스를 먼저 처리
     - 장점: 구현이 간단
     - 단점: 대기 시간이 길어질 수 있음
   - **Shortest Job Next (SJN) 또는 Shortest Job First (SJF)**
     - 가장 짧은 작업을 먼저 처리
     - 장점: 평균 대기 시간이 짧음
     - 단점: 긴 작업이 계속 대기할 수 있음 (기아 현상)
   - **Priority Scheduling**
     - 우선순위가 높은 프로세스를 먼저 처리
     - 장점: 중요한 작업을 먼저 처리
     - 단점: 낮은 우선순위 작업이 계속 대기할 수 있음 (기아 현상)
   - **Round Robin (RR)**
     - 일정 시간 간격으로 프로세스를 순환하며 처리
     - 장점: 응답 시간이 일정, 공정성 보장
     - 단점: 시간 간격 설정이 중요
   - **Multilevel Queue Scheduling**
     - 여러 큐를 사용하여 각 큐에 다른 스케줄링 알고리즘 적용
   - **Multilevel Feedback Queue Scheduling**
     - 프로세스의 특성에 따라 다른 큐로 이동 가능

3. **스케줄링 알고리즘 구현 및 시뮬레이션**
   - 각 스케줄링 알고리즘을 코드로 구현하여 시뮬레이션
   - 프로세스의 도착 시간, 실행 시간, 우선순위 등을 설정하여 결과 분석

**실습:**

1. **First-Come, First-Served (FCFS) 스케줄링 구현**

```c
#include <stdio.h>

typedef struct {
    int process_id;
    int arrival_time;
    int burst_time;
} Process;

void fcfs_scheduling(Process processes[], int n) {
    int wait_time[n], turn_around_time[n];
    int total_wait_time = 0, total_turn_around_time = 0;

    wait_time[0] = 0;
    for (int i = 1; i < n; i++) {
        wait_time[i] = processes[i - 1].burst_time + wait_time[i - 1];
    }

    for (int i = 0; i < n; i++) {
        turn_around_time[i] = processes[i].burst_time + wait_time[i];
        total_wait_time += wait_time[i];
        total_turn_around_time += turn_around_time[i];
    }

    printf("Process\tArrival Time\tBurst Time\tWait Time\tTurn-Around Time\n");
    for (int i = 0; i < n; i++) {
        printf("%d\t%d\t\t%d\t\t%d\t\t%d\n", processes[i].process_id, processes[i].arrival_time, processes[i].burst_time, wait_time[i], turn_around_time[i]);
    }

    printf("Average Wait Time: %.2f\n", (float)total_wait_time / n);
    printf("Average Turn-Around Time: %.2f\n", (float)total_turn_around_time / n);
}

int main() {
    Process processes[] = {{1, 0, 24}, {2, 1, 3}, {3, 2, 3}};
    int n = sizeof(processes) / sizeof(processes[0]);

    fcfs_scheduling(processes, n);
    return 0;
}
```

2. **Shortest Job First (SJF) 스케줄링 구현**

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int process_id;
    int arrival_time;
    int burst_time;
} Process;

int compare(const void *a, const void *b) {
    Process *p1 = (Process *)a;
    Process *p2 = (Process *)b;
    return p1->burst_time - p2->burst_time;
}

void sjf_scheduling(Process processes[], int n) {
    qsort(processes, n, sizeof(Process), compare);

    int wait_time[n], turn_around_time[n];
    int total_wait_time = 0, total_turn_around_time = 0;

    wait_time[0] = 0;
    for (int i = 1; i < n; i++) {
        wait_time[i] = processes[i - 1].burst_time + wait_time[i - 1];
    }

    for (int i = 0; i < n; i++) {
        turn_around_time[i] = processes[i].burst_time + wait_time[i];
        total_wait_time += wait_time[i];
        total_turn_around_time += turn_around_time[i];
    }

    printf("Process\tArrival Time\tBurst Time\tWait Time\tTurn-Around Time\n");
    for (int i = 0; i < n; i++) {
        printf("%d\t%d\t\t%d\t\t%d\t\t%d\n", processes[i].process_id, processes[i].arrival_time, processes[i].burst_time, wait_time[i], turn_around_time[i]);
    }

    printf("Average Wait Time: %.2f\n", (float)total_wait_time / n);
    printf("Average Turn-Around Time: %.2f\n", (float)total_turn_around_time / n);
}

int main() {
    Process processes[] = {{1, 0, 6}, {2, 1, 8}, {3, 2, 7}, {4, 3, 3}};
    int n = sizeof(processes) / sizeof(processes[0]);

    sjf_scheduling(processes, n);
    return 0;
}
```

**과제:**

1. **Priority Scheduling 구현**
   - 프로세스에 우선순위를 추가하고, 우선순위에 따라 스케줄링하는 알고리즘을 구현합니다.
   - 우선순위가 높은 프로세스를 먼저 실행하도록 합니다.

```c
#include <stdio.h>

typedef struct {
    int process_id;
    int arrival_time;
    int burst_time;
    int priority;
} Process;

void priority_scheduling(Process processes[], int n) {
    int wait_time[n], turn_around_time[n];
    int total_wait_time = 0, total_turn_around_time = 0;

    // 우선순위에 따라 정렬
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (processes[i].priority > processes[j].priority) {
                Process temp = processes[i];
                processes[i] = processes[j];
                processes[j] = temp;
            }
        }
    }

    wait_time[0] = 0;
    for (int i = 1; i < n; i++) {
        wait_time[i] = processes[i - 1].burst_time + wait_time[i - 1];
    }

    for (int i = 0; i < n; i++) {
        turn_around_time[i] = processes[i].burst_time + wait_time[i];
        total_wait_time += wait_time[i];
        total_turn_around_time += turn_around_time[i];
    }

    printf("Process\tArrival Time\tBurst Time\tPriority\tWait Time\tTurn-Around Time\n");
    for (int i = 0; i < n; i++) {
        printf("%d\t%d\t\t%d\t\t%d\t\t%d\t\t%d\n", processes[i].process_id, processes[i].arrival_time, processes[i].burst_time, processes[i].priority, wait_time[i], turn_around_time[i]);
    }

    printf("Average Wait Time: %.2f\n", (float)total_wait_time / n);
    printf("Average Turn-Around Time: %.2f\n", (float)total_turn_around_time / n);
}

int main() {
    Process processes[] = {{1, 0, 10, 3}, {2, 1, 1, 1}, {3, 2, 2, 4}, {4, 3, 1, 5}, {5, 4, 5, 2}};
    int n = sizeof(processes) / sizeof(processes[0]);

    priority_scheduling(processes, n);
    return 0;
}
```

2. **Round Robin (RR) 스케줄링 구현**
   - 타임 슬라이스(Time Slice)를 사용하여 프로세스를 순환하며 실행하는 Round Robin 알고리즘을 구현합니다.
   - 각 프로세스가 할당된 시간 동안 실행되고, 시간이 초과되면 다음 프로세스로 전환합니다.

```c
#include <stdio.h>

typedef struct {
    int process_id;
    int arrival_time;
    int

 burst_time;
    int remaining_time;
} Process;

void round_robin_scheduling(Process processes[], int n, int time_slice) {
    int wait_time[n], turn_around_time[n];
    int total_wait_time = 0, total_turn_around_time = 0;
    int time = 0;
    int completed = 0;

    for (int i = 0; i < n; i++) {
        wait_time[i] = 0;
        turn_around_time[i] = 0;
        processes[i].remaining_time = processes[i].burst_time;
    }

    while (completed < n) {
        for (int i = 0; i < n; i++) {
            if (processes[i].remaining_time > 0) {
                if (processes[i].remaining_time > time_slice) {
                    time += time_slice;
                    processes[i].remaining_time -= time_slice;
                } else {
                    time += processes[i].remaining_time;
                    wait_time[i] = time - processes[i].burst_time;
                    turn_around_time[i] = time;
                    processes[i].remaining_time = 0;
                    completed++;
                }
            }
        }
    }

    printf("Process\tArrival Time\tBurst Time\tWait Time\tTurn-Around Time\n");
    for (int i = 0; i < n; i++) {
        printf("%d\t%d\t\t%d\t\t%d\t\t%d\n", processes[i].process_id, processes[i].arrival_time, processes[i].burst_time, wait_time[i], turn_around_time[i]);
    }

    printf("Average Wait Time: %.2f\n", (float)total_wait_time / n);
    printf("Average Turn-Around Time: %.2f\n", (float)total_turn_around_time / n);
}

int main() {
    Process processes[] = {{1, 0, 24}, {2, 1, 3}, {3, 2, 3}};
    int n = sizeof(processes) / sizeof(processes[0]);
    int time_slice = 4;

    round_robin_scheduling(processes, n, time_slice);
    return 0;
}
```

**퀴즈 및 해설:**

1. **스케줄링의 주요 기준은 무엇인가요?**
   - 스케줄링의 주요 기준은 CPU 이용률, 처리율, 대기 시간, 응답 시간, 공정성입니다.

2. **FCFS 스케줄링 알고리즘의 단점은 무엇인가요?**
   - FCFS 스케줄링 알고리즘의 단점은 대기 시간이 길어질 수 있다는 점입니다. 특히, 긴 작업이 먼저 도착하면 다른 작업이 오랫동안 대기할 수 있습니다.

3. **Round Robin 스케줄링에서 타임 슬라이스의 역할은 무엇인가요?**
   - Round Robin 스케줄링에서 타임 슬라이스는 각 프로세스가 할당된 시간 동안 실행되는 시간을 의미합니다. 타임 슬라이스가 지나면 다음 프로세스로 전환됩니다. 이는 응답 시간을 일정하게 유지하고 공정성을 보장하는 데 사용됩니다.

**해설:**

1. **스케줄링의 주요 기준**은 CPU 이용률, 처리율, 대기 시간, 응답 시간, 공정성입니다. CPU 이용률은 CPU가 작업을 처리하는 비율을 의미하고, 처리율은 단위 시간당 완료된 작업의 수를 의미합니다. 대기 시간은 프로세스가 대기하는 시간을 의미하고, 응답 시간은 프로세스가 처음으로 응답하는 데 걸리는 시간을 의미합니다. 공정성은 모든 프로세스가 공평하게 CPU를 사용할 수 있도록 하는 것을 의미합니다.

2. **FCFS 스케줄링 알고리즘의 단점**은 대기 시간이 길어질 수 있다는 점입니다. 특히, 긴 작업이 먼저 도착하면 다른 작업이 오랫동안 대기할 수 있습니다. 이는 "콘보이 효과"로 알려져 있으며, 시스템 성능을 저하시킬 수 있습니다.

3. **Round Robin 스케줄링에서 타임 슬라이스의 역할**은 각 프로세스가 할당된 시간 동안 실행되는 시간을 의미합니다. 타임 슬라이스가 지나면 다음 프로세스로 전환됩니다. 이는 응답 시간을 일정하게 유지하고 공정성을 보장하는 데 사용됩니다. 타임 슬라이스가 너무 짧으면 문맥 전환 오버헤드가 증가할 수 있고, 너무 길면 응답 시간이 길어질 수 있습니다.

이로써 9주차 강의를 마무리합니다. 다음 주차에는 동기화 문제와 해결에 대해 학습합니다. 추가 질문이 있거나 특정 주제에 대해 더 알고 싶다면 언제든지 문의해 주세요.