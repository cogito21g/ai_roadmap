### 22주차: 실시간 시스템 및 스케줄링

**강의 목표:** 실시간 시스템의 개념과 중요성을 이해하고, 주요 실시간 스케줄링 알고리즘을 학습합니다. 또한, 실시간 스케줄링 시뮬레이션을 통해 실습합니다.

**강의 내용:**

1. **실시간 시스템 개념**
   - 실시간 시스템이란 무엇인가?
     - 시간 제약 조건을 만족해야 하는 시스템
   - 실시간 시스템의 응용 분야
     - 임베디드 시스템, 항공 우주, 의료 장비, 산업 제어 시스템 등
   - 실시간 시스템의 특성
     - 결정론적 응답, 높은 신뢰성, 시간 제약

2. **실시간 스케줄링 알고리즘**
   - 스케줄링의 중요성
     - 프로세스의 실행 순서를 결정하여 시스템의 효율성을 극대화
   - 고정 우선순위 스케줄링
     - Rate-Monotonic (RM)
     - 주기 시간이 짧은 태스크에 높은 우선순위 부여
   - 동적 우선순위 스케줄링
     - Earliest Deadline First (EDF)
     - 마감 시한이 가장 빠른 태스크에 높은 우선순위 부여
   - 스케줄링 알고리즘 비교
     - RM vs. EDF: 장단점 및 사용 사례

3. **실시간 스케줄링 시뮬레이션**
   - 시뮬레이션의 필요성
     - 스케줄링 알고리즘의 성능 및 효율성 평가
   - 시뮬레이션 도구 소개
     - 시뮬레이션을 위한 주요 도구 및 프레임워크
   - 시뮬레이션 구현
     - 간단한 실시간 스케줄링 시뮬레이션 구현

**실습:**

1. **Rate-Monotonic (RM) 스케줄링 시뮬레이션**

```python
import heapq

class Task:
    def __init__(self, id, period, execution_time):
        self.id = id
        self.period = period
        self.execution_time = execution_time
        self.next_deadline = period

def rm_schedule(tasks, simulation_time):
    current_time = 0
    task_queue = []
    for task in tasks:
        heapq.heappush(task_queue, (task.period, task))

    while current_time < simulation_time:
        if task_queue:
            period, task = heapq.heappop(task_queue)
            print(f"Time {current_time}: Task {task.id} is running")
            current_time += task.execution_time
            task.next_deadline += task.period
            heapq.heappush(task_queue, (task.next_deadline, task))
        else:
            current_time += 1

tasks = [
    Task(1, 5, 1),
    Task(2, 10, 2),
    Task(3, 15, 3)
]

rm_schedule(tasks, 30)
```

2. **Earliest Deadline First (EDF) 스케줄링 시뮬레이션**

```python
import heapq

class Task:
    def __init__(self, id, period, execution_time):
        self.id = id
        self.period = period
        self.execution_time = execution_time
        self.next_deadline = period

def edf_schedule(tasks, simulation_time):
    current_time = 0
    task_queue = []
    for task in tasks:
        heapq.heappush(task_queue, (task.next_deadline, task))

    while current_time < simulation_time:
        if task_queue:
            deadline, task = heapq.heappop(task_queue)
            print(f"Time {current_time}: Task {task.id} is running")
            current_time += task.execution_time
            task.next_deadline += task.period
            heapq.heappush(task_queue, (task.next_deadline, task))
        else:
            current_time += 1

tasks = [
    Task(1, 5, 1),
    Task(2, 10, 2),
    Task(3, 15, 3)
]

edf_schedule(tasks, 30)
```

**과제:**

1. **고급 실시간 스케줄링 알고리즘 구현**
   - Least Laxity First (LLF) 스케줄링 알고리즘을 구현합니다.
   - LLF 알고리즘의 특성과 장단점을 설명합니다.

2. **실시간 시스템 성능 평가**
   - RM과 EDF 스케줄링 알고리즘의 성능을 비교 분석합니다.
   - 주어진 태스크 세트에 대해 두 알고리즘의 성능을 시뮬레이션하고, 결과를 분석하여 보고서를 작성합니다.

**퀴즈 및 해설:**

1. **실시간 시스템의 주요 특성은 무엇인가요?**
   - 실시간 시스템의 주요 특성은 결정론적 응답, 높은 신뢰성, 시간 제약입니다.

2. **Rate-Monotonic (RM) 스케줄링의 주요 원칙은 무엇인가요?**
   - RM 스케줄링은 주기 시간이 짧은 태스크에 높은 우선순위를 부여하는 고정 우선순위 스케줄링 알고리즘입니다.

3. **Earliest Deadline First (EDF) 스케줄링의 주요 원칙은 무엇인가요?**
   - EDF 스케줄링은 마감 시한이 가장 빠른 태스크에 높은 우선순위를 부여하는 동적 우선순위 스케줄링 알고리즘입니다.

**해설:**

1. **실시간 시스템의 주요 특성**은 결정론적 응답, 높은 신뢰성, 시간 제약입니다. 실시간 시스템은 주어진 시간 내에 작업을 완료해야 하며, 이러한 특성 때문에 높은 신뢰성과 정확한 시간 제어가 필요합니다.

2. **Rate-Monotonic (RM) 스케줄링의 주요 원칙**은 주기 시간이 짧은 태스크에 높은 우선순위를 부여하는 것입니다. 이는 고정 우선순위 스케줄링 알고리즘으로, 주기가 짧은 태스크가 더 자주 실행되어야 하는 상황에 적합합니다.

3. **Earliest Deadline First (EDF) 스케줄링의 주요 원칙**은 마감 시한이 가장 빠른 태스크에 높은 우선순위를 부여하는 것입니다. 이는 동적 우선순위 스케줄링 알고리즘으로, 각 태스크의 마감 시한을 고려하여 효율적으로 스케줄링을 수행합니다.

이로써 22주차 강의를 마무리합니다. 다음 주차에는 팀 프로젝트 진행 상황 점검 및 중간 발표 준비를 학습합니다. 추가 질문이 있거나 특정 주제에 대해 더 알고 싶다면 언제든지 문의해 주세요.