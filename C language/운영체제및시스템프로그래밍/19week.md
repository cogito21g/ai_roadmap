### 19주차: 리눅스 커널 심화

**강의 목표:** 리눅스 커널의 동기화 기법, 메모리 관리, 커널 스케줄러의 동작 원리를 이해하고, 커널 스케줄러 모듈을 작성 및 분석하는 방법을 학습합니다.

**강의 내용:**

1. **리눅스 커널 동기화 기법**
   - 동기화의 필요성
     - 여러 프로세스나 스레드가 공유 자원에 접근할 때 데이터 일관성을 유지하기 위해 필요
   - 주요 동기화 기법
     - 스핀락 (Spinlock)
     - 세마포어 (Semaphore)
     - 뮤텍스 (Mutex)
     - 리드/라이트 락 (Read/Write Lock)

2. **커널 메모리 관리**
   - 메모리 관리의 개요
     - 커널에서의 메모리 할당과 해제
   - 메모리 할당 기법
     - 페이지 할당 (Page Allocation)
     - 슬랩 할당자 (Slab Allocator)
   - 메모리 풀 (Memory Pool)
     - 메모리 풀의 개념과 활용
   - 메모리 매핑
     - 가상 메모리와 물리 메모리 매핑

3. **커널 스케줄러**
   - 스케줄러의 역할
     - CPU 시간의 효율적인 분배
   - 주요 스케줄링 알고리즘
     - Completely Fair Scheduler (CFS)
     - Real-Time Scheduling
   - 스케줄링 정책
     - 우선순위 기반 스케줄링
     - 라운드 로빈 (Round Robin)

**실습:**

1. **커널 동기화 기법 실습**

**스핀락 예제:**

```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/spinlock.h>

static spinlock_t my_spinlock;
static int shared_data = 0;

static int __init spinlock_init(void) {
    spin_lock_init(&my_spinlock);
    printk(KERN_INFO "Spinlock module loaded\n");

    spin_lock(&my_spinlock);
    shared_data++;
    printk(KERN_INFO "Shared data: %d\n", shared_data);
    spin_unlock(&my_spinlock);

    return 0;
}

static void __exit spinlock_exit(void) {
    printk(KERN_INFO "Spinlock module unloaded\n");
}

module_init(spinlock_init);
module_exit(spinlock_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("A simple spinlock example");
MODULE_VERSION("0.1");
```

2. **커널 메모리 할당 실습**

**페이지 할당 예제:**

```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/gfp.h>
#include <linux/mm.h>

static int __init mem_alloc_init(void) {
    struct page *page;
    void *page_addr;

    page = alloc_pages(GFP_KERNEL, 0); // 0은 2^0 페이지 수 (1 페이지)
    if (!page) {
        printk(KERN_ERR "Page allocation failed\n");
        return -ENOMEM;
    }

    page_addr = page_address(page);
    printk(KERN_INFO "Page allocated at address: %p\n", page_addr);

    __free_pages(page, 0);

    return 0;
}

static void __exit mem_alloc_exit(void) {
    printk(KERN_INFO "Memory allocation module unloaded\n");
}

module_init(mem_alloc_init);
module_exit(mem_alloc_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("A simple memory allocation example");
MODULE_VERSION("0.1");
```

3. **커널 스케줄러 모듈 작성 및 분석**

**간단한 스케줄러 모듈 예제:**

```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/sched/signal.h>

static int __init scheduler_init(void) {
    struct task_struct *task;

    for_each_process(task) {
        printk(KERN_INFO "Process: %s [PID: %d]\n", task->comm, task->pid);
    }

    return 0;
}

static void __exit scheduler_exit(void) {
    printk(KERN_INFO "Scheduler module unloaded\n");
}

module_init(scheduler_init);
module_exit(scheduler_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("A simple scheduler example");
MODULE_VERSION("0.1");
```

**과제:**

1. **고급 커널 동기화 기법 실습**
   - 커널 모듈에서 세마포어를 사용하여 동기화를 구현합니다.
   - 두 개의 스레드가 공유 자원에 접근할 때 세마포어를 사용하여 동기화합니다.

2. **커널 메모리 관리 기법 비교 분석**
   - 페이지 할당과 슬랩 할당자를 비교 분석합니다.
   - 각각의 장단점과 사용 사례를 조사하고, 적합한 시나리오를 제시합니다.

**퀴즈 및 해설:**

1. **리눅스 커널에서 동기화가 필요한 이유는 무엇인가요?**
   - 여러 프로세스나 스레드가 공유 자원에 접근할 때 데이터의 일관성을 유지하기 위해 필요합니다.

2. **페이지 할당과 슬랩 할당자의 차이점은 무엇인가요?**
   - 페이지 할당은 페이지 단위로 메모리를 할당하는 반면, 슬랩 할당자는 작은 객체의 메모리 할당에 최적화되어 있습니다.

3. **CFS 스케줄러의 주요 특징은 무엇인가요?**
   - CFS (Completely Fair Scheduler)는 태스크의 실행 시간을 공평하게 분배하여 모든 태스크가 공정하게 CPU 시간을 받을 수 있도록 합니다.

**해설:**

1. **리눅스 커널에서 동기화가 필요한 이유**는 여러 프로세스나 스레드가 공유 자원에 접근할 때 데이터의 일관성을 유지하기 위해서입니다. 동기화 없이 여러 스레드가 동시에 데이터를 변경하면 데이터 일관성이 깨져 오류가 발생할 수 있습니다.

2. **페이지 할당과 슬랩 할당자의 차이점**은 페이지 할당은 페이지 단위로 메모리를 할당하는 반면, 슬랩 할당자는 작은 객체의 메모리 할당에 최적화되어 있다는 점입니다. 페이지 할당은 큰 메모리 블록을 할당하는 데 적합하고, 슬랩 할당자는 빈번한 작은 메모리 할당과 해제를 효율적으로 처리할 수 있습니다.

3. **CFS 스케줄러의 주요 특징**은 태스크의 실행 시간을 공평하게 분배하여 모든 태스크가 공정하게 CPU 시간을 받을 수 있도록 하는 것입니다. CFS는 각 태스크의 실행 시간을 균등하게 분배하고, 우선순위와 가중치를 사용하여 스케줄링을 조정합니다.

이로써 19주차 강의를 마무리합니다. 다음 주차에는 팀 프로젝트 진행 상황 점검 및 중간 발표 준비를 학습합니다. 추가 질문이 있거나 특정 주제에 대해 더 알고 싶다면 언제든지 문의해 주세요.