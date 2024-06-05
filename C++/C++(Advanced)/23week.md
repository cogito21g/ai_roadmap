### 23주차: 멀티스레딩 심화

#### 강의 목표
- C++20 코루틴 이해 및 사용
- 고급 동기화 기법 이해 및 사용
- 비동기 프로그래밍 이해 및 사용

#### 강의 내용

##### 1. C++20 코루틴
- **코루틴 기본 사용법**

```cpp
#include <iostream>
#include <coroutine>
using namespace std;

struct Generator {
    struct promise_type;
    using handle_type = coroutine_handle<promise_type>;

    struct promise_type {
        int value;
        coroutine_handle<> continuation;

        auto get_return_object() {
            return Generator{handle_type::from_promise(*this)};
        }
        auto initial_suspend() {
            return suspend_always{};
        }
        auto final_suspend() noexcept {
            return suspend_always{};
        }
        void unhandled_exception() {
            std::terminate();
        }
        auto yield_value(int value) {
            this->value = value;
            return suspend_always{};
        }
        void return_void() {}
    };

    handle_type handle;

    Generator(handle_type h) : handle(h) {}
    ~Generator() {
        if (handle) handle.destroy();
    }

    bool move_next() {
        handle.resume();
        return !handle.done();
    }

    int current_value() {
        return handle.promise().value;
    }
};

Generator counter() {
    for (int i = 0; i < 5; ++i) {
        co_yield i;
    }
}

int main() {
    auto gen = counter();
    while (gen.move_next()) {
        cout << gen.current_value() << endl;
    }
    return 0;
}
```

##### 2. 고급 동기화 기법
- **락 프리 프로그래밍**

```cpp
#include <iostream>
#include <atomic>
#include <thread>
#include <vector>
using namespace std;

atomic<int> counter(0);

void increment(int n) {
    for (int i = 0; i < n; ++i) {
        counter.fetch_add(1, memory_order_relaxed);
    }
}

int main() {
    const int num_threads = 10;
    const int num_increments = 1000;
    vector<thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(increment, num_increments);
    }

    for (auto& t : threads) {
        t.join();
    }

    cout << "Final counter value: " << counter << endl;
    return 0;
}
```

- **조건 변수**

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
using namespace std;

mutex mtx;
condition_variable cv;
bool ready = false;

void print_id(int id) {
    unique_lock<mutex> lck(mtx);
    while (!ready) cv.wait(lck);
    cout << "Thread " << id << endl;
}

void go() {
    unique_lock<mutex> lck(mtx);
    ready = true;
    cv.notify_all();
}

int main() {
    thread threads[10];
    for (int i = 0; i < 10; ++i) {
        threads[i] = thread(print_id, i);
    }

    cout << "10 threads ready to race..." << endl;
    this_thread::sleep_for(chrono::seconds(1));
    go();

    for (auto& t : threads) {
        t.join();
    }
    return 0;
}
```

##### 3. 비동기 프로그래밍
- **std::future와 std::async 사용**

```cpp
#include <iostream>
#include <future>
using namespace std;

int findAnswer() {
    this_thread::sleep_for(chrono::seconds(3));
    return 42;
}

int main() {
    cout << "Getting the answer..." << endl;
    future<int> answer = async(findAnswer);
    cout << "Answer: " << answer.get() << endl;
    return 0;
}
```

- **std::promise와 std::future 사용**

```cpp
#include <iostream>
#include <thread>
#include <future>
using namespace std;

void calculate(promise<int> prom) {
    this_thread::sleep_for(chrono::seconds(2));
    prom.set_value(42);
}

int main() {
    promise<int> prom;
    future<int> fut = prom.get_future();
    thread t(calculate, move(prom));

    cout << "Waiting for result..." << endl;
    cout << "Result: " << fut.get() << endl;

    t.join();
    return 0;
}
```

#### 과제

1. **C++20 코루틴을 사용하여 피보나치 수열을 생성하는 프로그램을 작성하세요.**

```cpp
#include <iostream>
#include <coroutine>
using namespace std;

struct Generator {
    struct promise_type;
    using handle_type = coroutine_handle<promise_type>;

    struct promise_type {
        int value;
        coroutine_handle<> continuation;

        auto get_return_object() {
            return Generator{handle_type::from_promise(*this)};
        }
        auto initial_suspend() {
            return suspend_always{};
        }
        auto final_suspend() noexcept {
            return suspend_always{};
        }
        void unhandled_exception() {
            std::terminate();
        }
        auto yield_value(int value) {
            this->value = value;
            return suspend_always{};
        }
        void return_void() {}
    };

    handle_type handle;

    Generator(handle_type h) : handle(h) {}
    ~Generator() {
        if (handle) handle.destroy();
    }

    bool move_next() {
        handle.resume();
        return !handle.done();
    }

    int current_value() {
        return handle.promise().value;
    }
};

Generator fibonacci() {
    int a = 0, b = 1;
    while (true) {
        co_yield a;
        int temp = a;
        a = b;
        b = temp + b;
    }
}

int main() {
    auto gen = fibonacci();
    for (int i = 0; i < 10; ++i) {
        gen.move_next();
        cout << gen.current_value() << " ";
    }
    cout << endl;
    return 0;
}
```

2. **락 프리 프로그래밍을 사용하여 공유 변수를 안전하게 증가시키는 프로그램을 작성하세요.**

```cpp
#include <iostream>
#include <atomic>
#include <thread>
#include <vector>
using namespace std;

atomic<int> counter(0);

void increment(int n) {
    for (int i = 0; i < n; ++i) {
        counter.fetch_add(1, memory_order_relaxed);
    }
}

int main() {
    const int num_threads = 10;
    const int num_increments = 1000;
    vector<thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(increment, num_increments);
    }

    for (auto& t : threads) {
        t.join();
    }

    cout << "Final counter value: " << counter << endl;
    return 0;
}
```

3. **조건 변수를 사용하여 여러 스레드가 신호를 기다렸다가 동시에 실행되도록 하는 프로그램을 작성하세요.**

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
using namespace std;

mutex mtx;
condition_variable cv;
bool ready = false;

void print_id(int id) {
    unique_lock<mutex> lck(mtx);
    while (!ready) cv.wait(lck);
    cout << "Thread " << id << endl;
}

void go() {
    unique_lock<mutex> lck(mtx);
    ready = true;
    cv.notify_all();
}

int main() {
    thread threads[10];
    for (int i = 0; i < 10; ++i) {
        threads[i] = thread(print_id, i);
    }

    cout << "10 threads ready to race..." << endl;
    this_thread::sleep_for(chrono::seconds(1));
    go();

    for (auto& t : threads) {
        t.join();
    }
    return 0;
}
```

4. **std::future와 std::async를 사용하여 비동기적으로 작업을 수행하는 프로그램을 작성하세요.**

```cpp
#include <iostream>
#include <future>
using namespace std;

int findAnswer() {
    this_thread::sleep_for(chrono::seconds(3));
    return 42;
}

int main() {
    cout << "Getting the answer..." << endl;
    future<int> answer = async(findAnswer);
    cout << "Answer: " << answer.get() << endl;
    return 0;
}
```

#### 퀴즈

1. **C++20 코루틴에 대한 설명 중 맞는 것은?**
   - A) 코루틴은 항상 동기적으로 실행된다.
   - B) 코루틴은 일시 중단하고 다시 시작할 수 있다.
   - C) 코루틴은 함수 호출과 동일하게 동작한다.
   - D) 코루틴은 멀티스레딩과 관련이 없다.

2. **락 프리 프로그래밍에 대한 설명 중 맞는 것은?**
   - A) 락 프리 프로그래밍은 항상 안전하다.
   - B) 락 프리 프로그래밍은 데이터 경합을 피할 수 있다.
   - C) 락 프리 프로그래밍은 항상 더 빠르다.
   - D) 락 프리 프로그래밍은 동

기화 메커니즘을 사용하지 않는다.

3. **조건 변수의 사용 목적은?**
   - A) 스레드 간의 통신을 위해 사용된다.
   - B) 메모리 할당을 최적화하기 위해 사용된다.
   - C) 파일 입출력을 관리하기 위해 사용된다.
   - D) 네트워크 통신을 최적화하기 위해 사용된다.

4. **std::future와 std::async에 대한 설명 중 맞는 것은?**
   - A) std::future는 비동기 작업의 결과를 나타낸다.
   - B) std::async는 항상 동기적으로 작업을 수행한다.
   - C) std::future는 스레드 간의 통신을 제공하지 않는다.
   - D) std::async는 스레드 생성과 관련이 없다.

#### 퀴즈 해설

1. **C++20 코루틴에 대한 설명 중 맞는 것은?**
   - **정답: B) 코루틴은 일시 중단하고 다시 시작할 수 있다.**
     - 해설: 코루틴은 실행을 일시 중단하고 나중에 다시 시작할 수 있는 함수입니다.

2. **락 프리 프로그래밍에 대한 설명 중 맞는 것은?**
   - **정답: B) 락 프리 프로그래밍은 데이터 경합을 피할 수 있다.**
     - 해설: 락 프리 프로그래밍은 잠금을 사용하지 않고 데이터 경합을 피하기 위해 원자적 연산을 사용합니다.

3. **조건 변수의 사용 목적은?**
   - **정답: A) 스레드 간의 통신을 위해 사용된다.**
     - 해설: 조건 변수는 스레드 간의 통신을 위한 동기화 메커니즘입니다.

4. **std::future와 std::async에 대한 설명 중 맞는 것은?**
   - **정답: A) std::future는 비동기 작업의 결과를 나타낸다.**
     - 해설: std::future는 비동기 작업의 결과를 나타내며, std::async는 비동기적으로 작업을 수행하는 데 사용됩니다.

다음 주차 강의 내용을 요청하시면, 24주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.