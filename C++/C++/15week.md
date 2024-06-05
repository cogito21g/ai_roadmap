### 15주차: 멀티스레딩

#### 강의 목표
- 멀티스레딩의 개념 이해
- C++11 스레드 사용법 이해
- 동기화 (mutex, lock)의 개념 및 사용법 이해
- 스레드 간 통신 (condition_variable) 이해 및 사용

#### 강의 내용

##### 1. C++11 스레드 기초 (std::thread)
- **기본 스레드 생성 및 사용**

```cpp
#include <iostream>
#include <thread>
using namespace std;

void printMessage(const string& message) {
    cout << message << endl;
}

int main() {
    thread t1(printMessage, "Hello from thread!");
    t1.join();  // 스레드가 완료될 때까지 대기

    return 0;
}
```

- **스레드에서 함수 객체 사용**

```cpp
#include <iostream>
#include <thread>
using namespace std;

class MessagePrinter {
public:
    void operator()(const string& message) {
        cout << message << endl;
    }
};

int main() {
    MessagePrinter printer;
    thread t1(printer, "Hello from thread object!");
    t1.join();

    return 0;
}
```

##### 2. 동기화 (mutex, lock)
- **mutex 사용**

```cpp
#include <iostream>
#include <thread>
#include <mutex>
using namespace std;

mutex mtx;

void printNumbers(int n) {
    mtx.lock();
    for (int i = 0; i < n; ++i) {
        cout << i << " ";
    }
    cout << endl;
    mtx.unlock();
}

int main() {
    thread t1(printNumbers, 10);
    thread t2(printNumbers, 10);

    t1.join();
    t2.join();

    return 0;
}
```

- **lock_guard 사용**

```cpp
#include <iostream>
#include <thread>
#include <mutex>
using namespace std;

mutex mtx;

void printNumbers(int n) {
    lock_guard<mutex> guard(mtx);
    for (int i = 0; i < n; ++i) {
        cout << i << " ";
    }
    cout << endl;
}

int main() {
    thread t1(printNumbers, 10);
    thread t2(printNumbers, 10);

    t1.join();
    t2.join();

    return 0;
}
```

##### 3. 스레드 간 통신 (condition_variable)
- **condition_variable 사용**

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
using namespace std;

mutex mtx;
condition_variable cv;
bool ready = false;

void printNumbers() {
    unique_lock<mutex> lock(mtx);
    cv.wait(lock, []{ return ready; });

    for (int i = 0; i < 10; ++i) {
        cout << i << " ";
    }
    cout << endl;
}

void setReady() {
    {
        lock_guard<mutex> guard(mtx);
        ready = true;
    }
    cv.notify_all();
}

int main() {
    thread t1(printNumbers);
    thread t2(setReady);

    t1.join();
    t2.join();

    return 0;
}
```

#### 과제

1. **스레드 사용**
   - 두 개의 스레드를 생성하여 각각 다른 메시지를 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <thread>
using namespace std;

void printMessage(const string& message) {
    cout << message << endl;
}

int main() {
    thread t1(printMessage, "Hello from thread 1!");
    thread t2(printMessage, "Hello from thread 2!");

    t1.join();
    t2.join();

    return 0;
}
```

2. **mutex 사용**
   - 두 개의 스레드가 동일한 함수에서 동시에 실행될 때 데이터 경쟁을 방지하기 위해 mutex를 사용하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <thread>
#include <mutex>
using namespace std;

mutex mtx;

void increment(int& counter) {
    for (int i = 0; i < 1000; ++i) {
        lock_guard<mutex> guard(mtx);
        ++counter;
    }
}

int main() {
    int counter = 0;
    thread t1(increment, ref(counter));
    thread t2(increment, ref(counter));

    t1.join();
    t2.join();

    cout << "Final counter value: " << counter << endl;
    return 0;
}
```

3. **condition_variable 사용**
   - condition_variable을 사용하여 한 스레드가 다른 스레드의 신호를 기다리도록 하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
using namespace std;

mutex mtx;
condition_variable cv;
bool ready = false;

void waitForSignal() {
    unique_lock<mutex> lock(mtx);
    cv.wait(lock, []{ return ready; });
    cout << "Signal received. Proceeding..." << endl;
}

void sendSignal() {
    {
        lock_guard<mutex> guard(mtx);
        ready = true;
    }
    cv.notify_one();
}

int main() {
    thread t1(waitForSignal);
    thread t2(sendSignal);

    t1.join();
    t2.join();

    return 0;
}
```

#### 퀴즈

1. **스레드에 대한 설명 중 맞는 것은?**
   - A) 스레드는 항상 동시에 실행된다.
   - B) 스레드는 프로세스 내에서 독립적으로 실행된다.
   - C) 스레드는 서로 다른 프로세스에서 실행될 수 없다.
   - D) 스레드는 항상 순차적으로 실행된다.

2. **mutex에 대한 설명 중 맞는 것은?**
   - A) mutex는 데이터 경쟁을 방지하기 위해 사용된다.
   - B) mutex는 스레드 간 통신을 위해 사용된다.
   - C) mutex는 항상 락을 걸지 않아도 된다.
   - D) mutex는 여러 스레드에서 동시에 잠글 수 있다.

3. **condition_variable에 대한 설명 중 맞는 것은?**
   - A) condition_variable은 스레드 간 통신을 위해 사용된다.
   - B) condition_variable은 데이터 경쟁을 방지하기 위해 사용된다.
   - C) condition_variable은 mutex와 함께 사용할 수 없다.
   - D) condition_variable은 항상 자동으로 신호를 보낸다.

4. **lock_guard에 대한 설명 중 맞는 것은?**
   - A) lock_guard는 수동으로 락을 해제해야 한다.
   - B) lock_guard는 자동으로 락을 걸고 해제한다.
   - C) lock_guard는 스레드 간 통신을 위해 사용된다.
   - D) lock_guard는 항상 mutex와 함께 사용할 수 없다.

#### 퀴즈 해설

1. **스레드에 대한 설명 중 맞는 것은?**
   - **정답: B) 스레드는 프로세스 내에서 독립적으로 실행된다.**
     - 해설: 스레드는 동일한 프로세스 내에서 독립적으로 실행되는 작업 단위입니다.

2. **mutex에 대한 설명 중 맞는 것은?**
   - **정답: A) mutex는 데이터 경쟁을 방지하기 위해 사용된다.**
     - 해설: mutex는 여러 스레드가 동일한 자원에 접근할 때 데이터 경쟁을 방지하기 위해 사용됩니다.

3. **condition_variable에 대한 설명 중 맞는 것은?**
   - **정답: A) condition_variable은 스레드 간 통신을 위해 사용된다.**
     - 해설: condition_variable은 한 스레드가 특정 조건을 기다리는 동안 다른 스레드로부터 신호를 받을 수 있도록 하는 스레드 간 통신 도구입니다.

4. **lock_guard에 대한 설명 중 맞는 것은?**
   - **정답: B) lock_guard는 자동으로 락을 걸고 해제한다.**
     - 해설: lock_guard는 스코프를 벗어날 때 자동으로 락을 해제하여 자원의 안전한 사용을 보장합니다.

다음 주차 강의 내용을 요청하시면, 16주차 강의 내용, 과제 및