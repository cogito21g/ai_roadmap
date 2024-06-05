### 11주차: STL - 연관 컨테이너

#### 강의 목표
- 연관 컨테이너의 개념 이해 및 사용
- set, multiset, map, multimap의 사용법 이해
- 연관 컨테이너의 주요 메서드 및 활용법 익히기

#### 강의 내용

##### 1. set
- **set 선언 및 사용**

```cpp
#include <iostream>
#include <set>
using namespace std;

int main() {
    set<int> s;

    s.insert(1);
    s.insert(2);
    s.insert(3);
    s.insert(2);  // 중복 요소는 무시

    for (auto it = s.begin(); it != s.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    return 0;
}
```

- **set의 주요 메서드**

```cpp
#include <iostream>
#include <set>
using namespace std;

int main() {
    set<int> s = {1, 2, 3, 4, 5};

    cout << "Size: " << s.size() << endl;
    cout << "Contains 3: " << (s.count(3) ? "Yes" : "No") << endl;

    s.erase(3);
    cout << "After erasing 3, Size: " << s.size() << endl;

    return 0;
}
```

##### 2. multiset
- **multiset 선언 및 사용**

```cpp
#include <iostream>
#include <set>
using namespace std;

int main() {
    multiset<int> ms;

    ms.insert(1);
    ms.insert(2);
    ms.insert(3);
    ms.insert(2);  // 중복 요소 허용

    for (auto it = ms.begin(); it != ms.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    return 0;
}
```

- **multiset의 주요 메서드**

```cpp
#include <iostream>
#include <set>
using namespace std;

int main() {
    multiset<int> ms = {1, 2, 2, 3, 4, 5};

    cout << "Size: " << ms.size() << endl;
    cout << "Count of 2: " << ms.count(2) << endl;

    ms.erase(ms.find(2));
    cout << "After erasing one 2, Size: " << ms.size() << endl;

    return 0;
}
```

##### 3. map
- **map 선언 및 사용**

```cpp
#include <iostream>
#include <map>
using namespace std;

int main() {
    map<int, string> m;

    m[1] = "One";
    m[2] = "Two";
    m[3] = "Three";

    for (auto it = m.begin(); it != m.end(); ++it) {
        cout << it->first << ": " << it->second << endl;
    }

    return 0;
}
```

- **map의 주요 메서드**

```cpp
#include <iostream>
#include <map>
using namespace std;

int main() {
    map<int, string> m = {{1, "One"}, {2, "Two"}, {3, "Three"}};

    cout << "Size: " << m.size() << endl;
    cout << "Value of key 2: " << m[2] << endl;

    m.erase(2);
    cout << "After erasing key 2, Size: " << m.size() << endl;

    return 0;
}
```

##### 4. multimap
- **multimap 선언 및 사용**

```cpp
#include <iostream>
#include <map>
using namespace std;

int main() {
    multimap<int, string> mm;

    mm.insert(pair<int, string>(1, "One"));
    mm.insert(pair<int, string>(2, "Two"));
    mm.insert(pair<int, string>(2, "Dos"));  // 중복 키 허용

    for (auto it = mm.begin(); it != mm.end(); ++it) {
        cout << it->first << ": " << it->second << endl;
    }

    return 0;
}
```

- **multimap의 주요 메서드**

```cpp
#include <iostream>
#include <map>
using namespace std;

int main() {
    multimap<int, string> mm = {{1, "One"}, {2, "Two"}, {2, "Dos"}};

    cout << "Size: " << mm.size() << endl;
    cout << "Count of key 2: " << mm.count(2) << endl;

    auto range = mm.equal_range(2);
    cout << "Values of key 2: ";
    for (auto it = range.first; it != range.second; ++it) {
        cout << it->second << " ";
    }
    cout << endl;

    return 0;
}
```

#### 과제

1. **set 사용**
   - 정수를 입력받아 set에 저장하고, 중복 없이 정렬된 상태로 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <set>
using namespace std;

int main() {
    set<int> s;
    int num;

    cout << "Enter numbers (enter -1 to stop): ";
    while (true) {
        cin >> num;
        if (num == -1) {
            break;
        }
        s.insert(num);
    }

    cout << "Unique sorted numbers: ";
    for (int v : s) {
        cout << v << " ";
    }
    cout << endl;

    return 0;
}
```

2. **multiset 사용**
   - 정수를 입력받아 multiset에 저장하고, 중복된 값을 포함하여 정렬된 상태로 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <set>
using namespace std;

int main() {
    multiset<int> ms;
    int num;

    cout << "Enter numbers (enter -1 to stop): ";
    while (true) {
        cin >> num;
        if (num == -1) {
            break;
        }
        ms.insert(num);
    }

    cout << "Sorted numbers with duplicates: ";
    for (int v : ms) {
        cout << v << " ";
    }
    cout << endl;

    return 0;
}
```

3. **map 사용**
   - 단어와 그 뜻을 입력받아 map에 저장하고, 모든 단어와 뜻을 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <map>
using namespace std;

int main() {
    map<string, string> dictionary;
    string word, meaning;

    cout << "Enter word and meaning (enter 'exit' to stop): ";
    while (true) {
        cin >> word;
        if (word == "exit") {
            break;
        }
        cin >> meaning;
        dictionary[word] = meaning;
    }

    cout << "Dictionary contents: " << endl;
    for (auto it = dictionary.begin(); it != dictionary.end(); ++it) {
        cout << it->first << ": " << it->second << endl;
    }

    return 0;
}
```

4. **multimap 사용**
   - 학생의 이름과 점수를 입력받아 multimap에 저장하고, 동일한 이름의 학생이 있을 때 모두 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <map>
using namespace std;

int main() {
    multimap<string, int> students;
    string name;
    int score;

    cout << "Enter student name and score (enter 'exit' to stop): ";
    while (true) {
        cin >> name;
        if (name == "exit") {
            break;
        }
        cin >> score;
        students.insert(pair<string, int>(name, score));
    }

    cout << "Student scores: " << endl;
    for (auto it = students.begin(); it != students.end(); ++it) {
        cout << it->first << ": " << it->second << endl;
    }

    return 0;
}
```

#### 퀴즈

1. **연관 컨테이너에 대한 설명 중 맞는 것은?**
   - A) set은 중복된 요소를 허용한다.
   - B) multiset은 중복된 요소를 허용하지 않는다.
   - C) map은 키와 값을 쌍으로 저장한다.
   - D) multimap은 중복된 키를 허용하지 않는다.

2. **map에 대한 설명 중 맞는 것은?**
   - A) map은 키와 값을 저장하며, 키는 중복될 수 있다.
   - B) map은 요소를 자동으로 정렬하지 않는다.
   - C) map의 키는 중복되지 않으며, 값은 중복될 수 있다.
   - D) map은 해시 함수를 사용하여 요소를 저장한다.

3. **set과 multiset의 차이점에 대한 설명 중 맞는 것은?**
   - A) set은 중복된 요소를 허용하며, multiset은 허용하지 않는다.
   - B) set은 요소를 자동으로 정렬하지 않으며, multiset은 자동으로 정렬한다.
   - C) set은 중복된 요소를 허용하지 않으며, multiset은 허용한다.
   - D) set과 multiset은 동일한 방식으로 요소를 저장한다.

4. **multimap에 대한 설명 중 맞는 것은?

**
   - A) multimap은 키와 값을 저장하며, 키는 중복될 수 없다.
   - B) multimap은 요소를 자동으로 정렬하지 않는다.
   - C) multimap의 키와 값은 중복될 수 있다.
   - D) multimap은 연관 배열과 유사한 기능을 제공하지 않는다.

#### 퀴즈 해설

1. **연관 컨테이너에 대한 설명 중 맞는 것은?**
   - **정답: C) map은 키와 값을 쌍으로 저장한다.**
     - 해설: map은 키와 값을 쌍으로 저장하며, 키는 중복되지 않습니다.

2. **map에 대한 설명 중 맞는 것은?**
   - **정답: C) map의 키는 중복되지 않으며, 값은 중복될 수 있다.**
     - 해설: map의 키는 중복되지 않지만, 값은 중복될 수 있습니다.

3. **set과 multiset의 차이점에 대한 설명 중 맞는 것은?**
   - **정답: C) set은 중복된 요소를 허용하지 않으며, multiset은 허용한다.**
     - 해설: set은 중복된 요소를 허용하지 않지만, multiset은 중복된 요소를 허용합니다.

4. **multimap에 대한 설명 중 맞는 것은?**
   - **정답: C) multimap의 키와 값은 중복될 수 있다.**
     - 해설: multimap은 중복된 키와 값을 허용하며, 연관 배열과 유사한 기능을 제공합니다.

다음 주차 강의 내용을 요청하시면, 12주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.