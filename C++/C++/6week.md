### 6주차: 구조체와 열거형

#### 강의 목표
- 구조체의 개념과 사용법 이해
- 열거형의 개념과 사용법 이해
- 공용체의 개념과 사용법 이해

#### 강의 내용

##### 1. 구조체 정의와 사용
- **구조체 선언 및 초기화**

```cpp
#include <iostream>
using namespace std;

struct Person {
    string name;
    int age;
    float height;
};

int main() {
    Person person1 = {"John Doe", 30, 5.9};

    cout << "Name: " << person1.name << endl;
    cout << "Age: " << person1.age << endl;
    cout << "Height: " << person1.height << endl;

    return 0;
}
```

- **구조체 배열 및 포인터**

```cpp
#include <iostream>
using namespace std;

struct Person {
    string name;
    int age;
    float height;
};

int main() {
    Person people[2] = {{"John Doe", 30, 5.9}, {"Jane Smith", 25, 5.7}};
    Person* ptr = people;

    for (int i = 0; i < 2; i++) {
        cout << "Name: " << ptr[i].name << endl;
        cout << "Age: " << ptr[i].age << endl;
        cout << "Height: " << ptr[i].height << endl;
    }

    return 0;
}
```

##### 2. 열거형 (enum) 사용
- **열거형 선언 및 사용**

```cpp
#include <iostream>
using namespace std;

enum Color {RED, GREEN, BLUE};

int main() {
    Color myColor = RED;

    if (myColor == RED) {
        cout << "The color is red." << endl;
    } else if (myColor == GREEN) {
        cout << "The color is green." << endl;
    } else {
        cout << "The color is blue." << endl;
    }

    return 0;
}
```

- **열거형 값 변경**

```cpp
#include <iostream>
using namespace std;

enum Color {RED, GREEN, BLUE};

int main() {
    Color myColor = GREEN;

    switch (myColor) {
        case RED:
            cout << "The color is red." << endl;
            break;
        case GREEN:
            cout << "The color is green." << endl;
            break;
        case BLUE:
            cout << "The color is blue." << endl;
            break;
    }

    return 0;
}
```

##### 3. 공용체 (union)
- **공용체 선언 및 사용**

```cpp
#include <iostream>
using namespace std;

union Data {
    int intValue;
    float floatValue;
    char charValue;
};

int main() {
    Data data;
    data.intValue = 42;
    cout << "Integer: " << data.intValue << endl;

    data.floatValue = 3.14;
    cout << "Float: " << data.floatValue << endl;

    data.charValue = 'A';
    cout << "Character: " << data.charValue << endl;

    return 0;
}
```

#### 과제

1. **구조체 사용**
   - 학생(Student) 구조체를 선언하고, 이름, 학년, 평균 점수를 저장하는 변수를 포함하세요. 학생 정보 3개를 입력받아 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
using namespace std;

struct Student {
    string name;
    int grade;
    float average;
};

int main() {
    Student students[3];

    for (int i = 0; i < 3; i++) {
        cout << "Enter details for student " << i + 1 << endl;
        cout << "Name: ";
        cin >> students[i].name;
        cout << "Grade: ";
        cin >> students[i].grade;
        cout << "Average: ";
        cin >> students[i].average;
    }

    for (int i = 0; i < 3; i++) {
        cout << "Details of student " << i + 1 << ":" << endl;
        cout << "Name: " << students[i].name << endl;
        cout << "Grade: " << students[i].grade << endl;
        cout << "Average: " << students[i].average << endl;
    }

    return 0;
}
```

2. **열거형 사용**
   - 요일을 나타내는 열거형 Day를 선언하고, 사용자로부터 요일을 입력받아 해당 요일을 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
using namespace std;

enum Day {SUNDAY, MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY};

int main() {
    int input;
    cout << "Enter a day (0 for Sunday, 1 for Monday, ..., 6 for Saturday): ";
    cin >> input;

    Day today = static_cast<Day>(input);

    switch (today) {
        case SUNDAY:
            cout << "Today is Sunday." << endl;
            break;
        case MONDAY:
            cout << "Today is Monday." << endl;
            break;
        case TUESDAY:
            cout << "Today is Tuesday." << endl;
            break;
        case WEDNESDAY:
            cout << "Today is Wednesday." << endl;
            break;
        case THURSDAY:
            cout << "Today is Thursday." << endl;
            break;
        case FRIDAY:
            cout << "Today is Friday." << endl;
            break;
        case SATURDAY:
            cout << "Today is Saturday." << endl;
            break;
    }

    return 0;
}
```

3. **공용체 사용**
   - 공용체 Data를 선언하고, 정수, 실수, 문자를 저장할 수 있는 변수를 포함하세요. 공용체 변수를 사용하여 정수, 실수, 문자를 순서대로 저장하고 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
using namespace std;

union Data {
    int intValue;
    float floatValue;
    char charValue;
};

int main() {
    Data data;

    data.intValue = 42;
    cout << "Integer: " << data.intValue << endl;

    data.floatValue = 3.14;
    cout << "Float: " << data.floatValue << endl;

    data.charValue = 'A';
    cout << "Character: " << data.charValue << endl;

    return 0;
}
```

#### 퀴즈

1. **구조체에 대한 설명 중 틀린 것은?**
   - A) 구조체는 서로 다른 데이터 타입을 저장할 수 있다.
   - B) 구조체는 클래스와 비슷하지만, 기본 접근 지정자가 public이다.
   - C) 구조체는 포인터로 선언할 수 없다.
   - D) 구조체 배열을 선언할 수 있다.

2. **열거형에 대한 설명 중 맞는 것은?**
   - A) 열거형은 문자열 값을 가질 수 있다.
   - B) 열거형의 값은 기본적으로 1부터 시작한다.
   - C) 열거형은 코드의 가독성을 높이기 위해 사용된다.
   - D) 열거형은 다른 데이터 타입과 혼합하여 사용할 수 없다.

3. **공용체에 대한 설명 중 맞는 것은?**
   - A) 공용체는 한 번에 여러 값을 저장할 수 있다.
   - B) 공용체의 크기는 가장 큰 멤버의 크기와 같다.
   - C) 공용체는 다른 데이터 타입의 멤버를 동시에 사용할 수 있다.
   - D) 공용체는 클래스와 비슷하지만, 기본 접근 지정자가 private이다.

#### 퀴즈 해설

1. **구조체에 대한 설명 중 틀린 것은?**
   - **정답: C) 구조체는 포인터로 선언할 수 없다.**
     - 해설: 구조체는 포인터로 선언할 수 있으며, 구조체 포인터를 통해 구조체 멤버에 접근할 수 있습니다.

2. **열거형에 대한 설명 중 맞는 것은?**
   - **정답: C) 열거형은 코드의 가독성을 높이기 위해 사용된다.**
     - 해설: 열거형은 코드의 가독성을 높이고, 상수 값을 더 이해하기 쉽게 하기 위해 사용됩니다. 기본 값은 0부터 시작합니다.

3. **공용체에 대한 설명 중 맞는 것은?**
   - **정답: B) 공용체의 크기는 가장 큰 멤버의 크기와 같다.**
     - 해설: 공용체는 가장 큰 멤버의 크기와 같으며, 한 번에 하나의 멤버만 저장할 수 있습니다. 

다음 주차 강의 내용을 요청하시면, 7주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.