### 3주차 강의 계획: 클래스 및 객체지향 프로그래밍

#### 강의 목표
- 클래스와 객체의 개념 이해
- 생성자와 소멸자의 역할 이해
- 상속 및 다형성의 기초 개념 습득

#### 강의 구성
- **강의 시간**: 2시간
- **구성**: 클래스와 객체 이론 (30분), 생성자와 소멸자 (30분), 상속 및 다형성 (30분), 실습 및 과제 안내 (30분)

#### 강의 내용

##### 1. 클래스와 객체 이론 (30분)

###### 1.1 클래스와 객체의 기본 개념
- **클래스 정의**:
  - 클래스의 구조 (멤버 변수, 멤버 함수)
  - 접근 지정자 (public, private, protected)
- **예제**:
```cpp
#include <iostream>
using namespace std;

class Person {
public:
    string name;
    int age;

    void introduce() {
        cout << "Name: " << name << ", Age: " << age << endl;
    }
};

int main() {
    Person person1;
    person1.name = "Alice";
    person1.age = 30;
    person1.introduce();
    return 0;
}
```

###### 1.2 객체 생성 및 사용
- **객체 생성**:
  - 클래스 인스턴스화
  - 객체의 멤버 변수 및 함수 접근

##### 2. 생성자와 소멸자 (30분)

###### 2.1 생성자의 역할
- **생성자**:
  - 생성자의 정의 및 사용
  - 기본 생성자와 매개변수가 있는 생성자
- **예제**:
```cpp
#include <iostream>
using namespace std;

class Person {
public:
    string name;
    int age;

    // 기본 생성자
    Person() {
        name = "Unknown";
        age = 0;
    }

    // 매개변수가 있는 생성자
    Person(string n, int a) {
        name = n;
        age = a;
    }

    void introduce() {
        cout << "Name: " << name << ", Age: " << age << endl;
    }
};

int main() {
    Person person1;
    Person person2("Bob", 25);
    person1.introduce();
    person2.introduce();
    return 0;
}
```

###### 2.2 소멸자의 역할
- **소멸자**:
  - 소멸자의 정의 및 사용
  - 객체가 소멸될 때 자원 해제
- **예제**:
```cpp
#include <iostream>
using namespace std;

class Person {
public:
    string name;
    int age;

    Person(string n, int a) {
        name = n;
        age = a;
    }

    ~Person() {
        cout << "Destructor called for " << name << endl;
    }

    void introduce() {
        cout << "Name: " << name << ", Age: " << age << endl;
    }
};

int main() {
    Person person1("Alice", 30);
    person1.introduce();
    return 0;
}
```

##### 3. 상속 및 다형성 (30분)

###### 3.1 상속의 기본 개념
- **상속**:
  - 상속의 정의 및 사용
  - 기본 클래스(Parent)와 파생 클래스(Child)
- **예제**:
```cpp
#include <iostream>
using namespace std;

class Animal {
public:
    void eat() {
        cout << "This animal is eating." << endl;
    }
};

class Dog : public Animal {
public:
    void bark() {
        cout << "The dog is barking." << endl;
    }
};

int main() {
    Dog myDog;
    myDog.eat();
    myDog.bark();
    return 0;
}
```

###### 3.2 다형성의 기본 개념
- **다형성**:
  - 함수 오버라이딩
  - 가상 함수와 순수 가상 함수
- **예제**:
```cpp
#include <iostream>
using namespace std;

class Animal {
public:
    virtual void sound() {
        cout << "This animal makes a sound." << endl;
    }
};

class Dog : public Animal {
public:
    void sound() override {
        cout << "The dog barks." << endl;
    }
};

class Cat : public Animal {
public:
    void sound() override {
        cout << "The cat meows." << endl;
    }
};

int main() {
    Animal* a1 = new Dog();
    Animal* a2 = new Cat();
    a1->sound();
    a2->sound();
    delete a1;
    delete a2;
    return 0;
}
```

##### 4. 실습 및 과제 안내 (30분)

###### 4.1 실습
- **실습 목표**:
  - 강의에서 다룬 내용을 직접 코드로 작성해보기
- **실습 문제**:
  - 클래스와 객체, 생성자와 소멸자, 상속과 다형성을 활용한 간단한 프로그램 작성

###### 4.2 과제 안내
- **과제 내용**:
  - `Person` 클래스를 확장하여 `Student`와 `Teacher` 클래스를 정의
  - 각 클래스는 고유의 멤버 변수와 함수를 포함
  - 상속 및 다형성을 활용하여 `Person` 클래스의 기능을 재사용
- **과제 제출 방법**:
  - 작성한 코드를 주차별 과제 제출 폴더에 업로드
  - 제출 기한: 다음 주 강의 전까지

##### 5. 퀴즈 및 해설 (30분)

###### 퀴즈 문제:
1. 클래스 내 멤버 변수와 함수를 외부에서 접근하지 못하도록 하는 접근 지정자는?
   - a) public
   - b) private
   - c) protected
   - d) internal
2. 생성자에 대한 설명 중 옳은 것은?
   - a) 생성자는 반환값을 가질 수 있다.
   - b) 생성자는 클래스와 이름이 다를 수 있다.
   - c) 생성자는 객체가 생성될 때 자동으로 호출된다.
   - d) 생성자는 명시적으로 호출해야 한다.
3. 다형성을 구현하는 데 사용되는 키워드는?
   - a) override
   - b) static
   - c) virtual
   - d) const

###### 퀴즈 해설:
1. **정답: b) private**
   - `private` 접근 지정자는 클래스 내 멤버 변수와 함수를 외부에서 접근하지 못하도록 합니다.
2. **정답: c) 생성자는 객체가 생성될 때 자동으로 호출된다.**
   - 생성자는 객체가 생성될 때 자동으로 호출됩니다.
3. **정답: c) virtual**
   - `virtual` 키워드는 다형성을 구현하는 데 사용되며, 이를 통해 함수 오버라이딩이 가능해집니다.

### 준비 자료
- 강의 슬라이드
- 실습 코드 예제
- 과제 안내서

### 과제 해설

##### 과제: `Person` 클래스를 확장하여 `Student`와 `Teacher` 클래스를 정의
- **문제**: `Person` 클래스를 기반으로 `Student`와 `Teacher` 클래스를 정의하고, 각 클래스는 고유의 멤버 변수와 함수를 포함하도록 작성
- **해설**:
  - `Student` 클래스와 `Teacher` 클래스는 `Person` 클래스를 상속받습니다.
  - 각 클래스는 생성자와 소멸자를 정의하고, 고유의 멤버 변수를 포함합니다.

```cpp
#include <iostream>
using namespace std;

class Person {
public:
    string name;
    int age;

    Person(string n, int a) : name(n), age(a) {}

    virtual void introduce() {
        cout << "Name: " << name << ", Age: " << age << endl;
    }

    virtual ~Person() {
        cout << "Destructor called for " << name << endl;
    }
};

class Student : public Person {
public:
    int studentID;

    Student(string n, int a, int id) : Person(n, a), studentID(id) {}

    void introduce() override {
        cout << "Student Name: " << name << ", Age: " << age << ", Student ID: " << studentID << endl;
    }
};

class Teacher : public Person {
public:
    string subject;

    Teacher(string n, int a, string s) : Person(n, a), subject(s) {}

    void introduce() override {
        cout << "Teacher Name: " << name << ", Age: " << age << ", Subject: " << subject << endl;
    }
};

int main() {
    Person* p1 = new Student("Alice", 20, 12345);
    Person* p2 = new Teacher("Bob", 45, "Mathematics");

    p1->introduce();
    p2->introduce();

    delete p1;
    delete p2;

    return 0;
}
```
- **설명**:
  - `Student` 클래스는 `Person` 클래스를 상속받아 `studentID` 멤버 변수를 추가합니다.
  - `Teacher` 클래스는 `Person` 클래스를 상속받아 `subject` 멤

버 변수를 추가합니다.
  - `introduce` 함수는 각 클래스에서 오버라이딩되어 고유의 멤버 변수를 포함하여 출력합니다.
  - 동적 메모리 할당과 소멸자 호출을 통해 자원 관리가 잘 되는지 확인합니다.

이로써 3주차 강의가 마무리됩니다. 학생들은 클래스와 객체지향 프로그래밍의 기본 개념을 이해하고, 이를 활용하여 다양한 프로그램을 작성하는 능력을 기르게 됩니다.