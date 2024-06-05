### 7주차: 클래스와 객체지향 프로그래밍

#### 강의 목표
- 클래스와 객체의 개념 이해
- 클래스의 선언 및 정의
- 접근 지정자 (public, private, protected) 이해
- 생성자와 소멸자의 개념 및 사용법 이해
- this 포인터의 사용법 이해

#### 강의 내용

##### 1. 클래스 정의와 객체 생성
- **클래스 선언 및 정의**

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
    person1.name = "John Doe";
    person1.age = 30;
    person1.introduce();

    return 0;
}
```

##### 2. 접근 지정자 (public, private, protected)
- **public, private 접근 지정자**

```cpp
#include <iostream>
using namespace std;

class Person {
private:
    string name;
    int age;

public:
    void setName(string n) {
        name = n;
    }

    string getName() {
        return name;
    }

    void setAge(int a) {
        age = a;
    }

    int getAge() {
        return age;
    }

    void introduce() {
        cout << "Name: " << name << ", Age: " << age << endl;
    }
};

int main() {
    Person person1;
    person1.setName("Jane Doe");
    person1.setAge(25);
    person1.introduce();

    return 0;
}
```

- **protected 접근 지정자**

```cpp
#include <iostream>
using namespace std;

class Base {
protected:
    int value;
};

class Derived : public Base {
public:
    void setValue(int v) {
        value = v;
    }

    void displayValue() {
        cout << "Value: " << value << endl;
    }
};

int main() {
    Derived obj;
    obj.setValue(42);
    obj.displayValue();

    return 0;
}
```

##### 3. 생성자와 소멸자
- **기본 생성자와 소멸자**

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

    // 매개변수 생성자
    Person(string n, int a) {
        name = n;
        age = a;
    }

    // 소멸자
    ~Person() {
        cout << "Destructor called for " << name << endl;
    }

    void introduce() {
        cout << "Name: " << name << ", Age: " << age << endl;
    }
};

int main() {
    Person person1;
    person1.introduce();

    Person person2("John Doe", 30);
    person2.introduce();

    return 0;
}
```

##### 4. this 포인터
- **this 포인터 사용**

```cpp
#include <iostream>
using namespace std;

class Person {
public:
    string name;
    int age;

    Person(string name, int age) {
        this->name = name;
        this->age = age;
    }

    void introduce() {
        cout << "Name: " << this->name << ", Age: " << this->age << endl;
    }
};

int main() {
    Person person1("John Doe", 30);
    person1.introduce();

    return 0;
}
```

#### 과제

1. **클래스 정의 및 사용**
   - Car 클래스를 정의하고, 브랜드, 모델, 연식을 멤버 변수로 가집니다. 기본 생성자와 매개변수 생성자를 정의하고, 차의 정보를 출력하는 메서드를 작성하세요.

```cpp
#include <iostream>
using namespace std;

class Car {
public:
    string brand;
    string model;
    int year;

    Car() {
        brand = "Unknown";
        model = "Unknown";
        year = 0;
    }

    Car(string b, string m, int y) {
        brand = b;
        model = m;
        year = y;
    }

    void displayInfo() {
        cout << "Brand: " << brand << ", Model: " << model << ", Year: " << year << endl;
    }
};

int main() {
    Car car1;
    car1.displayInfo();

    Car car2("Toyota", "Corolla", 2020);
    car2.displayInfo();

    return 0;
}
```

2. **접근 지정자 사용**
   - Rectangle 클래스를 정의하고, 길이와 너비를 private 멤버 변수로 가집니다. 길이와 너비를 설정하고 면적을 계산하는 public 메서드를 작성하세요.

```cpp
#include <iostream>
using namespace std;

class Rectangle {
private:
    double length;
    double width;

public:
    void setDimensions(double l, double w) {
        length = l;
        width = w;
    }

    double calculateArea() {
        return length * width;
    }
};

int main() {
    Rectangle rect;
    rect.setDimensions(5.0, 3.0);
    cout << "Area of the rectangle: " << rect.calculateArea() << endl;

    return 0;
}
```

3. **생성자와 소멸자**
   - Book 클래스를 정의하고, 제목, 저자, 페이지 수를 멤버 변수로 가집니다. 기본 생성자와 매개변수 생성자를 정의하고, 소멸자에서 책 정보를 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
using namespace std;

class Book {
public:
    string title;
    string author;
    int pages;

    Book() {
        title = "Unknown";
        author = "Unknown";
        pages = 0;
    }

    Book(string t, string a, int p) {
        title = t;
        author = a;
        pages = p;
    }

    ~Book() {
        cout << "Destructor called for book: " << title << endl;
    }

    void displayInfo() {
        cout << "Title: " << title << ", Author: " << author << ", Pages: " << pages << endl;
    }
};

int main() {
    Book book1;
    book1.displayInfo();

    Book book2("1984", "George Orwell", 328);
    book2.displayInfo();

    return 0;
}
```

#### 퀴즈

1. **클래스에 대한 설명 중 틀린 것은?**
   - A) 클래스는 객체를 생성하는 데 사용된다.
   - B) 클래스의 멤버 변수는 기본적으로 public이다.
   - C) 클래스는 데이터를 캡슐화할 수 있다.
   - D) 클래스는 멤버 함수와 변수를 포함할 수 있다.

2. **생성자에 대한 설명 중 맞는 것은?**
   - A) 생성자는 반환 타입이 없다.
   - B) 생성자는 클래스 외부에서 호출할 수 있다.
   - C) 생성자는 반드시 매개변수를 가져야 한다.
   - D) 생성자는 클래스의 멤버 변수가 아니라 지역 변수를 초기화한다.

3. **this 포인터에 대한 설명 중 맞는 것은?**
   - A) this 포인터는 객체 자신을 가리킨다.
   - B) this 포인터는 정적 멤버 함수에서 사용될 수 있다.
   - C) this 포인터는 함수 외부에서 사용할 수 있다.
   - D) this 포인터는 다른 객체를 가리킬 수 있다.

#### 퀴즈 해설

1. **클래스에 대한 설명 중 틀린 것은?**
   - **정답: B) 클래스의 멤버 변수는 기본적으로 public이다.**
     - 해설: 클래스의 멤버 변수는 기본적으로 private입니다.

2. **생성자에 대한 설명 중 맞는 것은?**
   - **정답: A) 생성자는 반환 타입이 없다.**
     - 해설: 생성자는 반환 타입이 없으며, 클래스가 생성될 때 자동으로 호출됩니다.

3. **this 포인터에 대한 설명 중 맞는 것은?**
   - **정답: A) this 포인터는 객체 자신을 가리킨다.**
     - 해설: this 포인터는 클래스의 멤버 함수 내에서 객체 자신을 가리킵니다.

다음 주차 강의 내용을 요청하시면, 8주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.