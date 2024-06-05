### 8주차: 상속과 다형성

#### 강의 목표
- 상속의 개념 이해 및 사용
- 다형성의 개념 이해 및 사용
- 가상 함수와 순수 가상 함수의 이해
- 연산자 오버로딩의 개념 이해 및 사용

#### 강의 내용

##### 1. 상속 기초
- **상속의 기본 개념과 사용**

```cpp
#include <iostream>
using namespace std;

// 기본 클래스
class Animal {
public:
    void eat() {
        cout << "Eating..." << endl;
    }
};

// 파생 클래스
class Dog : public Animal {
public:
    void bark() {
        cout << "Barking..." << endl;
    }
};

int main() {
    Dog dog;
    dog.eat();  // 기본 클래스의 메서드 호출
    dog.bark();  // 파생 클래스의 메서드 호출

    return 0;
}
```

##### 2. 다형성 (Polymorphism)과 가상 함수
- **다형성의 기본 개념**

```cpp
#include <iostream>
using namespace std;

class Animal {
public:
    virtual void sound() {
        cout << "Some sound..." << endl;
    }
};

class Dog : public Animal {
public:
    void sound() override {
        cout << "Woof!" << endl;
    }
};

class Cat : public Animal {
public:
    void sound() override {
        cout << "Meow!" << endl;
    }
};

void makeSound(Animal* animal) {
    animal->sound();
}

int main() {
    Dog dog;
    Cat cat;

    makeSound(&dog);  // Woof!
    makeSound(&cat);  // Meow!

    return 0;
}
```

##### 3. 순수 가상 함수와 추상 클래스
- **순수 가상 함수**

```cpp
#include <iostream>
using namespace std;

class Shape {
public:
    virtual void draw() = 0;  // 순수 가상 함수
};

class Circle : public Shape {
public:
    void draw() override {
        cout << "Drawing Circle" << endl;
    }
};

class Rectangle : public Shape {
public:
    void draw() override {
        cout << "Drawing Rectangle" << endl;
    }
};

int main() {
    Circle circle;
    Rectangle rectangle;

    circle.draw();
    rectangle.draw();

    return 0;
}
```

##### 4. 연산자 오버로딩
- **연산자 오버로딩**

```cpp
#include <iostream>
using namespace std;

class Complex {
private:
    float real;
    float imag;

public:
    Complex() : real(0), imag(0) {}
    Complex(float r, float i) : real(r), imag(i) {}

    // + 연산자 오버로딩
    Complex operator + (const Complex& other) {
        return Complex(real + other.real, imag + other.imag);
    }

    void display() {
        cout << "Real: " << real << ", Imaginary: " << imag << endl;
    }
};

int main() {
    Complex c1(3.0, 4.0);
    Complex c2(1.0, 2.0);
    Complex c3 = c1 + c2;

    c3.display();

    return 0;
}
```

#### 과제

1. **상속 사용**
   - 기본 클래스 Vehicle을 정의하고, 이를 상속하는 Car와 Bike 클래스를 작성하세요. 각 클래스는 고유한 메서드를 가집니다.

```cpp
#include <iostream>
using namespace std;

class Vehicle {
public:
    void start() {
        cout << "Vehicle starting..." << endl;
    }
};

class Car : public Vehicle {
public:
    void honk() {
        cout << "Car honking..." << endl;
    }
};

class Bike : public Vehicle {
public:
    void rev() {
        cout << "Bike revving..." << endl;
    }
};

int main() {
    Car car;
    Bike bike;

    car.start();
    car.honk();

    bike.start();
    bike.rev();

    return 0;
}
```

2. **다형성 사용**
   - 기본 클래스 Animal을 정의하고, 이를 상속하는 Dog와 Cat 클래스를 작성하세요. 각 클래스는 sound() 메서드를 오버라이딩합니다. 포인터를 사용하여 다형성을 구현하세요.

```cpp
#include <iostream>
using namespace std;

class Animal {
public:
    virtual void sound() {
        cout << "Some sound..." << endl;
    }
};

class Dog : public Animal {
public:
    void sound() override {
        cout << "Woof!" << endl;
    }
};

class Cat : public Animal {
public:
    void sound() override {
        cout << "Meow!" << endl;
    }
};

void makeSound(Animal* animal) {
    animal->sound();
}

int main() {
    Dog dog;
    Cat cat;

    makeSound(&dog);
    makeSound(&cat);

    return 0;
}
```

3. **순수 가상 함수 사용**
   - 기본 클래스 Shape을 정의하고, 순수 가상 함수 draw()를 선언합니다. 이를 상속하는 Circle과 Square 클래스를 작성하고, draw() 메서드를 구현하세요.

```cpp
#include <iostream>
using namespace std;

class Shape {
public:
    virtual void draw() = 0;
};

class Circle : public Shape {
public:
    void draw() override {
        cout << "Drawing Circle" << endl;
    }
};

class Square : public Shape {
public:
    void draw() override {
        cout << "Drawing Square" << endl;
    }
};

int main() {
    Circle circle;
    Square square;

    circle.draw();
    square.draw();

    return 0;
}
```

4. **연산자 오버로딩**
   - Complex 클래스에 + 연산자를 오버로딩하여 두 복소수를 더하는 프로그램을 작성하세요.

```cpp
#include <iostream>
using namespace std;

class Complex {
private:
    float real;
    float imag;

public:
    Complex() : real(0), imag(0) {}
    Complex(float r, float i) : real(r), imag(i) {}

    Complex operator + (const Complex& other) {
        return Complex(real + other.real, imag + other.imag);
    }

    void display() {
        cout << "Real: " << real << ", Imaginary: " << imag << endl;
    }
};

int main() {
    Complex c1(3.0, 4.0);
    Complex c2(1.0, 2.0);
    Complex c3 = c1 + c2;

    c3.display();

    return 0;
}
```

#### 퀴즈

1. **상속에 대한 설명 중 틀린 것은?**
   - A) 상속은 코드의 재사용성을 높인다.
   - B) 파생 클래스는 기본 클래스의 모든 멤버를 상속받는다.
   - C) 상속은 is-a 관계를 나타낸다.
   - D) private 멤버는 상속될 수 없다.

2. **다형성에 대한 설명 중 맞는 것은?**
   - A) 다형성은 동일한 함수가 다른 기능을 수행할 수 있도록 한다.
   - B) 다형성은 연산자 오버로딩에서만 사용된다.
   - C) 다형성은 기본 클래스의 멤버 함수를 숨긴다.
   - D) 다형성은 컴파일 타임에 결정된다.

3. **순수 가상 함수에 대한 설명 중 맞는 것은?**
   - A) 순수 가상 함수는 정의를 가질 수 있다.
   - B) 순수 가상 함수가 있는 클래스는 추상 클래스가 된다.
   - C) 추상 클래스는 인스턴스화할 수 있다.
   - D) 순수 가상 함수는 private 접근 지정자만 가질 수 있다.

4. **연산자 오버로딩에 대한 설명 중 맞는 것은?**
   - A) 모든 연산자는 오버로딩할 수 있다.
   - B) 연산자 오버로딩은 함수 오버로딩과 동일하다.
   - C) 연산자 오버로딩은 클래스 외부에서만 정의할 수 있다.
   - D) 연산자 오버로딩은 객체 간의 연산을 가능하게 한다.

#### 퀴즈 해설

1. **상속에 대한 설명 중 틀린 것은?**
   - **정답: D) private 멤버는 상속될 수 없다.**
     - 해설: private 멤버는 상속되지만, 파생 클래스에서 직접 접근할 수 없습니다.

2. **다형성에 대한 설명 중 맞는 것은?**
   - **정답: A) 다형성은 동일한 함수가 다른 기능을 수행할 수 있도록 한다.**
     - 해설: 다형성은 동일한 함수 호출이 객체의 타입에 따라 다른 기능을 수행할 수 있게 합니다.

3. **순수 가상 함수에 대한 설명 중 맞는 것은?**
   - **정답: B) 순수 가상 함수가 있는 클래스는 추상 클래스가 된다.**
     - 해설: 순수 가상 함수가 있는 클래스는 추상 클래스가 되며, 인스턴스화할 수 없습니다.

4. **연산자 오버로딩에 대한 설명 중 맞는 것은?**
   - **정답: D) 연산자 오버로딩은 객체

 간의 연산을 가능하게 한다.**
     - 해설: 연산자 오버로딩은 객체 간의 연산을 가능하게 하여 코드의 가독성을 높입니다.

다음 주차 강의 내용을 요청하시면, 9주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.