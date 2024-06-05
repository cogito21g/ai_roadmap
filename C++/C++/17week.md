### 17주차: 디자인 패턴

#### 강의 목표
- 디자인 패턴의 개념 이해 및 사용
- 싱글톤 패턴, 팩토리 패턴, 전략 패턴, 옵저버 패턴의 이해 및 구현

#### 강의 내용

##### 1. 싱글톤 패턴
- **싱글톤 패턴 구현**

```cpp
#include <iostream>
using namespace std;

class Singleton {
private:
    static Singleton* instance;
    Singleton() {}  // private 생성자

public:
    static Singleton* getInstance() {
        if (instance == nullptr) {
            instance = new Singleton();
        }
        return instance;
    }

    void showMessage() {
        cout << "Hello from Singleton!" << endl;
    }
};

// 인스턴스 포인터 초기화
Singleton* Singleton::instance = nullptr;

int main() {
    Singleton* s = Singleton::getInstance();
    s->showMessage();

    return 0;
}
```

##### 2. 팩토리 패턴
- **팩토리 패턴 구현**

```cpp
#include <iostream>
using namespace std;

class Product {
public:
    virtual void show() = 0;
};

class ProductA : public Product {
public:
    void show() override {
        cout << "Product A" << endl;
    }
};

class ProductB : public Product {
public:
    void show() override {
        cout << "Product B" << endl;
    }
};

class Factory {
public:
    static Product* createProduct(char type) {
        if (type == 'A') {
            return new ProductA();
        } else if (type == 'B') {
            return new ProductB();
        }
        return nullptr;
    }
};

int main() {
    Product* p1 = Factory::createProduct('A');
    if (p1) p1->show();

    Product* p2 = Factory::createProduct('B');
    if (p2) p2->show();

    delete p1;
    delete p2;

    return 0;
}
```

##### 3. 전략 패턴
- **전략 패턴 구현**

```cpp
#include <iostream>
using namespace std;

class Strategy {
public:
    virtual void execute() = 0;
};

class ConcreteStrategyA : public Strategy {
public:
    void execute() override {
        cout << "Strategy A" << endl;
    }
};

class ConcreteStrategyB : public Strategy {
public:
    void execute() override {
        cout << "Strategy B" << endl;
    }
};

class Context {
private:
    Strategy* strategy;

public:
    void setStrategy(Strategy* s) {
        strategy = s;
    }

    void executeStrategy() {
        strategy->execute();
    }
};

int main() {
    Context context;
    ConcreteStrategyA strategyA;
    ConcreteStrategyB strategyB;

    context.setStrategy(&strategyA);
    context.executeStrategy();

    context.setStrategy(&strategyB);
    context.executeStrategy();

    return 0;
}
```

##### 4. 옵저버 패턴
- **옵저버 패턴 구현**

```cpp
#include <iostream>
#include <vector>
using namespace std;

class Observer {
public:
    virtual void update() = 0;
};

class ConcreteObserver : public Observer {
public:
    void update() override {
        cout << "Observer updated" << endl;
    }
};

class Subject {
private:
    vector<Observer*> observers;

public:
    void addObserver(Observer* observer) {
        observers.push_back(observer);
    }

    void notifyObservers() {
        for (auto observer : observers) {
            observer->update();
        }
    }
};

int main() {
    Subject subject;
    ConcreteObserver observer1, observer2;

    subject.addObserver(&observer1);
    subject.addObserver(&observer2);

    subject.notifyObservers();

    return 0;
}
```

#### 과제

1. **싱글톤 패턴 구현**
   - 싱글

톤 패턴을 사용하여 설정 클래스를 구현하고, 설정 값을 저장하고 불러오는 프로그램을 작성하세요.

```cpp
#include <iostream>
using namespace std;

class ConfigurationManager {
private:
    static ConfigurationManager* instance;
    string configValue;

    ConfigurationManager() : configValue("DefaultConfig") {}

public:
    static ConfigurationManager* getInstance() {
        if (instance == nullptr) {
            instance = new ConfigurationManager();
        }
        return instance;
    }

    void setConfigValue(const string& value) {
        configValue = value;
    }

    string getConfigValue() const {
        return configValue;
    }
};

ConfigurationManager* ConfigurationManager::instance = nullptr;

int main() {
    ConfigurationManager* config = ConfigurationManager::getInstance();
    cout << "Initial Config: " << config->getConfigValue() << endl;

    config->setConfigValue("NewConfig");
    cout << "Updated Config: " << config->getConfigValue() << endl;

    return 0;
}
```

2. **팩토리 패턴 구현**
   - 팩토리 패턴을 사용하여 다양한 형태의 도형 객체를 생성하고, 도형의 면적을 계산하는 프로그램을 작성하세요.

```cpp
#include <iostream>
using namespace std;

class Shape {
public:
    virtual double getArea() = 0;
};

class Circle : public Shape {
private:
    double radius;

public:
    Circle(double r) : radius(r) {}

    double getArea() override {
        return 3.14159 * radius * radius;
    }
};

class Rectangle : public Shape {
private:
    double width, height;

public:
    Rectangle(double w, double h) : width(w), height(h) {}

    double getArea() override {
        return width * height;
    }
};

class ShapeFactory {
public:
    static Shape* createShape(char type, double a, double b = 0) {
        if (type == 'C') {
            return new Circle(a);
        } else if (type == 'R') {
            return new Rectangle(a, b);
        }
        return nullptr;
    }
};

int main() {
    Shape* circle = ShapeFactory::createShape('C', 5.0);
    Shape* rectangle = ShapeFactory::createShape('R', 4.0, 6.0);

    if (circle) {
        cout << "Circle Area: " << circle->getArea() << endl;
    }
    if (rectangle) {
        cout << "Rectangle Area: " << rectangle->getArea() << endl;
    }

    delete circle;
    delete rectangle;

    return 0;
}
```

3. **전략 패턴 구현**
   - 전략 패턴을 사용하여 다양한 정렬 알고리즘을 구현하고, 이를 사용하여 배열을 정렬하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class SortStrategy {
public:
    virtual void sort(vector<int>& data) = 0;
};

class BubbleSort : public SortStrategy {
public:
    void sort(vector<int>& data) override {
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data.size() - i - 1; ++j) {
                if (data[j] > data[j + 1]) {
                    swap(data[j], data[j + 1]);
                }
            }
        }
    }
};

class QuickSort : public SortStrategy {
public:
    void sort(vector<int>& data) override {
        quickSort(data, 0, data.size() - 1);
    }

private:
    void quickSort(vector<int>& data, int left, int right) {
        if (left >= right) return;
        int pivot = partition(data, left, right);
        quickSort(data, left, pivot - 1);
        quickSort(data, pivot + 1, right);
    }

    int partition(vector<int>& data, int left, int right) {
        int pivot = data[right];
        int i = left - 1;
        for (int j = left; j < right; ++j) {
            if (data[j] < pivot) {
                ++i;
                swap(data[i], data[j]);
            }
        }
        swap(data[i + 1], data[right]);
        return i + 1;
    }
};

class Context {
private:
    SortStrategy* strategy;

public:
    void setStrategy(SortStrategy* s) {
        strategy = s;
    }

    void executeStrategy(vector<int>& data) {
        strategy->sort(data);
    }
};

int main() {
    vector<int> data = {5, 2, 9, 1, 5, 6};
    Context context;
    BubbleSort bubbleSort;
    QuickSort quickSort;

    context.setStrategy(&bubbleSort);
    context.executeStrategy(data);

    cout << "BubbleSort: ";
    for (int num : data) {
        cout << num << " ";
    }
    cout << endl;

    data = {5, 2, 9, 1, 5, 6};
    context.setStrategy(&quickSort);
    context.executeStrategy(data);

    cout << "QuickSort: ";
    for (int num : data) {
        cout << num << " ";
    }
    cout << endl;

    return 0;
}
```

4. **옵저버 패턴 구현**
   - 옵저버 패턴을 사용하여 주식 가격을 감시하는 프로그램을 작성하세요. 주식 가격이 변경되면 모든 옵저버에게 알림을 보냅니다.

```cpp
#include <iostream>
#include <vector>
using namespace std;

class Observer {
public:
    virtual void update(float price) = 0;
};

class ConcreteObserver : public Observer {
private:
    string name;

public:
    ConcreteObserver(const string& n) : name(n) {}

    void update(float price) override {
        cout << name << " received price update: " << price << endl;
    }
};

class Stock {
private:
    vector<Observer*> observers;
    float price;

public:
    void addObserver(Observer* observer) {
        observers.push_back(observer);
    }

    void removeObserver(Observer* observer) {
        observers.erase(remove(observers.begin(), observers.end(), observer), observers.end());
    }

    void notifyObservers() {
        for (auto observer : observers) {
            observer->update(price);
        }
    }

    void setPrice(float p) {
        price = p;
        notifyObservers();
    }
};

int main() {
    Stock stock;
    ConcreteObserver observer1("Observer 1");
    ConcreteObserver observer2("Observer 2");

    stock.addObserver(&observer1);
    stock.addObserver(&observer2);

    stock.setPrice(100.0);
    stock.setPrice(105.5);

    return 0;
}
```

#### 퀴즈

1. **싱글톤 패턴에 대한 설명 중 맞는 것은?**
   - A) 싱글톤 패턴은 여러 개의 인스턴스를 생성할 수 있다.
   - B) 싱글톤 패턴은 전역 변수를 사용하는 것과 같다.
   - C) 싱글톤 패턴은 하나의 인스턴스만 생성되도록 보장한다.
   - D) 싱글톤 패턴은 다중 상속을 지원한다.

2. **팩토리 패턴에 대한 설명 중 맞는 것은?**
   - A) 팩토리 패턴은 객체 생성을 캡슐화한다.
   - B) 팩토리 패턴은 클래스의 인스턴스를 직접 생성한다.
   - C) 팩토리 패턴은 상속을 지원하지 않는다.
   - D) 팩토리 패턴은 객체의 생성을 제한한다.

3. **전략 패턴에 대한 설명 중 맞는 것은?**
   - A) 전략 패턴은 객체의 상태를 변경하는 데 사용된다.
   - B) 전략 패턴은 알고리즘을 캡슐화하여 동적으로 변경할 수 있게 한다.
   - C) 전략 패턴은 인터페이스를 사용하지 않는다.
   - D) 전략 패턴은 객체의 생성을 제한한다.

4. **옵저버 패턴에 대한 설명 중 맞는 것은?**
   - A) 옵저버 패턴은 이벤트 기반 프로그래밍에 사용된다.
   - B) 옵저버 패턴은 객체 간의 의존성을 줄인다.
   - C) 옵저버 패턴은 객체의 생성을 제한한다.
   - D) 옵저버 패턴은 객체의 상태를 변경하지 않는다.

#### 퀴즈 해설

1. **싱글톤 패턴에 대한 설명 중 맞는 것은?**
   - **정답: C) 싱글톤 패턴은 하나의 인스턴스만 생성되도록 보장한다.**
     - 해설: 싱글톤 패턴은 클래스의 인스턴스가 하나만 생성되도록 보장하는 디자인 패턴입니다.

2. **팩토리 패턴에 대한 설명 중 맞는 것은?**
   - **정답: A) 팩토리 패턴은 객체 생성을 캡슐화한다.**
     - 해설: 팩토리 패턴은 객체 생성 로직을 캡슐화하여 클라이언트 코드에서 객체 생성 방식을 숨깁니다.

3. **전략 패턴에 대한 설명 중 맞는 것은?**
   - **정답: B) 전략 패턴은 알고리즘을

 캡슐화하여 동적으로 변경할 수 있게 한다.**
     - 해설: 전략 패턴은 알고리즘을 인터페이스로 캡슐화하여 런타임에 동적으로 알고리즘을 변경할 수 있도록 합니다.

4. **옵저버 패턴에 대한 설명 중 맞는 것은?**
   - **정답: A) 옵저버 패턴은 이벤트 기반 프로그래밍에 사용된다.**
     - 해설: 옵저버 패턴은 주체와 관찰자 간의 관계를 정의하여 이벤트 발생 시 관찰자에게 알림을 보내는 방식으로 이벤트 기반 프로그래밍에 사용됩니다.

다음 주차 강의 내용을 요청하시면, 18주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.