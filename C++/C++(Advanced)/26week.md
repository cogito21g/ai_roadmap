### 26주차: 디자인 패턴 심화

#### 강의 목표
- 행동 패턴 이해 및 사용
- 생성 패턴 이해 및 사용
- 구조 패턴 이해 및 사용

#### 강의 내용

##### 1. 행동 패턴
- **전략 패턴 (Strategy Pattern)**

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

- **옵저버 패턴 (Observer Pattern)**

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

##### 2. 생성 패턴
- **팩토리 메서드 패턴 (Factory Method Pattern)**

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

class Creator {
public:
    virtual Product* createProduct() = 0;
};

class CreatorA : public Creator {
public:
    Product* createProduct() override {
        return new ProductA();
    }
};

class CreatorB : public Creator {
public:
    Product* createProduct() override {
        return new ProductB();
    }
};

int main() {
    Creator* creatorA = new CreatorA();
    Product* productA = creatorA->createProduct();
    productA->show();

    Creator* creatorB = new CreatorB();
    Product* productB = creatorB->createProduct();
    productB->show();

    delete productA;
    delete productB;
    delete creatorA;
    delete creatorB;

    return 0;
}
```

- **싱글톤 패턴 (Singleton Pattern)**

```cpp
#include <iostream>
using namespace std;

class Singleton {
private:
    static Singleton* instance;
    Singleton() {}

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

// Initialize pointer to zero so that it can be initialized in first call to getInstance
Singleton* Singleton::instance = nullptr;

int main() {
    Singleton* s = Singleton::getInstance();
    s->showMessage();

    return 0;
}
```

##### 3. 구조 패턴
- **어댑터 패턴 (Adapter Pattern)**

```cpp
#include <iostream>
using namespace std;

class Target {
public:
    virtual void request() {
        cout << "Target request" << endl;
    }
};

class Adaptee {
public:
    void specificRequest() {
        cout << "Adaptee specific request" << endl;
    }
};

class Adapter : public Target {
private:
    Adaptee* adaptee;

public:
    Adapter(Adaptee* a) : adaptee(a) {}

    void request() override {
        adaptee->specificRequest();
    }
};

int main() {
    Adaptee* adaptee = new Adaptee();
    Target* target = new Adapter(adaptee);

    target->request();

    delete adaptee;
    delete target;

    return 0;
}
```

- **데코레이터 패턴 (Decorator Pattern)**

```cpp
#include <iostream>
using namespace std;

class Component {
public:
    virtual void operation() = 0;
};

class ConcreteComponent : public Component {
public:
    void operation() override {
        cout << "ConcreteComponent operation" << endl;
    }
};

class Decorator : public Component {
protected:
    Component* component;

public:
    Decorator(Component* c) : component(c) {}

    void operation() override {
        component->operation();
    }
};

class ConcreteDecoratorA : public Decorator {
public:
    ConcreteDecoratorA(Component* c) : Decorator(c) {}

    void operation() override {
        Decorator::operation();
        cout << "ConcreteDecoratorA operation" << endl;
    }
};

class ConcreteDecoratorB : public Decorator {
public:
    ConcreteDecoratorB(Component* c) : Decorator(c) {}

    void operation() override {
        Decorator::operation();
        cout << "ConcreteDecoratorB operation" << endl;
    }
};

int main() {
    Component* component = new ConcreteComponent();
    Component* decoratorA = new ConcreteDecoratorA(component);
    Component* decoratorB = new ConcreteDecoratorB(decoratorA);

    decoratorB->operation();

    delete decoratorB;
    delete decoratorA;
    delete component;

    return 0;
}
```

#### 과제

1. **전략 패턴을 사용하여 다양한 정렬 알고리즘을 구현하고, 이를 사용하여 배열을 정렬하는 프로그램을 작성하세요.**

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

2. **옵저버 패턴을 사용하여 주식 가격을 감시하는 프로그램을 작성하세요. 주식 가격이 변경되면 모든 옵저버에게 알림을 보냅니다.**

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

    void removeObserver(Observer

* observer) {
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

    stock.setPrice(100.0f);
    stock.setPrice(105.5f);

    return 0;
}
```

3. **팩토리 메서드 패턴을 사용하여 다양한 형태의 문서 객체를 생성하는 프로그램을 작성하세요.**

```cpp
#include <iostream>
using namespace std;

class Document {
public:
    virtual void open() = 0;
};

class WordDocument : public Document {
public:
    void open() override {
        cout << "Opening Word document" << endl;
    }
};

class PDFDocument : public Document {
public:
    void open() override {
        cout << "Opening PDF document" << endl;
    }
};

class Application {
public:
    virtual Document* createDocument() = 0;

    void newDocument() {
        Document* doc = createDocument();
        doc->open();
        delete doc;
    }
};

class WordApplication : public Application {
public:
    Document* createDocument() override {
        return new WordDocument();
    }
};

class PDFApplication : public Application {
public:
    Document* createDocument() override {
        return new PDFDocument();
    }
};

int main() {
    Application* app1 = new WordApplication();
    app1->newDocument();

    Application* app2 = new PDFApplication();
    app2->newDocument();

    delete app1;
    delete app2;

    return 0;
}
```

4. **싱글톤 패턴을 사용하여 로그 시스템을 구현하세요.**

```cpp
#include <iostream>
#include <fstream>
using namespace std;

class Logger {
private:
    static Logger* instance;
    ofstream logFile;

    Logger() {
        logFile.open("log.txt", ios::app);
    }

public:
    static Logger* getInstance() {
        if (instance == nullptr) {
            instance = new Logger();
        }
        return instance;
    }

    void log(const string& message) {
        logFile << message << endl;
    }

    ~Logger() {
        logFile.close();
    }
};

Logger* Logger::instance = nullptr;

int main() {
    Logger* logger = Logger::getInstance();
    logger->log("This is a log message.");
    logger->log("This is another log message.");

    return 0;
}
```

#### 퀴즈

1. **전략 패턴의 주요 목적은 무엇인가?**
   - A) 객체의 상태를 변경하기 위해
   - B) 객체의 생성을 지연하기 위해
   - C) 알고리즘을 캡슐화하고 동적으로 변경하기 위해
   - D) 객체 간의 통신을 위해

2. **옵저버 패턴의 주요 목적은 무엇인가?**
   - A) 객체의 생성을 지연하기 위해
   - B) 객체 간의 상태 변화를 감시하고 알림을 보내기 위해
   - C) 알고리즘을 캡슐화하기 위해
   - D) 객체의 상태를 변경하기 위해

3. **팩토리 메서드 패턴의 주요 목적은 무엇인가?**
   - A) 객체의 상태를 변경하기 위해
   - B) 객체의 생성을 지연하기 위해
   - C) 객체 생성을 서브클래스에서 처리하도록 하기 위해
   - D) 객체 간의 통신을 위해

4. **싱글톤 패턴의 주요 목적은 무엇인가?**
   - A) 여러 객체의 생성을 지연하기 위해
   - B) 하나의 객체만 생성하고 전역적으로 접근할 수 있게 하기 위해
   - C) 객체의 상태를 변경하기 위해
   - D) 객체 간의 통신을 위해

#### 퀴즈 해설

1. **전략 패턴의 주요 목적은 무엇인가?**
   - **정답: C) 알고리즘을 캡슐화하고 동적으로 변경하기 위해**
     - 해설: 전략 패턴은 알고리즘을 캡슐화하여 런타임에 동적으로 변경할 수 있도록 합니다.

2. **옵저버 패턴의 주요 목적은 무엇인가?**
   - **정답: B) 객체 간의 상태 변화를 감시하고 알림을 보내기 위해**
     - 해설: 옵저버 패턴은 주체와 관찰자 간의 관계를 정의하여 이벤트 발생 시 관찰자에게 알림을 보내는 방식입니다.

3. **팩토리 메서드 패턴의 주요 목적은 무엇인가?**
   - **정답: C) 객체 생성을 서브클래스에서 처리하도록 하기 위해**
     - 해설: 팩토리 메서드 패턴은 객체 생성을 서브클래스에서 처리하도록 하여 객체 생성의 책임을 분리합니다.

4. **싱글톤 패턴의 주요 목적은 무엇인가?**
   - **정답: B) 하나의 객체만 생성하고 전역적으로 접근할 수 있게 하기 위해**
     - 해설: 싱글톤 패턴은 클래스의 인스턴스를 하나만 생성하고, 전역적으로 접근할 수 있도록 합니다.

이로써 26주차 디자인 패턴 심화 과정을 마쳤습니다. 추가로 학습하고 싶은 주제나 더 알고 싶은 내용이 있다면 언제든지 말씀해 주세요.