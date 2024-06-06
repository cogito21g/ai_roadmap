### 9주차 강의 계획: 해시 테이블 (Hash Table)

#### 강의 목표
- 해시 테이블의 기본 개념과 필요성 이해
- 해시 함수의 설계 및 충돌 해결 기법 학습
- 해시 테이블의 구현 및 활용

#### 강의 구성
- **강의 시간**: 2시간
- **구성**: 해시 테이블 이론 (30분), 해시 함수와 충돌 해결 (30분), 해시 테이블 구현 (30분), 실습 및 과제 안내 (30분)

#### 강의 내용

##### 1. 해시 테이블 이론 (30분)

###### 1.1 해시 테이블의 기본 개념
- **해시 테이블 개요**:
  - 키-값 쌍을 저장하는 데이터 구조
  - 해시 함수를 사용하여 키를 인덱스로 변환
- **해시 테이블의 장단점**:
  - **장점**: 평균적으로 빠른 검색, 삽입, 삭제
  - **단점**: 해시 충돌 가능성, 메모리 사용량

###### 1.2 해시 테이블의 활용 사례
- **활용 사례**:
  - 데이터베이스 인덱싱
  - 캐싱
  - 집합 연산 (Set)

##### 2. 해시 함수와 충돌 해결 (30분)

###### 2.1 해시 함수의 설계
- **해시 함수 설계 원칙**:
  - 균등 분포
  - 효율적인 계산
  - 결정론적 특성

###### 2.2 충돌 해결 기법
- **충돌 해결 기법**:
  - **체이닝**: 동일한 해시 값을 가진 항목들을 연결 리스트로 저장
  - **오픈 어드레싱**: 충돌 발생 시 다른 빈 슬롯을 찾는 방법 (선형 탐사, 이차 탐사, 이중 해싱)
- **예제 (체이닝)**:
```cpp
#include <iostream>
#include <list>
using namespace std;

class HashTable {
private:
    int BUCKET;    // Number of buckets
    list<int>* table; // Pointer to an array containing buckets
public:
    HashTable(int V);  // Constructor

    void insertItem(int x);
    void deleteItem(int key);

    int hashFunction(int x) {
        return (x % BUCKET);
    }

    void displayHash();
};

HashTable::HashTable(int b) {
    this->BUCKET = b;
    table = new list<int>[BUCKET];
}

void HashTable::insertItem(int key) {
    int index = hashFunction(key);
    table[index].push_back(key);
}

void HashTable::deleteItem(int key) {
    int index = hashFunction(key);

    list<int>::iterator i;
    for (i = table[index].begin(); i != table[index].end(); i++) {
        if (*i == key)
            break;
    }

    if (i != table[index].end())
        table[index].erase(i);
}

void HashTable::displayHash() {
    for (int i = 0; i < BUCKET; i++) {
        cout << i;
        for (auto x : table[i])
            cout << " --> " << x;
        cout << endl;
    }
}

int main() {
    int arr[] = {15, 11, 27, 8, 12};
    int n = sizeof(arr) / sizeof(arr[0]);

    HashTable h(7);   // 7 is count of buckets in hash table
    for (int i = 0; i < n; i++)
        h.insertItem(arr[i]);

    h.deleteItem(12);

    h.displayHash();
    return 0;
}
```

##### 3. 해시 테이블 구현 (30분)

###### 3.1 해시 테이블 클래스 구현
- **구현 요소**:
  - 해시 함수
  - 삽입 연산 (`insert`)
  - 삭제 연산 (`delete`)
  - 검색 연산 (`search`)

###### 3.2 충돌 해결 기법 구현
- **체이닝과 오픈 어드레싱 구현**:
  - 체이닝을 이용한 해시 테이블 구현
  - 오픈 어드레싱을 이용한 해시 테이블 구현
- **예제 (오픈 어드레싱)**:
```cpp
#include <iostream>
using namespace std;

class HashTable {
private:
    int* table;
    int capacity;
    int size;
    const int EMPTY = -1;
public:
    HashTable(int capacity) {
        this->capacity = capacity;
        table = new int[capacity];
        size = 0;
        for (int i = 0; i < capacity; i++) {
            table[i] = EMPTY;
        }
    }

    ~HashTable() {
        delete[] table;
    }

    int hashFunction(int key) {
        return key % capacity;
    }

    void insert(int key) {
        if (size == capacity) {
            cout << "Hash Table is full\n";
            return;
        }
        int index = hashFunction(key);
        while (table[index] != EMPTY) {
            index = (index + 1) % capacity;
        }
        table[index] = key;
        size++;
    }

    void deleteItem(int key) {
        int index = hashFunction(key);
        while (table[index] != EMPTY) {
            if (table[index] == key) {
                table[index] = EMPTY;
                size--;
                return;
            }
            index = (index + 1) % capacity;
        }
        cout << "Key not found\n";
    }

    bool search(int key) {
        int index = hashFunction(key);
        while (table[index] != EMPTY) {
            if (table[index] == key) {
                return true;
            }
            index = (index + 1) % capacity;
        }
        return false;
    }

    void displayHash() {
        for (int i = 0; i < capacity; i++) {
            if (table[i] != EMPTY) {
                cout << i << " --> " << table[i] << endl;
            } else {
                cout << i << endl;
            }
        }
    }
};

int main() {
    HashTable h(7);
    h.insert(15);
    h.insert(11);
    h.insert(27);
    h.insert(8);
    h.insert(12);

    h.displayHash();

    h.deleteItem(12);

    h.displayHash();

    if (h.search(27)) {
        cout << "27 found\n";
    } else {
        cout << "27 not found\n";
    }

    return 0;
}
```

##### 4. 실습 및 과제 안내 (30분)

###### 4.1 실습
- **실습 목표**:
  - 해시 테이블을 사용한 프로그램 작성
- **실습 문제**:
  - 체이닝을 이용한 해시 테이블을 구현하고, 데이터를 삽입, 삭제, 검색하는 프로그램 작성
  - 오픈 어드레싱을 이용한 해시 테이블을 구현하고, 데이터를 삽입, 삭제, 검색하는 프로그램 작성

###### 4.2 과제 안내
- **과제 내용**:
  - 해시 테이블을 사용하여 학생 성적 관리 프로그램 작성
  - 해시 테이블을 사용하여 전화번호부 프로그램 작성
- **과제 제출 방법**:
  - 작성한 코드를 주차별 과제 제출 폴더에 업로드
  - 제출 기한: 다음 주 강의 전까지

##### 5. 퀴즈 및 해설 (30분)

###### 퀴즈 문제:
1. 해시 테이블에서 충돌을 해결하는 방법이 아닌 것은?
   - a) 체이닝
   - b) 오픈 어드레싱
   - c) 이진 탐색
   - d) 이차 탐사
2. 해시 테이블의 주요 연산이 아닌 것은?
   - a) 삽입
   - b) 삭제
   - c) 검색
   - d) 정렬
3. 체이닝을 사용하는 해시 테이블에서 노드를 삽입할 때 삽입되는 위치는?
   - a) 배열의 시작
   - b) 배열의 끝
   - c) 연결 리스트의 시작
   - d) 연결 리스트의 끝

###### 퀴즈 해설:
1. **정답: c) 이진 탐색**
   - 해시 테이블의 충돌 해결 방법으로는 체이닝과 오픈 어드레싱이 있습니다. 이진 탐색은 사용되지 않습니다.
2. **정답: d) 정렬**
   - 해시 테이블의 주요 연산은 삽입, 삭제, 검색입니다. 정렬은 주요 연산이 아닙니다.
3. **정답: c) 연결 리스트의 시작**
   - 체이닝을 사용하는 해시 테이블에서 새로운 노드는 연결 리스트의 시작 부분에 삽입됩니다.

### 준비 자료
- 강의 슬라이드
- 실습 코드 예제
- 과제 안내서

### 과제 해설

##### 과제: 해시 테이블을 사용하여 학생 성적 관리 프로그램 작성
- **문제**: 해시 테이블을 사용하여 학생의 이름과 성적을 관리하는 프로그램 작성
- **해설**:
  -

 해시 테이블을 사용하여 학생의 이름을 키로, 성적을 값으로 저장합니다.

```cpp
#include <iostream>
#include <unordered_map>
using namespace std;

int main() {
    unordered_map<string, int> grades;
    grades["Alice"] = 90;
    grades["Bob"] = 85;
    grades["Charlie"] = 92;

    cout << "Alice's grade: " << grades["Alice"] << endl;
    cout << "Bob's grade: " << grades["Bob"] << endl;

    grades["Bob"] = 88;
    cout << "Bob's new grade: " << grades["Bob"] << endl;

    for (auto& pair : grades) {
        cout << pair.first << ": " << pair.second << endl;
    }

    return 0;
}
```
- **설명**:
  - `unordered_map`을 사용하여 학생의 이름과 성적을 저장하고 관리합니다.
  - 학생의 성적을 삽입, 업데이트, 출력하는 기능을 구현합니다.

##### 과제: 해시 테이블을 사용하여 전화번호부 프로그램 작성
- **문제**: 해시 테이블을 사용하여 이름과 전화번호를 관리하는 프로그램 작성
- **해설**:
  - 해시 테이블을 사용하여 이름을 키로, 전화번호를 값으로 저장합니다.

```cpp
#include <iostream>
#include <unordered_map>
using namespace std;

int main() {
    unordered_map<string, string> phoneBook;
    phoneBook["Alice"] = "123-456-7890";
    phoneBook["Bob"] = "987-654-3210";
    phoneBook["Charlie"] = "555-555-5555";

    cout << "Alice's phone number: " << phoneBook["Alice"] << endl;
    cout << "Bob's phone number: " << phoneBook["Bob"] << endl;

    phoneBook["Bob"] = "111-222-3333";
    cout << "Bob's new phone number: " << phoneBook["Bob"] << endl;

    for (auto& pair : phoneBook) {
        cout << pair.first << ": " << pair.second << endl;
    }

    return 0;
}
```
- **설명**:
  - `unordered_map`을 사용하여 이름과 전화번호를 저장하고 관리합니다.
  - 전화번호를 삽입, 업데이트, 출력하는 기능을 구현합니다.

이로써 9주차 강의가 마무리됩니다. 학생들은 해시 테이블의 기본 개념과 구현 방법을 이해하고, 이를 활용하여 다양한 프로그램을 작성하는 능력을 기르게 됩니다.