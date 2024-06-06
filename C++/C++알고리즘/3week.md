### 3주차 강의 계획안

#### 강의 주제: 검색 알고리즘
- 선형 검색
- 이진 검색
- 이진 검색 트리 (BST)

---

### 강의 내용

#### 1. 선형 검색
- **개념**: 순차적으로 모든 요소를 검사하여 원하는 값을 찾는 알고리즘
- **시간 복잡도**: O(n)

**예제**: 선형 검색
```cpp
#include <iostream>
using namespace std;

int linearSearch(int arr[], int size, int target) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}

int main() {
    int arr[] = {2, 4, 6, 8, 10};
    int size = sizeof(arr) / sizeof(arr[0]);
    int target = 6;
    int result = linearSearch(arr, size, target);

    if (result != -1) {
        cout << "Element found at index " << result << endl;
    } else {
        cout << "Element not found" << endl;
    }
    return 0;
}
```

#### 2. 이진 검색
- **개념**: 정렬된 배열에서 중간 요소와 비교하여 원하는 값을 찾는 알고리즘
- **시간 복잡도**: O(log n)
- **조건**: 배열이 정렬되어 있어야 함

**예제**: 이진 검색
```cpp
#include <iostream>
using namespace std;

int binarySearch(int arr[], int size, int target) {
    int left = 0, right = size - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}

int main() {
    int arr[] = {2, 4, 6, 8, 10};
    int size = sizeof(arr) / sizeof(arr[0]);
    int target = 8;
    int result = binarySearch(arr, size, target);

    if (result != -1) {
        cout << "Element found at index " << result << endl;
    } else {
        cout << "Element not found" << endl;
    }
    return 0;
}
```

#### 3. 이진 검색 트리 (BST)
- **개념**: 각 노드가 최대 두 개의 자식을 가지는 트리 구조로, 왼쪽 자식은 부모보다 작고, 오른쪽 자식은 부모보다 큰 값
- **삽입, 삭제, 검색**: 평균 시간 복잡도 O(log n), 최악의 경우 O(n)

**예제**: 이진 검색 트리 삽입 및 검색
```cpp
#include <iostream>
using namespace std;

struct Node {
    int data;
    Node* left;
    Node* right;
    Node(int value) : data(value), left(nullptr), right(nullptr) {}
};

Node* insert(Node* root, int data) {
    if (root == nullptr) {
        return new Node(data);
    }
    if (data < root->data) {
        root->left = insert(root->left, data);
    } else {
        root->right = insert(root->right, data);
    }
    return root;
}

bool search(Node* root, int data) {
    if (root == nullptr) {
        return false;
    }
    if (root->data == data) {
        return true;
    } else if (data < root->data) {
        return search(root->left, data);
    } else {
        return search(root->right, data);
    }
}

int main() {
    Node* root = nullptr;
    root = insert(root, 8);
    insert(root, 3);
    insert(root, 10);
    insert(root, 1);
    insert(root, 6);

    int target = 6;
    if (search(root, target)) {
        cout << "Element " << target << " found in the BST" << endl;
    } else {
        cout << "Element " << target << " not found in the BST" << endl;
    }
    return 0;
}
```

---

### 과제

#### 과제 1: 선형 검색 확장
사용자로부터 정수 배열과 찾고자 하는 값을 입력받아, 선형 검색 알고리즘을 사용해 값을 찾고 그 인덱스를 출력하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
using namespace std;

int linearSearch(int arr[], int size, int target) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}

int main() {
    int size, target;
    cout << "Enter the size of the array: ";
    cin >> size;
    int* arr = new int[size];
    
    cout << "Enter the elements of the array: ";
    for (int i = 0; i < size; i++) {
        cin >> arr[i];
    }

    cout << "Enter the target value: ";
    cin >> target;

    int result = linearSearch(arr, size, target);

    if (result != -1) {
        cout << "Element found at index " << result << endl;
    } else {
        cout << "Element not found" << endl;
    }

    delete[] arr;
    return 0;
}
```

**해설**:
1. 배열 크기와 요소를 사용자로부터 입력받습니다.
2. 선형 검색 알고리즘을 사용하여 값을 검색하고, 결과를 출력합니다.

#### 과제 2: 이진 검색 확장
사용자로부터 정렬된 정수 배열과 찾고자 하는 값을 입력받아, 이진 검색 알고리즘을 사용해 값을 찾고 그 인덱스를 출력하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
using namespace std;

int binarySearch(int arr[], int size, int target) {
    int left = 0, right = size - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}

int main() {
    int size, target;
    cout << "Enter the size of the array: ";
    cin >> size;
    int* arr = new int[size];
    
    cout << "Enter the elements of the sorted array: ";
    for (int i = 0; i < size; i++) {
        cin >> arr[i];
    }

    cout << "Enter the target value: ";
    cin >> target;

    int result = binarySearch(arr, size, target);

    if (result != -1) {
        cout << "Element found at index " << result << endl;
    } else {
        cout << "Element not found" << endl;
    }

    delete[] arr;
    return 0;
}
```

**해설**:
1. 정렬된 배열의 크기와 요소를 사용자로부터 입력받습니다.
2. 이진 검색 알고리즘을 사용하여 값을 검색하고, 결과를 출력합니다.

---

### 퀴즈

#### 퀴즈 1: 다음 중 선형 검색에 대한 설명으로 옳은 것은 무엇인가요?
1. 정렬된 배열에서만 동작합니다.
2. 시간 복잡도는 O(log n)입니다.
3. 순차적으로 모든 요소를 검사합니다.
4. 이진 검색 트리에서만 사용됩니다.

**정답**: 3. 순차적으로 모든 요소를 검사합니다.

#### 퀴즈 2: 다음 코드의 출력 결과는 무엇인가요?
```cpp
#include <iostream>
using namespace std;

int binarySearch(int arr[], int size, int target) {
    int left = 0, right = size - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}

int main() {
    int arr[] = {1, 3, 5, 7, 9};
    int size = sizeof(arr) / sizeof(arr[0]);
    int target = 7;
    int result = binarySearch(arr, size, target);

    if (result != -1) {
        cout << "Element found at index " << result << endl;
    } else {
        cout << "Element not found" << endl;
    }
    return 0;
}
```

**정답**: `Element found at index 3`

**해설**:
- 배열 `[1, 3, 5, 7, 9]`에서 이진 검색으로 `7`을 찾습니다.
- 인덱스 

3에서 `7`을 찾았으므로, 해당 인덱스를 출력합니다.

이 계획안은 3주차에 필요한 검색 알고리즘의 기본 개념과 실습을 통해 학습할 수 있도록 구성되었습니다. 과제와 퀴즈를 통해 학습한 내용을 복습하고, 이해도를 높일 수 있습니다.