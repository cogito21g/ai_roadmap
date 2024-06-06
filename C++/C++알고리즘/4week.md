### 4주차 강의 계획안

#### 강의 주제: 기본 정렬 알고리즘
- 버블 정렬
- 선택 정렬
- 삽입 정렬

---

### 강의 내용

#### 1. 버블 정렬
- **개념**: 인접한 두 요소를 비교하여 정렬하는 가장 간단한 정렬 알고리즘
- **시간 복잡도**: O(n^2)

**예제**: 버블 정렬
```cpp
#include <iostream>
using namespace std;

void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

int main() {
    int arr[] = {64, 25, 12, 22, 11};
    int n = sizeof(arr)/sizeof(arr[0]);
    bubbleSort(arr, n);
    cout << "Sorted array: ";
    for (int i=0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
    return 0;
}
```

#### 2. 선택 정렬
- **개념**: 주어진 리스트에서 가장 작은 요소를 선택하여 정렬된 부분의 맨 끝에 삽입하는 방식
- **시간 복잡도**: O(n^2)

**예제**: 선택 정렬
```cpp
#include <iostream>
using namespace std;

void selectionSort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        int min_idx = i;
        for (int j = i+1; j < n; j++) {
            if (arr[j] < arr[min_idx]) {
                min_idx = j;
            }
        }
        int temp = arr[min_idx];
        arr[min_idx] = arr[i];
        arr[i] = temp;
    }
}

int main() {
    int arr[] = {64, 25, 12, 22, 11};
    int n = sizeof(arr)/sizeof(arr[0]);
    selectionSort(arr, n);
    cout << "Sorted array: ";
    for (int i=0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
    return 0;
}
```

#### 3. 삽입 정렬
- **개념**: 정렬된 부분 배열에 새로운 요소를 삽입하여 정렬된 상태를 유지하는 방식
- **시간 복잡도**: O(n^2)

**예제**: 삽입 정렬
```cpp
#include <iostream>
using namespace std;

void insertionSort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

int main() {
    int arr[] = {64, 25, 12, 22, 11};
    int n = sizeof(arr)/sizeof(arr[0]);
    insertionSort(arr, n);
    cout << "Sorted array: ";
    for (int i=0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
    return 0;
}
```

---

### 과제

#### 과제 1: 버블 정렬 구현 및 시간 측정
사용자로부터 정수 배열을 입력받아 버블 정렬 알고리즘으로 정렬한 후, 정렬된 배열과 정렬에 소요된 시간을 출력하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;

void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

int main() {
    int size;
    cout << "Enter the size of the array: ";
    cin >> size;
    int* arr = new int[size];
    
    cout << "Enter the elements of the array: ";
    for (int i = 0; i < size; i++) {
        cin >> arr[i];
    }

    auto start = high_resolution_clock::now();
    bubbleSort(arr, size);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Sorted array: ";
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    cout << "Time taken by bubble sort: " << duration.count() << " microseconds" << endl;

    delete[] arr;
    return 0;
}
```

**해설**:
1. 사용자로부터 배열 크기와 요소를 입력받습니다.
2. `bubbleSort` 함수를 사용해 배열을 정렬합니다.
3. `chrono` 라이브러리를 사용해 정렬에 소요된 시간을 측정하고 출력합니다.

#### 과제 2: 선택 정렬과 삽입 정렬 비교
사용자로부터 정수 배열을 입력받아 선택 정렬과 삽입 정렬 알고리즘으로 각각 정렬한 후, 두 정렬 방법의 소요 시간을 비교하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;

void selectionSort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        int min_idx = i;
        for (int j = i+1; j < n; j++) {
            if (arr[j] < arr[min_idx]) {
                min_idx = j;
            }
        }
        int temp = arr[min_idx];
        arr[min_idx] = arr[i];
        arr[i] = temp;
    }
}

void insertionSort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

int main() {
    int size;
    cout << "Enter the size of the array: ";
    cin >> size;
    int* arr1 = new int[size];
    int* arr2 = new int[size];
    
    cout << "Enter the elements of the array: ";
    for (int i = 0; i < size; i++) {
        cin >> arr1[i];
        arr2[i] = arr1[i];
    }

    auto start = high_resolution_clock::now();
    selectionSort(arr1, size);
    auto stop = high_resolution_clock::now();
    auto duration1 = duration_cast<microseconds>(stop - start);

    start = high_resolution_clock::now();
    insertionSort(arr2, size);
    stop = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop - start);

    cout << "Time taken by selection sort: " << duration1.count() << " microseconds" << endl;
    cout << "Time taken by insertion sort: " << duration2.count() << " microseconds" << endl;

    delete[] arr1;
    delete[] arr2;
    return 0;
}
```

**해설**:
1. 사용자로부터 배열 크기와 요소를 입력받습니다.
2. `selectionSort`와 `insertionSort` 함수를 사용해 배열을 각각 정렬합니다.
3. `chrono` 라이브러리를 사용해 두 정렬 방법의 소요 시간을 측정하고 비교합니다.

---

### 퀴즈

#### 퀴즈 1: 다음 중 버블 정렬의 시간 복잡도는 무엇인가요?
1. O(n)
2. O(n log n)
3. O(n^2)
4. O(log n)

**정답**: 3. O(n^2)

#### 퀴즈 2: 선택 정렬의 기본 원리는 무엇인가요?
1. 인접한 두 요소를 비교하여 정렬한다.
2. 배열의 첫 번째 요소를 선택하여 정렬한다.
3. 주어진 리스트에서 가장 작은 요소를 선택하여 정렬된 부분의 맨 끝에 삽입한다.
4. 새로운 요소를 삽입하여 정렬된 상태를 유지한다.

**정답**: 3. 주어진 리스트에서 가장 작은 요소를 선택하여 정렬된 부분의 맨 끝에 삽입한다.

#### 퀴즈 3: 삽입 정렬

에서 배열의 몇 번째 요소부터 정렬을 시작하나요?
1. 첫 번째 요소
2. 두 번째 요소
3. 세 번째 요소
4. 마지막 요소

**정답**: 2. 두 번째 요소

이 계획안은 4주차에 필요한 기본 정렬 알고리즘의 개념과 실습을 통해 학습할 수 있도록 구성되었습니다. 과제와 퀴즈를 통해 학습한 내용을 복습하고, 이해도를 높일 수 있습니다.