### 11주차 강의 계획안

#### 강의 주제: 분할 정복 알고리즘
- 분할 정복의 개념
- 대표 문제 (퀵 정렬, 병합 정렬, 이진 검색 등)

---

### 강의 내용

#### 1. 분할 정복의 개념
- **개념**: 문제를 작은 하위 문제로 나누어 각각을 해결한 후, 결과를 합쳐서 전체 문제를 해결하는 알고리즘
- **단계**:
  1. 분할 (Divide): 문제를 더 작은 하위 문제로 나눔
  2. 정복 (Conquer): 하위 문제를 해결
  3. 합병 (Combine): 하위 문제의 해를 합쳐서 원래 문제의 해를 구함

#### 2. 퀵 정렬 (Quick Sort)
- **개념**: 피벗을 기준으로 배열을 두 부분으로 나누어 정렬하는 알고리즘
- **시간 복잡도**: 평균 O(n log n), 최악 O(n^2)

**예제**: 퀵 정렬
```cpp
#include <iostream>
using namespace std;

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return (i + 1);
}

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int main() {
    int arr[] = {10, 7, 8, 9, 1, 5};
    int n = sizeof(arr) / sizeof(arr[0]);
    quickSort(arr, 0, n - 1);
    cout << "Sorted array: \n";
    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";
    return 0;
}
```

#### 3. 병합 정렬 (Merge Sort)
- **개념**: 배열을 반으로 나누어 각각 정렬한 후, 다시 병합하여 전체를 정렬하는 알고리즘
- **시간 복잡도**: O(n log n)

**예제**: 병합 정렬
```cpp
#include <iostream>
using namespace std;

void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int L[n1], R[n2];

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

int main() {
    int arr[] = {12, 11, 13, 5, 6, 7};
    int arr_size = sizeof(arr) / sizeof(arr[0]);

    cout << "Given array is \n";
    for (int i = 0; i < arr_size; i++)
        cout << arr[i] << " ";

    mergeSort(arr, 0, arr_size - 1);

    cout << "\nSorted array is \n";
    for (int i = 0; i < arr_size; i++)
        cout << arr[i] << " ";
    return 0;
}
```

#### 4. 이진 검색 (Binary Search)
- **개념**: 정렬된 배열에서 중간 요소와 비교하여 원하는 값을 찾는 알고리즘
- **시간 복잡도**: O(log n)

**예제**: 이진 검색
```cpp
#include <iostream>
using namespace std;

int binarySearch(int arr[], int left, int right, int x) {
    if (right >= left) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == x)
            return mid;

        if (arr[mid] > x)
            return binarySearch(arr, left, mid - 1, x);

        return binarySearch(arr, mid + 1, right, x);
    }

    return -1;
}

int main() {
    int arr[] = {2, 3, 4, 10, 40};
    int n = sizeof(arr) / sizeof(arr[0]);
    int x = 10;
    int result = binarySearch(arr, 0, n - 1, x);
    (result == -1) ? cout << "Element is not present in array"
                   : cout << "Element is present at index " << result;
    return 0;
}
```

---

### 과제

#### 과제 1: 퀵 정렬 구현 및 시간 측정
사용자로부터 정수 배열을 입력받아 퀵 정렬 알고리즘으로 정렬한 후, 정렬된 배열과 정렬에 소요된 시간을 출력하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return (i + 1);
}

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
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
    quickSort(arr, 0, size - 1);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Sorted array: ";
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    cout << "Time taken by quick sort: " << duration.count() << " microseconds" << endl;

    delete[] arr;
    return 0;
}
```

**해설**:
1. 사용자로부터 배열 크기와 요소를 입력받습니다.
2. `quickSort` 함수를 사용해 배열을 정렬합니다.
3. `chrono` 라이브러리를 사용해 정렬에 소요된 시간을 측정하고 출력합니다.

#### 과제 2: 병합 정렬과 이진 검색 구현
사용자로부터 정수 배열을 입력받아 병합 정렬 알고리즘으로 정렬한 후, 정렬된 배열에서 특정 값을 이진 검색 알고리즘으로 찾는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
using namespace std;

void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int* L = new int[n1];
    int* R = new int[n2];

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }



    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    delete[] L;
    delete[] R;
}

void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

int binarySearch(int arr[], int left, int right, int x) {
    if (right >= left) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == x)
            return mid;

        if (arr[mid] > x)
            return binarySearch(arr, left, mid - 1, x);

        return binarySearch(arr, mid + 1, right, x);
    }

    return -1;
}

int main() {
    int size, x;
    cout << "Enter the size of the array: ";
    cin >> size;
    int* arr = new int[size];
    
    cout << "Enter the elements of the array: ";
    for (int i = 0; i < size; i++) {
        cin >> arr[i];
    }

    mergeSort(arr, 0, size - 1);

    cout << "Sorted array: ";
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    cout << "Enter the element to search: ";
    cin >> x;
    int result = binarySearch(arr, 0, size - 1, x);

    (result == -1) ? cout << "Element is not present in array"
                   : cout << "Element is present at index " << result;
    cout << endl;

    delete[] arr;
    return 0;
}
```

**해설**:
1. 사용자로부터 배열 크기와 요소를 입력받습니다.
2. `mergeSort` 함수를 사용해 배열을 정렬합니다.
3. `binarySearch` 함수를 사용해 정렬된 배열에서 특정 값을 찾습니다.

---

### 퀴즈

#### 퀴즈 1: 분할 정복 알고리즘의 세 가지 주요 단계는 무엇인가요?
1. 정의, 나누기, 합병
2. 나누기, 정복, 합병
3. 나누기, 해결, 결합
4. 분할, 해결, 병합

**정답**: 2. 나누기, 정복, 합병

#### 퀴즈 2: 퀵 정렬의 평균 시간 복잡도는 무엇인가요?
1. O(n)
2. O(n log n)
3. O(n^2)
4. O(log n)

**정답**: 2. O(n log n)

#### 퀴즈 3: 이진 검색의 시간 복잡도는 무엇인가요?
1. O(n)
2. O(n log n)
3. O(log n)
4. O(1)

**정답**: 3. O(log n)

이 계획안은 11주차에 필요한 분할 정복 알고리즘의 개념과 실습을 통해 학습할 수 있도록 구성되었습니다. 과제와 퀴즈를 통해 학습한 내용을 복습하고, 이해도를 높일 수 있습니다.