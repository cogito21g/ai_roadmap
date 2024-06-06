### 5주차 강의 계획안

#### 강의 주제: 고급 정렬 알고리즘
- 병합 정렬
- 퀵 정렬
- 힙 정렬

---

### 강의 내용

#### 1. 병합 정렬
- **개념**: 분할 정복 알고리즘으로, 배열을 반으로 나누고 정렬 후 병합
- **시간 복잡도**: O(n log n)

**예제**: 병합 정렬
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
    if (left >= right) return;

    int mid = left + (right - left) / 2;
    mergeSort(arr, left, mid);
    mergeSort(arr, mid + 1, right);
    merge(arr, left, mid, right);
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

#### 2. 퀵 정렬
- **개념**: 분할 정복 알고리즘으로, 피벗을 기준으로 배열을 두 부분으로 나눈 후 정렬
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

#### 3. 힙 정렬
- **개념**: 완전 이진 트리 구조를 이용한 정렬 알고리즘
- **시간 복잡도**: O(n log n)

**예제**: 힙 정렬
```cpp
#include <iostream>
using namespace std;

void heapify(int arr[], int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[largest])
        largest = left;

    if (right < n && arr[right] > arr[largest])
        largest = right;

    if (largest != i) {
        swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

void heapSort(int arr[], int n) {
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]);
        heapify(arr, i, 0);
    }
}

int main() {
    int arr[] = {12, 11, 13, 5, 6, 7};
    int n = sizeof(arr) / sizeof(arr[0]);

    heapSort(arr, n);

    cout << "Sorted array is \n";
    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";
    return 0;
}
```

---

### 과제

#### 과제 1: 병합 정렬 구현 및 시간 측정
사용자로부터 정수 배열을 입력받아 병합 정렬 알고리즘으로 정렬한 후, 정렬된 배열과 정렬에 소요된 시간을 출력하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;

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
    if (left >= right) return;

    int mid = left + (right - left) / 2;
    mergeSort(arr, left, mid);
    mergeSort(arr, mid + 1, right);
    merge(arr, left, mid, right);
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
    mergeSort(arr, 0, size - 1);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Sorted array: ";
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    cout << "Time taken by merge sort: " << duration.count() << " microseconds" << endl;

    delete[] arr;
    return 0;
}
```

**해설**:
1. 사용자로부터 배열 크기와 요소를 입력받습니다.
2. `mergeSort` 함수를 사용해 배열을 정렬합니다.
3. `chrono` 라이브러리를 사용해 정렬에 소요된 시간을 측정하고 출력합니다.

#### 과제 2: 퀵 정렬과 힙 정렬 비교
사용자로부터 정수 배열을 입력받아 퀵 정렬과 힙 정렬 알고리즘으로 각각 정렬한 후, 두 정렬 방법의 소요 시간을 비교하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++)

 {
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

void heapify(int arr[], int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[largest])
        largest = left;

    if (right < n && arr[right] > arr[largest])
        largest = right;

    if (largest != i) {
        swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

void heapSort(int arr[], int n) {
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]);
        heapify(arr, i, 0);
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
    quickSort(arr1, 0, size - 1);
    auto stop = high_resolution_clock::now();
    auto duration1 = duration_cast<microseconds>(stop - start);

    start = high_resolution_clock::now();
    heapSort(arr2, size);
    stop = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop - start);

    cout << "Time taken by quick sort: " << duration1.count() << " microseconds" << endl;
    cout << "Time taken by heap sort: " << duration2.count() << " microseconds" << endl;

    delete[] arr1;
    delete[] arr2;
    return 0;
}
```

**해설**:
1. 사용자로부터 배열 크기와 요소를 입력받습니다.
2. `quickSort`와 `heapSort` 함수를 사용해 배열을 각각 정렬합니다.
3. `chrono` 라이브러리를 사용해 두 정렬 방법의 소요 시간을 측정하고 비교합니다.

---

### 퀴즈

#### 퀴즈 1: 병합 정렬의 시간 복잡도는 무엇인가요?
1. O(n)
2. O(n log n)
3. O(n^2)
4. O(log n)

**정답**: 2. O(n log n)

#### 퀴즈 2: 퀵 정렬의 최악 시간 복잡도는 무엇인가요?
1. O(n)
2. O(n log n)
3. O(n^2)
4. O(log n)

**정답**: 3. O(n^2)

#### 퀴즈 3: 힙 정렬의 시간 복잡도는 무엇인가요?
1. O(n)
2. O(n log n)
3. O(n^2)
4. O(log n)

**정답**: 2. O(n log n)

이 계획안은 5주차에 필요한 고급 정렬 알고리즘의 개념과 실습을 통해 학습할 수 있도록 구성되었습니다. 과제와 퀴즈를 통해 학습한 내용을 복습하고, 이해도를 높일 수 있습니다.