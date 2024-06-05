
### C 언어 20주차 심화 교육과정 - 13주차: 알고리즘 기초

#### 13주차: 알고리즘 기초

**강의 목표:**
13주차의 목표는 기본적인 정렬 알고리즘과 탐색 알고리즘을 이해하고 구현하는 것입니다. 이러한 알고리즘의 원리와 효율성을 학습하여 데이터 처리를 최적화하는 능력을 기르는 데 중점을 둡니다.

**강의 구성:**

##### 1. 정렬 알고리즘
- **강의 내용:**
  - 정렬 알고리즘의 개념과 필요성
  - 버블 정렬 (Bubble Sort)
    - 알고리즘 설명 및 시간 복잡도
    - 구현 방법
  - 선택 정렬 (Selection Sort)
    - 알고리즘 설명 및 시간 복잡도
    - 구현 방법
  - 삽입 정렬 (Insertion Sort)
    - 알고리즘 설명 및 시간 복잡도
    - 구현 방법
- **실습:**
  - 버블 정렬 구현 예제 작성
    ```c
    #include <stdio.h>

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

    void printArray(int arr[], int size) {
        for (int i = 0; i < size; i++)
            printf("%d ", arr[i]);
        printf("\n");
    }

    int main() {
        int arr[] = {64, 34, 25, 12, 22, 11, 90};
        int n = sizeof(arr)/sizeof(arr[0]);
        bubbleSort(arr, n);
        printf("Sorted array: \n");
        printArray(arr, n);
        return 0;
    }
    ```

  - 선택 정렬 구현 예제 작성
    ```c
    #include <stdio.h>

    void selectionSort(int arr[], int n) {
        for (int i = 0; i < n-1; i++) {
            int min_idx = i;
            for (int j = i+1; j < n; j++) {
                if (arr[j] < arr[min_idx])
                    min_idx = j;
            }
            int temp = arr[min_idx];
            arr[min_idx] = arr[i];
            arr[i] = temp;
        }
    }

    void printArray(int arr[], int size) {
        for (int i = 0; i < size; i++)
            printf("%d ", arr[i]);
        printf("\n");
    }

    int main() {
        int arr[] = {64, 25, 12, 22, 11};
        int n = sizeof(arr)/sizeof(arr[0]);
        selectionSort(arr, n);
        printf("Sorted array: \n");
        printArray(arr, n);
        return 0;
    }
    ```

  - 삽입 정렬 구현 예제 작성
    ```c
    #include <stdio.h>

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

    void printArray(int arr[], int size) {
        for (int i = 0; i < size; i++)
            printf("%d ", arr[i]);
        printf("\n");
    }

    int main() {
        int arr[] = {12, 11, 13, 5, 6};
        int n = sizeof(arr)/sizeof(arr[0]);
        insertionSort(arr, n);
        printf("Sorted array: \n");
        printArray(arr, n);
        return 0;
    }
    ```

##### 2. 탐색 알고리즘
- **강의 내용:**
  - 탐색 알고리즘의 개념과 필요성
  - 선형 탐색 (Linear Search)
    - 알고리즘 설명 및 시간 복잡도
    - 구현 방법
  - 이진 탐색 (Binary Search)
    - 알고리즘 설명 및 시간 복잡도
    - 구현 방법
    - 이진 탐색을 위한 배열 정렬의 필요성
- **실습:**
  - 선형 탐색 구현 예제 작성
    ```c
    #include <stdio.h>

    int linearSearch(int arr[], int n, int x) {
        for (int i = 0; i < n; i++) {
            if (arr[i] == x)
                return i;
        }
        return -1;
    }

    int main() {
        int arr[] = {2, 3, 4, 10, 40};
        int x = 10;
        int n = sizeof(arr)/sizeof(arr[0]);
        int result = linearSearch(arr, n, x);
        if (result != -1)
            printf("Element found at index %d\n", result);
        else
            printf("Element not found\n");
        return 0;
    }
    ```

  - 이진 탐색 구현 예제 작성
    ```c
    #include <stdio.h>

    int binarySearch(int arr[], int l, int r, int x) {
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (arr[m] == x)
                return m;
            if (arr[m] < x)
                l = m + 1;
            else
                r = m - 1;
        }
        return -1;
    }

    int main() {
        int arr[] = {2, 3, 4, 10, 40};
        int x = 10;
        int n = sizeof(arr)/sizeof(arr[0]);
        int result = binarySearch(arr, 0, n - 1, x);
        if (result != -1)
            printf("Element found at index %d\n", result);
        else
            printf("Element not found\n");
        return 0;
    }
    ```

**과제:**
13주차 과제는 다음과 같습니다.
- 버블 정렬, 선택 정렬, 삽입 정렬을 각각 구현하고, 주어진 배열을 정렬하는 프로그램 작성
- 선형 탐색과 이진 탐색을 구현하여, 사용자가 입력한 숫자를 배열에서 찾는 프로그램 작성
- 주어진 배열을 정렬한 후, 이진 탐색을 통해 특정 값을 찾는 프로그램 작성

**퀴즈 및 해설:**

1. **버블 정렬의 시간 복잡도는 무엇인가요?**
   - 버블 정렬의 시간 복잡도는 최악과 평균 모두 O(n^2)입니다. 배열의 모든 요소를 반복적으로 비교하고 교환하기 때문입니다.

2. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    int arr[] = {64, 25, 12, 22, 11};
    int n = sizeof(arr)/sizeof(arr[0]);
    selectionSort(arr, n);
    printArray(arr, n);
    ```
   - 출력 결과는 `11 12 22 25 64`입니다. 선택 정렬은 배열을 오름차순으로 정렬합니다.

3. **이진 탐색을 사용하기 위한 전제 조건은 무엇인가요?**
   - 이진 탐색을 사용하기 위해서는 배열이 정렬되어 있어야 합니다. 이진 탐색은 중간 요소와 비교하여 탐색 범위를 반으로 줄여 나가므로, 정렬되지 않은 배열에서는 올바르게 동작하지 않습니다.

4. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    int arr[] = {2, 3, 4, 10, 40};
    int x = 10;
    int n = sizeof(arr)/sizeof(arr[0]);
    int result = binarySearch(arr, 0, n - 1, x);
    printf("%d\n", result);
    ```
   - 출력 결과는 `3`입니다. 이진 탐색은 배열에서 값 `10`을 찾아 인덱스 `3`을 반환합니다.

5. **선형 탐색과 이진 탐색의 차이점은 무엇인가요?**
   - 선형 탐색은 배열의 모든 요소를 처음부터 끝까지 순차적으로 비교하여 값을 찾습니다. 시간 복잡도는 O(n)입니다. 이진 탐색은 배열을 반씩 나누어 중간 요소와 비교하여 값을 찾습니다. 시간 복잡도는 O(log n)입니다.

**해설:**
1. 버블 정렬의 시간 복잡도는 최악과 평균 모두 O(n^2)입니다. 배열의 모든 요소를 반복적으로 비교하고 교환하기 때문입니다.
2. 선택 정렬은 배열을 오름차순으로 정렬하므로 출력 결과는 `11 12 22 25 64`입니다.
3. 이진 탐색을 사용하기 위해

서는 배열이 정렬되어 있어야 합니다. 이진 탐색은 중간 요소와 비교하여 탐색 범위를 반으로 줄여 나가므로, 정렬되지 않은 배열에서는 올바르게 동작하지 않습니다.
4. 이진 탐색은 배열에서 값 `10`을 찾아 인덱스 `3`을 반환하므로 출력 결과는 `3`입니다.
5. 선형 탐색은 배열의 모든 요소를 처음부터 끝까지 순차적으로 비교하여 값을 찾고, 이진 탐색은 배열을 반씩 나누어 중간 요소와 비교하여 값을 찾습니다. 선형 탐색의 시간 복잡도는 O(n), 이진 탐색의 시간 복잡도는 O(log n)입니다.

이 13주차 강의는 학생들이 기본적인 정렬 알고리즘과 탐색 알고리즘을 이해하고 구현하는 능력을 기를 수 있도록 도와줍니다.