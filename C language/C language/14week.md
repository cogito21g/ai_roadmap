### C 언어 20주차 심화 교육과정 - 14주차: 고급 알고리즘

#### 14주차: 고급 알고리즘

**강의 목표:**
14주차의 목표는 고급 정렬 알고리즘과 해시 테이블의 개념 및 구현을 이해하는 것입니다. 이러한 알고리즘을 통해 효율적인 데이터 처리와 검색을 수행하는 능력을 기르는 데 중점을 둡니다.

**강의 구성:**

##### 1. 퀵 정렬 (Quick Sort)
- **강의 내용:**
  - 퀵 정렬의 개념
    - 분할 정복 알고리즘의 일종
    - 피벗을 중심으로 작은 값과 큰 값을 분할
  - 퀵 정렬의 시간 복잡도
    - 평균 시간 복잡도: O(n log n)
    - 최악 시간 복잡도: O(n^2)
  - 퀵 정렬의 구현 방법
- **실습:**
  - 퀵 정렬 구현 예제 작성
    ```c
    #include <stdio.h>

    void swap(int* a, int* b) {
        int t = *a;
        *a = *b;
        *b = t;
    }

    int partition(int arr[], int low, int high) {
        int pivot = arr[high];
        int i = (low - 1);

        for (int j = low; j < high; j++) {
            if (arr[j] < pivot) {
                i++;
                swap(&arr[i], &arr[j]);
            }
        }
        swap(&arr[i + 1], &arr[high]);
        return (i + 1);
    }

    void quickSort(int arr[], int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);

            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }

    void printArray(int arr[], int size) {
        for (int i = 0; i < size; i++)
            printf("%d ", arr[i]);
        printf("\n");
    }

    int main() {
        int arr[] = {10, 7, 8, 9, 1, 5};
        int n = sizeof(arr) / sizeof(arr[0]);
        quickSort(arr, 0, n - 1);
        printf("Sorted array: ");
        printArray(arr, n);
        return 0;
    }
    ```

##### 2. 합병 정렬 (Merge Sort)
- **강의 내용:**
  - 합병 정렬의 개념
    - 분할 정복 알고리즘의 일종
    - 배열을 분할하고 병합하여 정렬
  - 합병 정렬의 시간 복잡도
    - 모든 경우에서 시간 복잡도: O(n log n)
  - 합병 정렬의 구현 방법
- **실습:**
  - 합병 정렬 구현 예제 작성
    ```c
    #include <stdio.h>

    void merge(int arr[], int l, int m, int r) {
        int i, j, k;
        int n1 = m - l + 1;
        int n2 = r - m;

        int L[n1], R[n2];

        for (i = 0; i < n1; i++)
            L[i] = arr[l + i];
        for (j = 0; j < n2; j++)
            R[j] = arr[m + 1 + j];

        i = 0;
        j = 0;
        k = l;
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

    void mergeSort(int arr[], int l, int r) {
        if (l < r) {
            int m = l + (r - l) / 2;

            mergeSort(arr, l, m);
            mergeSort(arr, m + 1, r);

            merge(arr, l, m, r);
        }
    }

    void printArray(int arr[], int size) {
        for (int i = 0; i < size; i++)
            printf("%d ", arr[i]);
        printf("\n");
    }

    int main() {
        int arr[] = {12, 11, 13, 5, 6, 7};
        int arr_size = sizeof(arr) / sizeof(arr[0]);

        printf("Given array: ");
        printArray(arr, arr_size);

        mergeSort(arr, 0, arr_size - 1);

        printf("Sorted array: ");
        printArray(arr, arr_size);
        return 0;
    }
    ```

##### 3. 해시 테이블 (Hash Table)
- **강의 내용:**
  - 해시 테이블의 개념
    - 해시 함수를 사용하여 키를 인덱스로 변환하여 데이터 저장
  - 해시 함수의 설계
    - 좋은 해시 함수의 특징: 균등 분포, 효율성
  - 충돌 처리 방법
    - 개방 주소법 (Open Addressing)
    - 체이닝 (Chaining)
- **실습:**
  - 체이닝을 이용한 해시 테이블 구현 예제 작성
    ```c
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    struct Node {
        int data;
        struct Node* next;
    };

    struct HashTable {
        int size;
        struct Node** table;
    };

    struct HashTable* createTable(int size) {
        struct HashTable* newTable = (struct HashTable*)malloc(sizeof(struct HashTable));
        newTable->size = size;
        newTable->table = (struct Node**)malloc(sizeof(struct Node*) * size);
        for (int i = 0; i < size; i++)
            newTable->table[i] = NULL;
        return newTable;
    }

    int hashFunction(struct HashTable* table, int key) {
        return key % table->size;
    }

    void insert(struct HashTable* table, int key) {
        int hashIndex = hashFunction(table, key);
        struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
        newNode->data = key;
        newNode->next = table->table[hashIndex];
        table->table[hashIndex] = newNode;
    }

    struct Node* search(struct HashTable* table, int key) {
        int hashIndex = hashFunction(table, key);
        struct Node* currentNode = table->table[hashIndex];
        while (currentNode != NULL) {
            if (currentNode->data == key)
                return currentNode;
            currentNode = currentNode->next;
        }
        return NULL;
    }

    void display(struct HashTable* table) {
        for (int i = 0; i < table->size; i++) {
            struct Node* currentNode = table->table[i];
            printf("Bucket %d: ", i);
            while (currentNode != NULL) {
                printf("%d -> ", currentNode->data);
                currentNode = currentNode->next;
            }
            printf("NULL\n");
        }
    }

    int main() {
        struct HashTable* hashTable = createTable(10);

        insert(hashTable, 1);
        insert(hashTable, 2);
        insert(hashTable, 42);
        insert(hashTable, 4);
        insert(hashTable, 12);
        insert(hashTable, 14);
        insert(hashTable, 17);
        insert(hashTable, 13);
        insert(hashTable, 37);

        display(hashTable);

        struct Node* result = search(hashTable, 37);
        if (result != NULL)
            printf("Element found: %d\n", result->data);
        else
            printf("Element not found\n");

        return 0;
    }
    ```

**과제:**
14주차 과제는 다음과 같습니다.
- 퀵 정렬과 합병 정렬을 구현하고, 주어진 배열을 정렬하는 프로그램 작성
- 해시 테이블을 구현하여, 사용자가 입력한 키-값 쌍을 저장하고 검색하는 프로그램 작성
- 주어진 데이터를 해시 테이블에 저장한 후, 특정 키를 검색하여 결과를 출력하는 프로그램 작성

**퀴즈 및 해설:**

1. **퀵 정렬의 시간 복잡도는 무엇인가요?**
   - 평균 시간 복잡도: O(n log n)
   - 최악 시간 복잡도: O(n^2)
   - 퀵 정렬은 분할 정복 알고리즘으로, 평균적으로 매우 효율적이지만, 피벗 선택에 따라 최악의 경우 O(n^2)의 시간 복잡도를 가집니다.

2. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    int arr[] = {12, 11, 13, 5, 6,

 7};
    int n = sizeof(arr)/sizeof(arr[0]);
    mergeSort(arr, 0, n - 1);
    printArray(arr, n);
    ```
   - 출력 결과는 `5 6 7 11 12 13`입니다. 합병 정렬은 배열을 오름차순으로 정렬합니다.

3. **해시 테이블에서 충돌 처리 방법 중 체이닝의 개념을 설명하세요.**
   - 체이닝은 해시 테이블에서 동일한 해시 값을 가진 여러 요소를 연결 리스트로 연결하여 충돌을 처리하는 방법입니다. 해시 충돌 시, 새로운 요소는 해당 버킷의 연결 리스트에 추가됩니다.

4. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    struct HashTable* table = createTable(10);
    insert(table, 42);
    insert(table, 52);
    insert(table, 12);
    display(table);
    ```
   - 출력 결과는 다음과 같습니다 (구체적인 해시 함수와 충돌 여부에 따라 다를 수 있음):
     ```
     Bucket 0: NULL
     Bucket 1: NULL
     Bucket 2: 12 -> NULL
     Bucket 3: NULL
     Bucket 4: NULL
     Bucket 5: NULL
     Bucket 6: NULL
     Bucket 7: 52 -> 42 -> NULL
     Bucket 8: NULL
     Bucket 9: NULL
     ```

5. **합병 정렬의 시간 복잡도는 무엇인가요?**
   - 합병 정렬의 시간 복잡도는 모든 경우에서 O(n log n)입니다. 배열을 분할하고 병합하는 과정에서 균등하게 시간을 소모하기 때문입니다.

**해설:**
1. 퀵 정렬의 평균 시간 복잡도는 O(n log n), 최악 시간 복잡도는 O(n^2)입니다. 피벗 선택에 따라 최악의 경우 발생할 수 있습니다.
2. 합병 정렬은 배열을 오름차순으로 정렬하므로 출력 결과는 `5 6 7 11 12 13`입니다.
3. 체이닝은 해시 테이블에서 동일한 해시 값을 가진 요소들을 연결 리스트로 연결하여 충돌을 처리하는 방법입니다.
4. 해시 함수와 충돌 여부에 따라 해시 테이블의 내용이 달라질 수 있지만, 기본적으로 각 버킷에 요소가 연결 리스트 형태로 저장됩니다.
5. 합병 정렬의 시간 복잡도는 모든 경우에서 O(n log n)입니다. 배열을 분할하고 병합하는 과정에서 균등하게 시간을 소모하기 때문입니다.

이 14주차 강의는 학생들이 고급 정렬 알고리즘과 해시 테이블의 개념 및 구현을 이해하고, 이를 활용하여 효율적인 데이터 처리를 수행하는 능력을 기를 수 있도록 도와줍니다.