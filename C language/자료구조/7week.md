### 자료구조 교육과정 - 7주차: 힙과 우선순위 큐

**강의 목표:**
힙의 개념과 구현 방법을 학습하고, 우선순위 큐를 활용한 응용을 이해합니다.

**강의 구성:**

#### 7. 힙과 우선순위 큐

### 힙

**강의 내용:**
- 힙의 개념
  - 힙이란 무엇인가?
  - 힙의 종류 (최대 힙, 최소 힙)
- 힙의 구조
  - 완전 이진 트리
  - 힙 속성
- 힙의 연산
  - 삽입 연산
  - 삭제 연산 (최대값/최소값 삭제)

**실습:**
- 힙 구현 예제 (최소 힙)
  ```c
  #include <stdio.h>
  #include <stdlib.h>

  void swap(int *x, int *y) {
      int temp = *x;
      *x = *y;
      *y = temp;
  }

  struct MinHeap {
      int *harr;
      int capacity;
      int heap_size;
  };

  struct MinHeap* createMinHeap(int capacity) {
      struct MinHeap* minHeap = (struct MinHeap*) malloc(sizeof(struct MinHeap));
      minHeap->heap_size = 0;
      minHeap->capacity = capacity;
      minHeap->harr = (int*) malloc(capacity * sizeof(int));
      return minHeap;
  }

  void MinHeapify(struct MinHeap* minHeap, int idx) {
      int smallest = idx;
      int left = 2 * idx + 1;
      int right = 2 * idx + 2;

      if (left < minHeap->heap_size && minHeap->harr[left] < minHeap->harr[smallest])
          smallest = left;

      if (right < minHeap->heap_size && minHeap->harr[right] < minHeap->harr[smallest])
          smallest = right;

      if (smallest != idx) {
          swap(&minHeap->harr[smallest], &minHeap->harr[idx]);
          MinHeapify(minHeap, smallest);
      }
  }

  void insertKey(struct MinHeap* minHeap, int k) {
      if (minHeap->heap_size == minHeap->capacity) {
          printf("Overflow: Could not insertKey\n");
          return;
      }

      int i = minHeap->heap_size++;
      minHeap->harr[i] = k;

      while (i != 0 && minHeap->harr[(i - 1) / 2] > minHeap->harr[i]) {
          swap(&minHeap->harr[i], &minHeap->harr[(i - 1) / 2]);
          i = (i - 1) / 2;
      }
  }

  int extractMin(struct MinHeap* minHeap) {
      if (minHeap->heap_size <= 0)
          return INT_MAX;
      if (minHeap->heap_size == 1) {
          minHeap->heap_size--;
          return minHeap->harr[0];
      }

      int root = minHeap->harr[0];
      minHeap->harr[0] = minHeap->harr[minHeap->heap_size - 1];
      minHeap->heap_size--;
      MinHeapify(minHeap, 0);

      return root;
  }

  void decreaseKey(struct MinHeap* minHeap, int i, int new_val) {
      minHeap->harr[i] = new_val;
      while (i != 0 && minHeap->harr[(i - 1) / 2] > minHeap->harr[i]) {
          swap(&minHeap->harr[i], &minHeap->harr[(i - 1) / 2]);
          i = (i - 1) / 2;
      }
  }

  void deleteKey(struct MinHeap* minHeap, int i) {
      decreaseKey(minHeap, i, INT_MIN);
      extractMin(minHeap);
  }

  void printHeap(struct MinHeap* minHeap) {
      for (int i = 0; i < minHeap->heap_size; i++) {
          printf("%d ", minHeap->harr[i]);
      }
      printf("\n");
  }

  int main() {
      struct MinHeap* minHeap = createMinHeap(11);
      insertKey(minHeap, 3);
      insertKey(minHeap, 2);
      deleteKey(minHeap, 1);
      insertKey(minHeap, 15);
      insertKey(minHeap, 5);
      insertKey(minHeap, 4);
      insertKey(minHeap, 45);

      printf("Min-Heap array: ");
      printHeap(minHeap);

      printf("%d extracted from heap\n", extractMin(minHeap));
      printHeap(minHeap);

      decreaseKey(minHeap, 2, 1);
      printf("Heap after decrease key: ");
      printHeap(minHeap);

      return 0;
  }
  ```

**과제:**
- 최소 힙에서 특정 값을 삽입하고 삭제하는 함수를 각각 구현
- 최소 힙에서 최솟값을 추출하는 함수를 구현하고, 해당 값이 올바르게 추출되는지 확인

**퀴즈 및 해설:**

1. **힙이란 무엇인가요?**
   - 힙은 완전 이진 트리로, 각 노드의 키 값이 자식 노드의 키 값보다 작거나 같은(최소 힙) 또는 크거나 같은(최대 힙) 특성을 가집니다. 힙은 우선순위 큐를 구현하는 데 사용됩니다.

2. **힙의 삽입 연산의 시간 복잡도는 무엇인가요?**
   - 힙의 삽입 연산의 시간 복잡도는 O(log n)입니다. 삽입 후 힙 속성을 유지하기 위해 부모 노드와의 비교 및 교환을 반복해야 하기 때문입니다.

3. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    struct MinHeap* minHeap = createMinHeap(11);
    insertKey(minHeap, 3);
    insertKey(minHeap, 2);
    deleteKey(minHeap, 1);
    insertKey(minHeap, 15);
    insertKey(minHeap, 5);
    insertKey(minHeap, 4);
    insertKey(minHeap, 45);

    printf("Min-Heap array: ");
    printHeap(minHeap);

    printf("%d extracted from heap\n", extractMin(minHeap));
    printHeap(minHeap);

    decreaseKey(minHeap, 2, 1);
    printf("Heap after decrease key: ");
    printHeap(minHeap);
    ```
   - 출력 결과:
     ```
     Min-Heap array: 2 3 45 15 5 4 
     2 extracted from heap
     3 4 45 15 5 
     Heap after decrease key: 1 4 3 15 5 
     ```

**해설:**
1. **힙의 정의**는 완전 이진 트리로, 각 노드의 키 값이 자식 노드의 키 값보다 작거나 같은(최소 힙) 또는 크거나 같은(최대 힙) 특성을 가집니다. 힙은 우선순위 큐를 구현하는 데 사용됩니다.
2. **힙의 삽입 연산의 시간 복잡도**는 O(log n)입니다. 삽입 후 힙 속성을 유지하기 위해 부모 노드와의 비교 및 교환을 반복해야 하기 때문입니다.
3. **코드 출력 결과**는 최소 힙의 삽입, 삭제, 최솟값 추출, 키 값 감소 연산 후의 상태를 보여줍니다. 힙의 각 연산은 힙 속성을 유지하면서 수행됩니다.

---

### 우선순위 큐

**강의 내용:**
- 우선순위 큐의 개념
  - 우선순위 큐란?
  - 우선순위 큐의 응용
- 우선순위 큐의 구조
  - 힙을 이용한 우선순위 큐 구현
- 우선순위 큐의 연산
  - 삽입 연산
  - 삭제 연산 (최대값/최소값 삭제)
  - 추출 연산 (최대값/최소값 추출)

**실습:**
- 우선순위 큐 구현 예제 (최대 힙)
  ```c
  #include <stdio.h>
  #include <stdlib.h>

  void swap(int *x, int *y) {
      int temp = *x;
      *x = *y;
      *y = temp;
  }

  struct MaxHeap {
      int *harr;
      int capacity;
      int heap_size;
  };

  struct MaxHeap* createMaxHeap(int capacity) {
      struct MaxHeap* maxHeap = (struct MaxHeap*) malloc(sizeof(struct MaxHeap));
      maxHeap->heap_size = 0;
      maxHeap->capacity = capacity;
      maxHeap->harr = (int*) malloc(capacity * sizeof(int));
      return maxHeap;
  }

  void MaxHeapify(struct MaxHeap* maxHeap, int idx) {
      int largest = idx;
      int left = 2 * idx + 1

;
      int right = 2 * idx + 2;

      if (left < maxHeap->heap_size && maxHeap->harr[left] > maxHeap->harr[largest])
          largest = left;

      if (right < maxHeap->heap_size && maxHeap->harr[right] > maxHeap->harr[largest])
          largest = right;

      if (largest != idx) {
          swap(&maxHeap->harr[largest], &maxHeap->harr[idx]);
          MaxHeapify(maxHeap, largest);
      }
  }

  void insertKey(struct MaxHeap* maxHeap, int k) {
      if (maxHeap->heap_size == maxHeap->capacity) {
          printf("Overflow: Could not insertKey\n");
          return;
      }

      int i = maxHeap->heap_size++;
      maxHeap->harr[i] = k;

      while (i != 0 && maxHeap->harr[(i - 1) / 2] < maxHeap->harr[i]) {
          swap(&maxHeap->harr[i], &maxHeap->harr[(i - 1) / 2]);
          i = (i - 1) / 2;
      }
  }

  int extractMax(struct MaxHeap* maxHeap) {
      if (maxHeap->heap_size <= 0)
          return INT_MIN;
      if (maxHeap->heap_size == 1) {
          maxHeap->heap_size--;
          return maxHeap->harr[0];
      }

      int root = maxHeap->harr[0];
      maxHeap->harr[0] = maxHeap->harr[maxHeap->heap_size - 1];
      maxHeap->heap_size--;
      MaxHeapify(maxHeap, 0);

      return root;
  }

  void decreaseKey(struct MaxHeap* maxHeap, int i, int new_val) {
      maxHeap->harr[i] = new_val;
      while (i != 0 && maxHeap->harr[(i - 1) / 2] < maxHeap->harr[i]) {
          swap(&maxHeap->harr[i], &maxHeap->harr[(i - 1) / 2]);
          i = (i - 1) / 2;
      }
  }

  void deleteKey(struct MaxHeap* maxHeap, int i) {
      decreaseKey(maxHeap, i, INT_MAX);
      extractMax(maxHeap);
  }

  void printHeap(struct MaxHeap* maxHeap) {
      for (int i = 0; i < maxHeap->heap_size; i++) {
          printf("%d ", maxHeap->harr[i]);
      }
      printf("\n");
  }

  int main() {
      struct MaxHeap* maxHeap = createMaxHeap(11);
      insertKey(maxHeap, 3);
      insertKey(maxHeap, 2);
      deleteKey(maxHeap, 1);
      insertKey(maxHeap, 15);
      insertKey(maxHeap, 5);
      insertKey(maxHeap, 4);
      insertKey(maxHeap, 45);

      printf("Max-Heap array: ");
      printHeap(maxHeap);

      printf("%d extracted from heap\n", extractMax(maxHeap));
      printHeap(maxHeap);

      decreaseKey(maxHeap, 2, 1);
      printf("Heap after decrease key: ");
      printHeap(maxHeap);

      return 0;
  }
  ```

**과제:**
- 최대 힙을 이용하여 우선순위 큐를 구현하고, 특정 값을 삽입하고 삭제하는 함수를 각각 구현
- 최대 힙을 이용하여 우선순위 큐에서 최대값을 추출하는 함수를 구현하고, 해당 값이 올바르게 추출되는지 확인

**퀴즈 및 해설:**

1. **우선순위 큐란 무엇인가요?**
   - 우선순위 큐는 각 요소에 우선순위를 부여하여 우선순위가 높은 요소가 먼저 처리되도록 하는 데이터 구조입니다. 일반적으로 힙을 이용하여 구현됩니다.

2. **우선순위 큐의 삽입 연산의 시간 복잡도는 무엇인가요?**
   - 우선순위 큐의 삽입 연산의 시간 복잡도는 O(log n)입니다. 삽입 후 우선순위 큐의 속성을 유지하기 위해 힙 속성을 유지해야 하기 때문입니다.

3. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    struct MaxHeap* maxHeap = createMaxHeap(11);
    insertKey(maxHeap, 3);
    insertKey(maxHeap, 2);
    deleteKey(maxHeap, 1);
    insertKey(maxHeap, 15);
    insertKey(maxHeap, 5);
    insertKey(maxHeap, 4);
    insertKey(maxHeap, 45);

    printf("Max-Heap array: ");
    printHeap(maxHeap);

    printf("%d extracted from heap\n", extractMax(maxHeap));
    printHeap(maxHeap);

    decreaseKey(maxHeap, 2, 1);
    printf("Heap after decrease key: ");
    printHeap(maxHeap);
    ```
   - 출력 결과:
     ```
     Max-Heap array: 45 5 15 3 2 4 
     45 extracted from heap
     15 5 4 3 2 
     Heap after decrease key: 15 5 1 3 2 
     ```

**해설:**
1. **우선순위 큐의 정의**는 각 요소에 우선순위를 부여하여 우선순위가 높은 요소가 먼저 처리되도록 하는 데이터 구조입니다. 일반적으로 힙을 이용하여 구현됩니다.
2. **우선순위 큐의 삽입 연산의 시간 복잡도**는 O(log n)입니다. 삽입 후 우선순위 큐의 속성을 유지하기 위해 힙 속성을 유지해야 하기 때문입니다.
3. **코드 출력 결과**는 최대 힙을 이용한 우선순위 큐의 삽입, 삭제, 최대값 추출, 키 값 감소 연산 후의 상태를 보여줍니다. 힙의 각 연산은 힙 속성을 유지하면서 수행됩니다.

---
