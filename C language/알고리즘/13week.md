### 알고리즘 교육과정 - 13주차: 고급 자료구조와 알고리즘

**강의 목표:**
고급 자료구조의 개념과 원리를 이해하고, 이를 활용한 다양한 알고리즘을 학습합니다. 세그먼트 트리, 펜윅 트리, 트라이(Trie) 등을 중심으로 복잡한 문제를 효율적으로 해결하는 방법을 학습합니다.

**강의 구성:**

#### 13. 고급 자료구조와 알고리즘

**강의 내용:**
- 세그먼트 트리
- 펜윅 트리 (BIT)
- 트라이 (Trie)

**실습:**
- 고급 자료구조 구현 및 성능 분석

### 세그먼트 트리

**강의 내용:**
- 세그먼트 트리의 개념
  - 배열의 구간 합, 구간 최소값, 구간 최대값 등을 빠르게 계산하는 자료구조
- 세그먼트 트리의 구축과 업데이트
- 세그먼트 트리의 시간 복잡도
  - 구축: O(n)
  - 업데이트 및 질의: O(log n)

**실습:**
- 세그먼트 트리 구현 예제
  ```c
  #include <stdio.h>
  #include <math.h>
  #include <stdlib.h>

  int minVal(int x, int y) {
      return (x < y) ? x : y;
  }

  int getMid(int s, int e) {
      return s + (e - s) / 2;
  }

  int RMQUtil(int* st, int ss, int se, int qs, int qe, int index) {
      if (qs <= ss && qe >= se)
          return st[index];

      if (se < qs || ss > qe)
          return INT_MAX;

      int mid = getMid(ss, se);
      return minVal(RMQUtil(st, ss, mid, qs, qe, 2 * index + 1),
                    RMQUtil(st, mid + 1, se, qs, qe, 2 * index + 2));
  }

  void updateValueUtil(int* st, int ss, int se, int i, int new_val, int index) {
      if (i < ss || i > se)
          return;

      if (ss != se) {
          int mid = getMid(ss, se);
          updateValueUtil(st, ss, mid, i, new_val, 2 * index + 1);
          updateValueUtil(st, mid + 1, se, i, new_val, 2 * index + 2);
          st[index] = minVal(st[2 * index + 1], st[2 * index + 2]);
      }
  }

  void updateValue(int arr[], int* st, int n, int i, int new_val) {
      if (i < 0 || i > n - 1) {
          printf("Invalid Input");
          return;
      }

      arr[i] = new_val;
      updateValueUtil(st, 0, n - 1, i, new_val, 0);
  }

  int* constructSTUtil(int arr[], int ss, int se, int* st, int si) {
      if (ss == se) {
          st[si] = arr[ss];
          return st;
      }

      int mid = getMid(ss, se);
      st[si] = minVal(constructSTUtil(arr, ss, mid, st, si * 2 + 1),
                      constructSTUtil(arr, mid + 1, se, st, si * 2 + 2));
      return st;
  }

  int* constructST(int arr[], int n) {
      int x = (int)(ceil(log2(n)));
      int max_size = 2 * (int)pow(2, x) - 1;
      int* st = (int*)malloc(max_size * sizeof(int));
      constructSTUtil(arr, 0, n - 1, st, 0);
      return st;
  }

  int RMQ(int* st, int n, int qs, int qe) {
      if (qs < 0 || qe > n - 1 || qs > qe) {
          printf("Invalid Input");
          return -1;
      }
      return RMQUtil(st, 0, n - 1, qs, qe, 0);
  }

  int main() {
      int arr[] = {1, 3, 2, 7, 9, 11};
      int n = sizeof(arr) / sizeof(arr[0]);
      int* st = constructST(arr, n);

      printf("Minimum value in range [1, 5] is %d\n", RMQ(st, n, 1, 5));

      updateValue(arr, st, n, 3, 6);

      printf("Minimum value in range [1, 5] after update is %d\n", RMQ(st, n, 1, 5));

      free(st);
      return 0;
  }
  ```

### 펜윅 트리 (BIT)

**강의 내용:**
- 펜윅 트리의 개념
  - 배열의 구간 합을 효율적으로 계산하고 업데이트하는 자료구조
- 펜윅 트리의 구축과 업데이트
- 펜윅 트리의 시간 복잡도
  - 업데이트 및 질의: O(log n)

**실습:**
- 펜윅 트리 구현 예제
  ```c
  #include <stdio.h>

  void updateBIT(int BITree[], int n, int index, int val) {
      index = index + 1;
      while (index <= n) {
          BITree[index] += val;
          index += index & (-index);
      }
  }

  int *constructBITree(int arr[], int n) {
      int *BITree = (int *)malloc((n + 1) * sizeof(int));
      for (int i = 1; i <= n; i++)
          BITree[i] = 0;

      for (int i = 0; i < n; i++)
          updateBIT(BITree, n, i, arr[i]);

      return BITree;
  }

  int getSum(int BITree[], int index) {
      int sum = 0;
      index = index + 1;

      while (index > 0) {
          sum += BITree[index];
          index -= index & (-index);
      }
      return sum;
  }

  void update(int arr[], int BITree[], int n, int index, int val) {
      int diff = val - arr[index];
      arr[index] = val;
      updateBIT(BITree, n, index, diff);
  }

  int main() {
      int freq[] = {3, 2, -1, 6, 5, 4, -3, 3, 7, 2, 3};
      int n = sizeof(freq) / sizeof(freq[0]);
      int *BITree = constructBITree(freq, n);

      printf("Sum of elements in arr[0..5] is %d\n", getSum(BITree, 5));

      update(freq, BITree, n, 3, 10);

      printf("Sum of elements in arr[0..5] after update is %d\n", getSum(BITree, 5));

      free(BITree);
      return 0;
  }
  ```

### 트라이 (Trie)

**강의 내용:**
- 트라이의 개념
  - 문자열 집합을 효율적으로 저장하고 검색하는 자료구조
- 트라이의 구축과 검색
- 트라이의 시간 복잡도
  - 삽입 및 검색: O(m) (문자열의 길이 m)

**실습:**
- 트라이 구현 예제
  ```c
  #include <stdio.h>
  #include <stdlib.h>
  #include <string.h>

  #define ALPHABET_SIZE 26

  struct TrieNode {
      struct TrieNode *children[ALPHABET_SIZE];
      bool isEndOfWord;
  };

  struct TrieNode *getNode(void) {
      struct TrieNode *pNode = (struct TrieNode *)malloc(sizeof(struct TrieNode));
      pNode->isEndOfWord = false;
      for (int i = 0; i < ALPHABET_SIZE; i++)
          pNode->children[i] = NULL;
      return pNode;
  }

  void insert(struct TrieNode *root, const char *key) {
      struct TrieNode *pCrawl = root;
      for (int level = 0; level < strlen(key); level++) {
          int index = key[level] - 'a';
          if (!pCrawl->children[index])
              pCrawl->children[index] = getNode();
          pCrawl = pCrawl->children[index];
      }
      pCrawl->isEndOfWord = true;
  }

  bool search(struct TrieNode *root, const char *key) {
      struct TrieNode *pCrawl = root;
      for (int level = 0; level < strlen(key); level++) {
          int index = key[level] - 'a';
          if (!pCrawl->children[index])
              return false;
          pCrawl = pCrawl->children[index];
      }
      return (pCrawl != NULL && pCrawl->isEndOfWord);
 

 }

  int main() {
      char keys[][8] = {"the", "a", "there", "answer", "any", "by", "bye", "their"};
      int n = sizeof(keys) / sizeof(keys[0]);

      struct TrieNode *root = getNode();

      for (int i = 0; i < n; i++)
          insert(root, keys[i]);

      search(root, "the") ? printf("Yes\n") : printf("No\n");
      search(root, "these") ? printf("Yes\n") : printf("No\n");
      return 0;
  }
  ```

**과제:**
- 다양한 고급 자료구조를 구현하고, 이를 활용한 알고리즘을 작성
- 각 자료구조의 시간 복잡도를 분석하고, 실제 실행 시간을 측정하여 비교

**퀴즈 및 해설:**

1. **세그먼트 트리의 시간 복잡도는 무엇인가요?**
   - 세그먼트 트리의 구축 시간은 O(n)이며, 업데이트 및 질의 시간은 O(log n)입니다.

2. **펜윅 트리의 주요 용도는 무엇인가요?**
   - 펜윅 트리는 배열의 구간 합을 효율적으로 계산하고 업데이트하는 데 사용됩니다.

3. **트라이 자료구조의 삽입 및 검색 시간 복잡도는 무엇인가요?**
   - 트라이 자료구조의 삽입 및 검색 시간 복잡도는 문자열의 길이에 비례하여 O(m)입니다.

**해설:**
1. **세그먼트 트리의 시간 복잡도**는 구축 시간 O(n), 업데이트 및 질의 시간 O(log n)입니다. 이는 세그먼트 트리가 완전 이진 트리 형태를 가지기 때문입니다.
2. **펜윅 트리**는 배열의 구간 합을 효율적으로 계산하고 업데이트하는 자료구조로, BIT(Binary Indexed Tree)라고도 불립니다. 이를 통해 O(log n) 시간 내에 구간 합을 계산할 수 있습니다.
3. **트라이 자료구조**는 문자열 집합을 효율적으로 저장하고 검색하는 데 사용됩니다. 각 노드는 문자에 해당하며, 문자열의 길이에 비례하여 O(m) 시간 내에 삽입 및 검색이 가능합니다.

