### 알고리즘 교육과정 - 8주차: 탐욕 알고리즘 (Greedy Algorithms)

**강의 목표:**
탐욕 알고리즘의 개념과 원리를 이해하고, 이를 활용한 다양한 알고리즘을 학습합니다. 최적 부분 구조와 탐욕 선택 속성을 이용하여 복잡한 문제를 효율적으로 해결하는 방법을 학습합니다.

**강의 구성:**

#### 8. 탐욕 알고리즘 (Greedy Algorithms)

**강의 내용:**
- 탐욕 알고리즘의 개념과 원리
- 탐욕 알고리즘의 기본 패턴
- 대표적인 탐욕 알고리즘 문제

**실습:**
- 탐욕 알고리즘 구현 및 성능 분석

### 탐욕 알고리즘의 개념과 원리

**강의 내용:**
- 탐욕 알고리즘이란 무엇인가?
  - 최적 부분 구조와 탐욕 선택 속성
  - 국소 최적해가 전체 최적해로 이어짐
- 탐욕 알고리즘의 적용 사례
  - 활동 선택 문제
  - 최소 신장 트리
  - Huffman 코딩

**실습:**
- 탐욕 알고리즘 이해를 위한 간단한 예제

### 활동 선택 문제

**강의 내용:**
- 활동 선택 문제의 개념
  - 시작 시간과 종료 시간이 주어진 활동들 중 최대한 많은 활동을 선택하는 문제
- 활동 선택 문제의 시간 복잡도
  - O(n log n) (정렬 포함)

**실습:**
- 활동 선택 문제 구현 예제
  ```c
  #include <stdio.h>

  void printMaxActivities(int s[], int f[], int n) {
      int i, j;
      printf("Selected activities: \n");
      i = 0;
      printf("%d ", i);

      for (j = 1; j < n; j++) {
          if (s[j] >= f[i]) {
              printf("%d ", j);
              i = j;
          }
      }
      printf("\n");
  }

  int main() {
      int s[] = {1, 3, 0, 5, 8, 5};
      int f[] = {2, 4, 6, 7, 9, 9};
      int n = sizeof(s) / sizeof(s[0]);
      printMaxActivities(s, f, n);
      return 0;
  }
  ```

### 최소 신장 트리 (MST)

**강의 내용:**
- 최소 신장 트리의 개념
  - 그래프의 모든 정점을 포함하면서 간선의 가중치 합이 최소가 되는 트리
- 크루스칼 알고리즘
  - 간선을 정렬하고 사이클을 이루지 않도록 선택
- 프림 알고리즘
  - 하나의 정점에서 시작하여 최소 비용 간선을 선택

**실습:**
- 크루스칼 알고리즘 구현 예제
  ```c
  #include <stdio.h>
  #include <stdlib.h>

  struct Edge {
      int src, dest, weight;
  };

  struct Graph {
      int V, E;
      struct Edge* edge;
  };

  struct Graph* createGraph(int V, int E) {
      struct Graph* graph = (struct Graph*) malloc(sizeof(struct Graph));
      graph->V = V;
      graph->E = E;
      graph->edge = (struct Edge*) malloc(graph->E * sizeof(struct Edge));
      return graph;
  }

  struct subset {
      int parent;
      int rank;
  };

  int find(struct subset subsets[], int i) {
      if (subsets[i].parent != i)
          subsets[i].parent = find(subsets, subsets[i].parent);
      return subsets[i].parent;
  }

  void Union(struct subset subsets[], int x, int y) {
      int rootX = find(subsets, x);
      int rootY = find(subsets, y);

      if (subsets[rootX].rank < subsets[rootY].rank)
          subsets[rootX].parent = rootY;
      else if (subsets[rootX].rank > subsets[rootY].rank)
          subsets[rootY].parent = rootX;
      else {
          subsets[rootY].parent = rootX;
          subsets[rootX].rank++;
      }
  }

  int compare(const void* a, const void* b) {
      struct Edge* a1 = (struct Edge*) a;
      struct Edge* b1 = (struct Edge*) b;
      return a1->weight > b1->weight;
  }

  void KruskalMST(struct Graph* graph) {
      int V = graph->V;
      struct Edge result[V];
      int e = 0;
      int i = 0;

      qsort(graph->edge, graph->E, sizeof(graph->edge[0]), compare);

      struct subset* subsets = (struct subset*) malloc(V * sizeof(struct subset));
      for (int v = 0; v < V; ++v) {
          subsets[v].parent = v;
          subsets[v].rank = 0;
      }

      while (e < V - 1 && i < graph->E) {
          struct Edge next_edge = graph->edge[i++];
          int x = find(subsets, next_edge.src);
          int y = find(subsets, next_edge.dest);

          if (x != y) {
              result[e++] = next_edge;
              Union(subsets, x, y);
          }
      }

      printf("Following are the edges in the constructed MST\n");
      for (i = 0; i < e; ++i)
          printf("%d -- %d == %d\n", result[i].src, result[i].dest, result[i].weight);
      return;
  }

  int main() {
      int V = 4;
      int E = 5;
      struct Graph* graph = createGraph(V, E);

      graph->edge[0].src = 0;
      graph->edge[0].dest = 1;
      graph->edge[0].weight = 10;

      graph->edge[1].src = 0;
      graph->edge[1].dest = 2;
      graph->edge[1].weight = 6;

      graph->edge[2].src = 0;
      graph->edge[2].dest = 3;
      graph->edge[2].weight = 5;

      graph->edge[3].src = 1;
      graph->edge[3].dest = 3;
      graph->edge[3].weight = 15;

      graph->edge[4].src = 2;
      graph->edge[4].dest = 3;
      graph->edge[4].weight = 4;

      KruskalMST(graph);

      return 0;
  }
  ```

### Huffman 코딩

**강의 내용:**
- Huffman 코딩의 개념
  - 빈도가 높은 문자는 짧은 코드를, 빈도가 낮은 문자는 긴 코드를 부여하여 압축하는 방법
- Huffman 트리 구성
  - 빈도를 기반으로 트리를 구성하고, 각 문자의 코드를 생성

**실습:**
- Huffman 코딩 구현 예제
  ```c
  #include <stdio.h>
  #include <stdlib.h>

  struct MinHeapNode {
      char data;
      unsigned freq;
      struct MinHeapNode *left, *right;
  };

  struct MinHeap {
      unsigned size;
      unsigned capacity;
      struct MinHeapNode** array;
  };

  struct MinHeapNode* newNode(char data, unsigned freq) {
      struct MinHeapNode* temp = (struct MinHeapNode*) malloc(sizeof(struct MinHeapNode));
      temp->left = temp->right = NULL;
      temp->data = data;
      temp->freq = freq;
      return temp;
  }

  struct MinHeap* createMinHeap(unsigned capacity) {
      struct MinHeap* minHeap = (struct MinHeap*) malloc(sizeof(struct MinHeap));
      minHeap->size = 0;
      minHeap->capacity = capacity;
      minHeap->array = (struct MinHeapNode**) malloc(minHeap->capacity * sizeof(struct MinHeapNode*));
      return minHeap;
  }

  void swapMinHeapNode(struct MinHeapNode** a, struct MinHeapNode** b) {
      struct MinHeapNode* t = *a;
      *a = *b;
      *b = t;
  }

  void minHeapify(struct MinHeap* minHeap, int idx) {
      int smallest = idx;
      int left = 2 * idx + 1;
      int right = 2 * idx + 2;

      if (left < minHeap->size && minHeap->array[left]->freq < minHeap->array[smallest]->freq)
          smallest = left;

      if (right < minHeap->size && minHeap->array[right]->freq < minHeap->array[smallest]->freq)
          smallest = right;

      if (smallest != idx) {
          swapMinHeapNode(&minHeap->array[smallest], &minHeap->array[idx]);
          minHeapify(minHeap, smallest);
      }
  }

  int isSizeOne(struct MinHeap* minHeap) {
      return (minHeap->size == 1);
  }

  struct MinHeapNode* extractMin(struct MinHeap* minHeap) {
      struct MinHeapNode* temp = minHeap->array[0];
      minHeap->array[0] =

 minHeap->array[minHeap->size - 1];
      --minHeap->size;
      minHeapify(minHeap, 0);
      return temp;
  }

  void insertMinHeap(struct MinHeap* minHeap, struct MinHeapNode* minHeapNode) {
      ++minHeap->size;
      int i = minHeap->size - 1;
      while (i && minHeapNode->freq < minHeap->array[(i - 1) / 2]->freq) {
          minHeap->array[i] = minHeap->array[(i - 1) / 2];
          i = (i - 1) / 2;
      }
      minHeap->array[i] = minHeapNode;
  }

  void buildMinHeap(struct MinHeap* minHeap) {
      int n = minHeap->size - 1;
      for (int i = (n - 1) / 2; i >= 0; --i)
          minHeapify(minHeap, i);
  }

  void printArr(int arr[], int n) {
      for (int i = 0; i < n; ++i)
          printf("%d", arr[i]);
      printf("\n");
  }

  int isLeaf(struct MinHeapNode* root) {
      return !(root->left) && !(root->right);
  }

  struct MinHeap* createAndBuildMinHeap(char data[], int freq[], int size) {
      struct MinHeap* minHeap = createMinHeap(size);
      for (int i = 0; i < size; ++i)
          minHeap->array[i] = newNode(data[i], freq[i]);
      minHeap->size = size;
      buildMinHeap(minHeap);
      return minHeap;
  }

  struct MinHeapNode* buildHuffmanTree(char data[], int freq[], int size) {
      struct MinHeapNode *left, *right, *top;
      struct MinHeap* minHeap = createAndBuildMinHeap(data, freq, size);

      while (!isSizeOne(minHeap)) {
          left = extractMin(minHeap);
          right = extractMin(minHeap);

          top = newNode('$', left->freq + right->freq);
          top->left = left;
          top->right = right;

          insertMinHeap(minHeap, top);
      }

      return extractMin(minHeap);
  }

  void printCodes(struct MinHeapNode* root, int arr[], int top) {
      if (root->left) {
          arr[top] = 0;
          printCodes(root->left, arr, top + 1);
      }

      if (root->right) {
          arr[top] = 1;
          printCodes(root->right, arr, top + 1);
      }

      if (isLeaf(root)) {
          printf("%c: ", root->data);
          printArr(arr, top);
      }
  }

  void HuffmanCodes(char data[], int freq[], int size) {
      struct MinHeapNode* root = buildHuffmanTree(data, freq, size);
      int arr[100], top = 0;
      printCodes(root, arr, top);
  }

  int main() {
      char arr[] = {'a', 'b', 'c', 'd', 'e', 'f'};
      int freq[] = {5, 9, 12, 13, 16, 45};
      int size = sizeof(arr) / sizeof(arr[0]);
      HuffmanCodes(arr, freq, size);
      return 0;
  }
  ```

**과제:**
- 다양한 탐욕 알고리즘 문제를 해결하는 알고리즘을 구현하고, 성능을 비교
- 각 알고리즘의 시간 복잡도를 분석하고, 실제 실행 시간을 측정하여 비교

**퀴즈 및 해설:**

1. **탐욕 알고리즘이란 무엇인가요?**
   - 탐욕 알고리즘은 매 단계에서 가장 최적인 선택을 함으로써 문제를 해결하는 알고리즘 기법입니다.

2. **크루스칼 알고리즘의 시간 복잡도는 무엇인가요?**
   - 크루스칼 알고리즘의 시간 복잡도는 O(E log E)입니다. 여기서 E는 그래프의 간선 수입니다.

3. **Huffman 코딩의 기본 원리는 무엇인가요?**
   - Huffman 코딩은 빈도가 높은 문자는 짧은 코드를, 빈도가 낮은 문자는 긴 코드를 부여하여 압축하는 방법입니다.

**해설:**
1. **탐욕 알고리즘**은 매 단계에서 가장 최적인 선택을 함으로써 문제를 해결하는 알고리즘 기법입니다. 이는 국소 최적해가 전체 최적해로 이어지는 경우에 유용합니다.
2. **크루스칼 알고리즘의 시간 복잡도**는 간선을 정렬하는 과정이 O(E log E)이며, 그 후의 합집합 연산이 거의 O(1)에 가깝기 때문에 전체 시간 복잡도는 O(E log E)입니다.
3. **Huffman 코딩의 기본 원리**는 빈도가 높은 문자는 짧은 코드를, 빈도가 낮은 문자는 긴 코드를 부여하여 데이터를 압축하는 방법입니다. 이는 Huffman 트리를 이용하여 구현됩니다.

---
