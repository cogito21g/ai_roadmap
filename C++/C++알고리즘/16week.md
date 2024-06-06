### 16주차 강의 계획안

#### 강의 주제: 고급 자료구조
- 트라이 (Trie)
- 세그먼트 트리
- 펜윅 트리 (Fenwick Tree)

---

### 강의 내용

#### 1. 트라이 (Trie)
- **개념**: 문자열 집합을 효율적으로 저장하고 검색하기 위한 트리 자료구조
- **응용**: 사전, 자동 완성, 스펠 체크 등

**예제**: 트라이 자료구조
```cpp
#include <iostream>
using namespace std;

const int ALPHABET_SIZE = 26;

struct TrieNode {
    TrieNode* children[ALPHABET_SIZE];
    bool isEndOfWord;
};

TrieNode* getNode() {
    TrieNode* pNode = new TrieNode;
    pNode->isEndOfWord = false;
    for (int i = 0; i < ALPHABET_SIZE; i++)
        pNode->children[i] = nullptr;
    return pNode;
}

void insert(TrieNode* root, string key) {
    TrieNode* pCrawl = root;
    for (int i = 0; i < key.length(); i++) {
        int index = key[i] - 'a';
        if (!pCrawl->children[index])
            pCrawl->children[index] = getNode();
        pCrawl = pCrawl->children[index];
    }
    pCrawl->isEndOfWord = true;
}

bool search(TrieNode* root, string key) {
    TrieNode* pCrawl = root;
    for (int i = 0; i < key.length(); i++) {
        int index = key[i] - 'a';
        if (!pCrawl->children[index])
            return false;
        pCrawl = pCrawl->children[index];
    }
    return (pCrawl != nullptr && pCrawl->isEndOfWord);
}

int main() {
    string keys[] = {"the", "a", "there", "answer", "any", "by", "bye", "their"};
    int n = sizeof(keys) / sizeof(keys[0]);

    TrieNode* root = getNode();

    for (int i = 0; i < n; i++)
        insert(root, keys[i]);

    search(root, "the") ? cout << "Yes\n" : cout << "No\n";
    search(root, "these") ? cout << "Yes\n" : cout << "No\n";
    return 0;
}
```

#### 2. 세그먼트 트리 (Segment Tree)
- **개념**: 배열의 구간 합, 최솟값, 최댓값 등을 효율적으로 계산하기 위한 트리 자료구조
- **응용**: 구간 질의 문제

**예제**: 세그먼트 트리
```cpp
#include <iostream>
#include <vector>
using namespace std;

void buildSegmentTree(vector<int>& arr, vector<int>& segTree, int left, int right, int pos) {
    if (left == right) {
        segTree[pos] = arr[left];
        return;
    }
    int mid = (left + right) / 2;
    buildSegmentTree(arr, segTree, left, mid, 2 * pos + 1);
    buildSegmentTree(arr, segTree, mid + 1, right, 2 * pos + 2);
    segTree[pos] = segTree[2 * pos + 1] + segTree[2 * pos + 2];
}

int rangeQuery(vector<int>& segTree, int qlow, int qhigh, int low, int high, int pos) {
    if (qlow <= low && qhigh >= high) 
        return segTree[pos];
    if (qlow > high || qhigh < low)
        return 0;
    int mid = (low + high) / 2;
    return rangeQuery(segTree, qlow, qhigh, low, mid, 2 * pos + 1) + rangeQuery(segTree, qlow, qhigh, mid + 1, high, 2 * pos + 2);
}

int main() {
    vector<int> arr = {1, 3, 5, 7, 9, 11};
    int n = arr.size();
    vector<int> segTree(2 * n - 1, 0);

    buildSegmentTree(arr, segTree, 0, n - 1, 0);

    cout << "Sum of values in given range = " << rangeQuery(segTree, 1, 3, 0, n - 1, 0) << endl;

    return 0;
}
```

#### 3. 펜윅 트리 (Fenwick Tree)
- **개념**: 배열의 구간 합을 효율적으로 계산하고 업데이트하기 위한 자료구조
- **응용**: 구간 질의 문제

**예제**: 펜윅 트리
```cpp
#include <iostream>
#include <vector>
using namespace std;

class FenwickTree {
    vector<int> BIT;
    int n;

public:
    FenwickTree(int size) {
        n = size;
        BIT = vector<int>(n + 1, 0);
    }

    void update(int index, int val) {
        index++;
        while (index <= n) {
            BIT[index] += val;
            index += index & (-index);
        }
    }

    int query(int index) {
        int sum = 0;
        index++;
        while (index > 0) {
            sum += BIT[index];
            index -= index & (-index);
        }
        return sum;
    }

    int rangeQuery(int left, int right) {
        return query(right) - query(left - 1);
    }
};

int main() {
    vector<int> arr = {1, 3, 5, 7, 9, 11};
    int n = arr.size();
    FenwickTree fenwickTree(n);

    for (int i = 0; i < n; i++)
        fenwickTree.update(i, arr[i]);

    cout << "Sum of values in given range = " << fenwickTree.rangeQuery(1, 3) << endl;

    return 0;
}
```

---

### 과제

#### 과제 1: 트라이 자료구조를 이용한 문자열 삽입 및 검색
트라이 자료구조를 구현하고, 사용자로부터 단어 리스트와 검색할 단어를 입력받아 단어를 삽입하고 검색하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
using namespace std;

const int ALPHABET_SIZE = 26;

struct TrieNode {
    TrieNode* children[ALPHABET_SIZE];
    bool isEndOfWord;
};

TrieNode* getNode() {
    TrieNode* pNode = new TrieNode;
    pNode->isEndOfWord = false;
    for (int i = 0; i < ALPHABET_SIZE; i++)
        pNode->children[i] = nullptr;
    return pNode;
}

void insert(TrieNode* root, string key) {
    TrieNode* pCrawl = root;
    for (int i = 0; i < key.length(); i++) {
        int index = key[i] - 'a';
        if (!pCrawl->children[index])
            pCrawl->children[index] = getNode();
        pCrawl = pCrawl->children[index];
    }
    pCrawl->isEndOfWord = true;
}

bool search(TrieNode* root, string key) {
    TrieNode* pCrawl = root;
    for (int i = 0; i < key.length(); i++) {
        int index = key[i] - 'a';
        if (!pCrawl->children[index])
            return false;
        pCrawl = pCrawl->children[index];
    }
    return (pCrawl != nullptr && pCrawl->isEndOfWord);
}

int main() {
    int n;
    cout << "Enter the number of words: ";
    cin >> n;
    vector<string> keys(n);
    cout << "Enter the words:\n";
    for (int i = 0; i < n; i++) {
        cin >> keys[i];
    }

    TrieNode* root = getNode();

    for (int i = 0; i < n; i++)
        insert(root, keys[i]);

    string searchWord;
    cout << "Enter the word to search: ";
    cin >> searchWord;
    search(root, searchWord) ? cout << "Yes\n" : cout << "No\n";

    return 0;
}
```

**해설**:
1. 사용자로부터 단어 리스트와 검색할 단어를 입력받습니다.
2. `insert` 함수를 사용해 단어 리스트를 트라이에 삽입합니다.
3. `search` 함수를 사용해 단어를 검색하고, 결과를 출력합니다.

#### 과제 2: 세그먼트 트리를 이용한 구간 합 계산
사용자로부터 배열을 입력받아 세그먼트 트리를 구축하고, 주어진 구간의 합을 계산하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
using namespace std;

void buildSegmentTree(vector<int>& arr, vector<int>& segTree, int left, int right, int pos) {
    if (left

 == right) {
        segTree[pos] = arr[left];
        return;
    }
    int mid = (left + right) / 2;
    buildSegmentTree(arr, segTree, left, mid, 2 * pos + 1);
    buildSegmentTree(arr, segTree, mid + 1, right, 2 * pos + 2);
    segTree[pos] = segTree[2 * pos + 1] + segTree[2 * pos + 2];
}

int rangeQuery(vector<int>& segTree, int qlow, int qhigh, int low, int high, int pos) {
    if (qlow <= low && qhigh >= high) 
        return segTree[pos];
    if (qlow > high || qhigh < low)
        return 0;
    int mid = (low + high) / 2;
    return rangeQuery(segTree, qlow, qhigh, low, mid, 2 * pos + 1) + rangeQuery(segTree, qlow, qhigh, mid + 1, high, 2 * pos + 2);
}

int main() {
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;
    vector<int> arr(n);
    cout << "Enter the elements:\n";
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    vector<int> segTree(2 * n - 1, 0);
    buildSegmentTree(arr, segTree, 0, n - 1, 0);

    int qlow, qhigh;
    cout << "Enter the range (qlow and qhigh): ";
    cin >> qlow >> qhigh;
    cout << "Sum of values in given range = " << rangeQuery(segTree, qlow, qhigh, 0, n - 1, 0) << endl;

    return 0;
}
```

**해설**:
1. 사용자로부터 배열 크기와 요소를 입력받습니다.
2. `buildSegmentTree` 함수를 사용해 세그먼트 트리를 구축합니다.
3. `rangeQuery` 함수를 사용해 주어진 구간의 합을 계산하고, 결과를 출력합니다.

#### 과제 3: 펜윅 트리를 이용한 구간 합 계산
사용자로부터 배열을 입력받아 펜윅 트리를 구축하고, 주어진 구간의 합을 계산하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
using namespace std;

class FenwickTree {
    vector<int> BIT;
    int n;

public:
    FenwickTree(int size) {
        n = size;
        BIT = vector<int>(n + 1, 0);
    }

    void update(int index, int val) {
        index++;
        while (index <= n) {
            BIT[index] += val;
            index += index & (-index);
        }
    }

    int query(int index) {
        int sum = 0;
        index++;
        while (index > 0) {
            sum += BIT[index];
            index -= index & (-index);
        }
        return sum;
    }

    int rangeQuery(int left, int right) {
        return query(right) - query(left - 1);
    }
};

int main() {
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;
    vector<int> arr(n);
    cout << "Enter the elements:\n";
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    FenwickTree fenwickTree(n);

    for (int i = 0; i < n; i++)
        fenwickTree.update(i, arr[i]);

    int left, right;
    cout << "Enter the range (left and right): ";
    cin >> left >> right;
    cout << "Sum of values in given range = " << fenwickTree.rangeQuery(left, right) << endl;

    return 0;
}
```

**해설**:
1. 사용자로부터 배열 크기와 요소를 입력받습니다.
2. `update` 함수를 사용해 펜윅 트리를 구축합니다.
3. `rangeQuery` 함수를 사용해 주어진 구간의 합을 계산하고, 결과를 출력합니다.

---

### 퀴즈

#### 퀴즈 1: 트라이 자료구조의 주요 특징은 무엇인가요?
1. 문자열의 접두사를 효율적으로 저장하고 검색할 수 있다.
2. 문자열의 접미사를 효율적으로 저장하고 검색할 수 있다.
3. 정렬된 배열을 사용한다.
4. 문자열을 압축 저장할 수 있다.

**정답**: 1. 문자열의 접두사를 효율적으로 저장하고 검색할 수 있다.

#### 퀴즈 2: 세그먼트 트리의 시간 복잡도는 무엇인가요?
1. O(log n) for update and query
2. O(n) for update and query
3. O(n log n) for update and query
4. O(1) for update and query

**정답**: 1. O(log n) for update and query

#### 퀴즈 3: 펜윅 트리의 주요 응용 분야는 무엇인가요?
1. 문자열 검색
2. 구간 질의 문제
3. 그래프 탐색
4. 정렬 문제

**정답**: 2. 구간 질의 문제

이 계획안은 16주차에 필요한 고급 자료구조의 개념과 실습을 통해 학습할 수 있도록 구성되었습니다. 과제와 퀴즈를 통해 학습한 내용을 복습하고, 이해도를 높일 수 있습니다.