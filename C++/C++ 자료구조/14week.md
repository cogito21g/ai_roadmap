### 14주차 강의 계획: 트라이와 접미사 트리

#### 강의 목표
- 트라이(Trie)와 접미사 트리(Suffix Tree)의 기본 개념 이해
- 트라이의 구현 및 활용 사례 학습
- 접미사 트리의 구현 및 문자열 검색 응용 이해

#### 강의 구성
- **강의 시간**: 2시간
- **구성**: 트라이 이론 (30분), 트라이 구현 (30분), 접미사 트리 이론 (30분), 접미사 트리 구현 및 응용 (30분)

#### 강의 내용

##### 1. 트라이(Trie) 이론 (30분)

###### 1.1 트라이의 기본 개념
- **트라이 개요**:
  - 문자열을 저장하고 효율적으로 검색하기 위한 트리 자료 구조
  - 각 노드는 문자열의 문자 하나를 나타냄

###### 1.2 트라이의 특징
- **특징**:
  - 문자열 검색이 O(m) 시간 복잡도로 가능 (m은 문자열의 길이)
  - 공통 접두사를 공유하는 구조

###### 1.3 트라이의 활용 사례
- **활용 사례**:
  - 사전 구현
  - 자동 완성
  - 문자열 검색

##### 2. 트라이 구현 (30분)

###### 2.1 트라이의 구조
- **노드 구조**:
  - 자식 노드를 가리키는 포인터 배열
  - 문자열의 끝을 나타내는 플래그

###### 2.2 트라이의 주요 연산 구현
- **삽입, 검색, 삭제 연산 구현**:
- **예제**:
```cpp
#include <iostream>
using namespace std;

const int ALPHABET_SIZE = 26;

class TrieNode {
public:
    TrieNode* children[ALPHABET_SIZE];
    bool isEndOfWord;

    TrieNode() {
        isEndOfWord = false;
        for (int i = 0; i < ALPHABET_SIZE; i++)
            children[i] = nullptr;
    }
};

class Trie {
private:
    TrieNode* root;

public:
    Trie() {
        root = new TrieNode();
    }

    void insert(string key) {
        TrieNode* pCrawl = root;
        for (int i = 0; i < key.length(); i++) {
            int index = key[i] - 'a';
            if (!pCrawl->children[index])
                pCrawl->children[index] = new TrieNode();
            pCrawl = pCrawl->children[index];
        }
        pCrawl->isEndOfWord = true;
    }

    bool search(string key) {
        TrieNode* pCrawl = root;
        for (int i = 0; i < key.length(); i++) {
            int index = key[i] - 'a';
            if (!pCrawl->children[index])
                return false;
            pCrawl = pCrawl->children[index];
        }
        return (pCrawl != nullptr && pCrawl->isEndOfWord);
    }

    bool isEmpty(TrieNode* root) {
        for (int i = 0; i < ALPHABET_SIZE; i++)
            if (root->children[i])
                return false;
        return true;
    }

    TrieNode* remove(TrieNode* root, string key, int depth = 0) {
        if (!root)
            return nullptr;

        if (depth == key.size()) {
            if (root->isEndOfWord)
                root->isEndOfWord = false;

            if (isEmpty(root)) {
                delete root;
                root = nullptr;
            }

            return root;
        }

        int index = key[depth] - 'a';
        root->children[index] = remove(root->children[index], key, depth + 1);

        if (isEmpty(root) && !root->isEndOfWord) {
            delete root;
            root = nullptr;
        }

        return root;
    }

    void remove(string key) {
        root = remove(root, key);
    }
};

int main() {
    string keys[] = {"the", "a", "there", "answer", "any", "by", "bye", "their"};
    int n = sizeof(keys) / sizeof(keys[0]);

    Trie trie;
    for (int i = 0; i < n; i++)
        trie.insert(keys[i]);

    trie.search("the") ? cout << "Yes\n" : cout << "No\n";
    trie.search("these") ? cout << "Yes\n" : cout << "No\n";
    trie.search("their") ? cout << "Yes\n" : cout << "No\n";
    trie.search("thaw") ? cout << "Yes\n" : cout << "No\n";

    trie.remove("her");
    trie.search("her") ? cout << "Yes\n" : cout << "No\n";

    return 0;
}
```

##### 3. 접미사 트리(Suffix Tree) 이론 (30분)

###### 3.1 접미사 트리의 기본 개념
- **접미사 트리 개요**:
  - 주어진 문자열의 모든 접미사를 포함하는 트리 구조
  - 문자열 검색을 효율적으로 수행

###### 3.2 접미사 트리의 특징
- **특징**:
  - 문자열의 모든 접미사를 저장
  - 특정 패턴 검색이 O(m) 시간 복잡도로 가능 (m은 패턴의 길이)

###### 3.3 접미사 트리의 활용 사례
- **활용 사례**:
  - 문자열 검색
  - 문자열 매칭
  - 유전자 서열 분석

##### 4. 접미사 트리 구현 및 응용 (30분)

###### 4.1 접미사 트리의 구조
- **노드 구조**:
  - 자식 노드를 가리키는 포인터 배열 또는 맵
  - 시작 및 끝 인덱스를 저장하는 배열

###### 4.2 접미사 트리의 주요 연산 구현
- **구현 방법**:
  - Ukkonen's 알고리즘을 사용하여 접미사 트리 구축
- **예제**:
```cpp
#include <iostream>
#include <map>
using namespace std;

class SuffixTreeNode {
public:
    map<char, SuffixTreeNode*> children;
    SuffixTreeNode* suffixLink;
    int start;
    int* end;
    int suffixIndex;

    SuffixTreeNode(int start, int* end) {
        this->start = start;
        this->end = end;
        suffixLink = nullptr;
        suffixIndex = -1;
    }
};

class SuffixTree {
private:
    SuffixTreeNode* root;
    string text;
    SuffixTreeNode* lastNewNode;
    SuffixTreeNode* activeNode;
    int activeEdge;
    int activeLength;
    int remainingSuffixCount;
    int leafEnd;
    int* rootEnd;
    int* splitEnd;
    int size;

    SuffixTreeNode* newNode(int start, int* end) {
        SuffixTreeNode* node = new SuffixTreeNode(start, end);
        return node;
    }

    int edgeLength(SuffixTreeNode* n) {
        return *(n->end) - (n->start) + 1;
    }

    bool walkDown(SuffixTreeNode* currNode) {
        if (activeLength >= edgeLength(currNode)) {
            activeEdge += edgeLength(currNode);
            activeLength -= edgeLength(currNode);
            activeNode = currNode;
            return true;
        }
        return false;
    }

    void extendSuffixTree(int pos) {
        leafEnd = pos;
        remainingSuffixCount++;
        lastNewNode = nullptr;

        while (remainingSuffixCount > 0) {
            if (activeLength == 0)
                activeEdge = pos;

            if (activeNode->children.find(text[activeEdge]) == activeNode->children.end()) {
                activeNode->children[text[activeEdge]] = newNode(pos, &leafEnd);

                if (lastNewNode != nullptr) {
                    lastNewNode->suffixLink = activeNode;
                    lastNewNode = nullptr;
                }
            } else {
                SuffixTreeNode* next = activeNode->children[text[activeEdge]];
                if (walkDown(next))
                    continue;

                if (text[next->start + activeLength] == text[pos]) {
                    if (lastNewNode != nullptr && activeNode != root) {
                        lastNewNode->suffixLink = activeNode;
                        lastNewNode = nullptr;
                    }
                    activeLength++;
                    break;
                }

                splitEnd = new int;
                *splitEnd = next->start + activeLength - 1;

                SuffixTreeNode* split = newNode(next->start, splitEnd);
                activeNode->children[text[activeEdge]] = split;

                split->children[text[pos]] = newNode(pos, &leafEnd);
                next->start += activeLength;
                split->children[text[next->start]] = next;

                if (lastNewNode != nullptr) {
                    lastNewNode->suffixLink = split;
                }

                lastNewNode = split;
            }

            remainingSuffixCount--;

            if (activeNode == root && activeLength > 0) {
                activeLength--;
                activeEdge = pos - remainingSuffixCount + 1;
            } else if (activeNode != root) {
                activeNode = active

Node->suffixLink;
            }
        }
    }

    void setSuffixIndexByDFS(SuffixTreeNode* n, int labelHeight) {
        if (n == nullptr) return;

        bool leaf = true;
        for (auto& it : n->children) {
            leaf = false;
            setSuffixIndexByDFS(it.second, labelHeight + edgeLength(it.second));
        }
        if (leaf) {
            n->suffixIndex = size - labelHeight;
        }
    }

public:
    SuffixTree(string text) {
        this->text = text;
        size = text.size();
        rootEnd = new int(-1);
        root = newNode(-1, rootEnd);
        activeNode = root;
        activeEdge = -1;
        activeLength = 0;
        remainingSuffixCount = 0;
        leafEnd = -1;

        for (int i = 0; i < size; i++) {
            extendSuffixTree(i);
        }
        setSuffixIndexByDFS(root, 0);
    }

    ~SuffixTree() {
        // Clean up the dynamically allocated memory
    }

    void print(int i, int j) {
        for (int k = i; k <= j; k++) {
            cout << text[k];
        }
    }

    void printTree(SuffixTreeNode* n, int labelHeight) {
        if (n == nullptr) return;

        if (n->start != -1) {
            print(n->start, *(n->end));
        }

        for (auto& it : n->children) {
            if (n->start != -1) cout << " [" << n->suffixIndex << "]";
            cout << endl;
            printTree(it.second, labelHeight + edgeLength(it.second));
        }
    }

    void printTree() {
        printTree(root, 0);
    }
};

int main() {
    string text = "banana";
    SuffixTree tree(text);

    cout << "Suffix Tree for " << text << ":\n";
    tree.printTree();

    return 0;
}
```

##### 4. 실습 및 과제 안내 (30분)

###### 4.1 실습
- **실습 목표**:
  - 트라이와 접미사 트리를 사용한 프로그램 작성
- **실습 문제**:
  - 트라이를 구현하고, 문자열 삽입, 검색, 삭제 연산을 수행하는 프로그램 작성
  - 접미사 트리를 구현하고, 문자열 검색을 효율적으로 수행하는 프로그램 작성

###### 4.2 과제 안내
- **과제 내용**:
  - 트라이를 사용하여 자동 완성 기능을 구현하는 프로그램 작성
  - 접미사 트리를 사용하여 주어진 문자열에서 특정 패턴을 검색하는 프로그램 작성
- **과제 제출 방법**:
  - 작성한 코드를 주차별 과제 제출 폴더에 업로드
  - 제출 기한: 다음 주 강의 전까지

##### 5. 퀴즈 및 해설 (30분)

###### 퀴즈 문제:
1. 트라이에서 문자열 검색의 시간 복잡도는?
   - a) O(n)
   - b) O(m)
   - c) O(log n)
   - d) O(n log n)
2. 접미사 트리의 주요 응용 분야는?
   - a) 그래프 탐색
   - b) 문자열 검색 및 매칭
   - c) 정렬
   - d) 데이터베이스 인덱싱
3. 접미사 트리를 구축하는데 사용되는 알고리즘은?
   - a) KMP 알고리즘
   - b) 타잔 알고리즘
   - c) Ukkonen 알고리즘
   - d) 크루스칼 알고리즘

###### 퀴즈 해설:
1. **정답: b) O(m)**
   - 트라이에서 문자열 검색의 시간 복잡도는 문자열의 길이에 비례하여 O(m)입니다.
2. **정답: b) 문자열 검색 및 매칭**
   - 접미사 트리는 주로 문자열 검색 및 매칭에 사용됩니다.
3. **정답: c) Ukkonen 알고리즘**
   - 접미사 트리를 구축하는데 사용되는 알고리즘은 Ukkonen 알고리즘입니다.

### 준비 자료
- 강의 슬라이드
- 실습 코드 예제
- 과제 안내서

### 과제 해설

##### 과제: 트라이를 사용하여 자동 완성 기능을 구현하는 프로그램 작성
- **문제**: 트라이를 사용하여 자동 완성 기능을 구현하는 프로그램 작성
- **해설**:
  - 트라이에 문자열을 삽입하고, 주어진 접두사로 시작하는 문자열을 자동 완성합니다.

```cpp
#include <iostream>
#include <vector>
using namespace std;

const int ALPHABET_SIZE = 26;

class TrieNode {
public:
    TrieNode* children[ALPHABET_SIZE];
    bool isEndOfWord;

    TrieNode() {
        isEndOfWord = false;
        for (int i = 0; i < ALPHABET_SIZE; i++)
            children[i] = nullptr;
    }
};

class Trie {
private:
    TrieNode* root;

    void suggestionsRec(TrieNode* root, string currentPrefix) {
        if (root->isEndOfWord) {
            cout << currentPrefix << endl;
        }
        for (int i = 0; i < ALPHABET_SIZE; i++) {
            if (root->children[i]) {
                suggestionsRec(root->children[i], currentPrefix + char(i + 'a'));
            }
        }
    }

public:
    Trie() {
        root = new TrieNode();
    }

    void insert(string key) {
        TrieNode* pCrawl = root;
        for (int i = 0; i < key.length(); i++) {
            int index = key[i] - 'a';
            if (!pCrawl->children[index])
                pCrawl->children[index] = new TrieNode();
            pCrawl = pCrawl->children[index];
        }
        pCrawl->isEndOfWord = true;
    }

    bool search(string key) {
        TrieNode* pCrawl = root;
        for (int i = 0; i < key.length(); i++) {
            int index = key[i] - 'a';
            if (!pCrawl->children[index])
                return false;
            pCrawl = pCrawl->children[index];
        }
        return (pCrawl != nullptr && pCrawl->isEndOfWord);
    }

    void printAutoSuggestions(string query) {
        TrieNode* pCrawl = root;
        for (int i = 0; i < query.length(); i++) {
            int index = query[i] - 'a';
            if (!pCrawl->children[index]) {
                cout << "No suggestions found for the given query." << endl;
                return;
            }
            pCrawl = pCrawl->children[index];
        }
        suggestionsRec(pCrawl, query);
    }
};

int main() {
    string keys[] = {"hello", "hell", "heaven", "heavy"};
    int n = sizeof(keys) / sizeof(keys[0]);

    Trie trie;
    for (int i = 0; i < n; i++)
        trie.insert(keys[i]);

    string query = "he";
    cout << "Auto-suggestions for \"" << query << "\":" << endl;
    trie.printAutoSuggestions(query);

    return 0;
}
```
- **설명**:
  - `Trie` 클래스를 사용하여 문자열을 삽입하고, 자동 완성 기능을 구현합니다.
  - 주어진 접두사로 시작하는 문자열을 출력합니다.

##### 과제: 접미사 트리를 사용하여 주어진 문자열에서 특정 패턴을 검색하는 프로그램 작성
- **문제**: 접미사 트리를 사용하여 주어진 문자열에서 특정 패턴을 검색하는 프로그램 작성
- **해설**:
  - 접미사 트리를 구축하고, 특정 패턴이 문자열에 포함되어 있는지 확인합니다.

```cpp
#include <iostream>
#include <map>
using namespace std;

class SuffixTreeNode {
public:
    map<char, SuffixTreeNode*> children;
    SuffixTreeNode* suffixLink;
    int start;
    int* end;
    int suffixIndex;

    SuffixTreeNode(int start, int* end) {
        this->start = start;
        this->end = end;
        suffixLink = nullptr;
        suffixIndex = -1;
    }
};

class SuffixTree {
private:
    SuffixTreeNode* root;
    string text;
    SuffixTreeNode* lastNewNode;
    SuffixTreeNode* activeNode;
    int activeEdge;
    int activeLength;
    int remainingSuffixCount;
    int leafEnd;
    int* rootEnd;
    int* splitEnd;
    int size;

    SuffixTreeNode* newNode(int start, int* end) {
        SuffixTreeNode* node = new SuffixTreeNode(start, end);
        return node;
    }

    int edgeLength(SuffixTreeNode* n) {
        return *(n->end) - (n->start) + 1;
    }

    bool walkDown(SuffixTreeNode* currNode) {
        if (activeLength >= edgeLength(currNode)) {
            activeEdge += edgeLength(currNode);
            activeLength -= edgeLength(currNode);
            activeNode = currNode;
            return true;
        }
        return

 false;
    }

    void extendSuffixTree(int pos) {
        leafEnd = pos;
        remainingSuffixCount++;
        lastNewNode = nullptr;

        while (remainingSuffixCount > 0) {
            if (activeLength == 0)
                activeEdge = pos;

            if (activeNode->children.find(text[activeEdge]) == activeNode->children.end()) {
                activeNode->children[text[activeEdge]] = newNode(pos, &leafEnd);

                if (lastNewNode != nullptr) {
                    lastNewNode->suffixLink = activeNode;
                    lastNewNode = nullptr;
                }
            } else {
                SuffixTreeNode* next = activeNode->children[text[activeEdge]];
                if (walkDown(next))
                    continue;

                if (text[next->start + activeLength] == text[pos]) {
                    if (lastNewNode != nullptr && activeNode != root) {
                        lastNewNode->suffixLink = activeNode;
                        lastNewNode = nullptr;
                    }
                    activeLength++;
                    break;
                }

                splitEnd = new int;
                *splitEnd = next->start + activeLength - 1;

                SuffixTreeNode* split = newNode(next->start, splitEnd);
                activeNode->children[text[activeEdge]] = split;

                split->children[text[pos]] = newNode(pos, &leafEnd);
                next->start += activeLength;
                split->children[text[next->start]] = next;

                if (lastNewNode != nullptr) {
                    lastNewNode->suffixLink = split;
                }

                lastNewNode = split;
            }

            remainingSuffixCount--;

            if (activeNode == root && activeLength > 0) {
                activeLength--;
                activeEdge = pos - remainingSuffixCount + 1;
            } else if (activeNode != root) {
                activeNode = activeNode->suffixLink;
            }
        }
    }

    void setSuffixIndexByDFS(SuffixTreeNode* n, int labelHeight) {
        if (n == nullptr) return;

        bool leaf = true;
        for (auto& it : n->children) {
            leaf = false;
            setSuffixIndexByDFS(it.second, labelHeight + edgeLength(it.second));
        }
        if (leaf) {
            n->suffixIndex = size - labelHeight;
        }
    }

public:
    SuffixTree(string text) {
        this->text = text;
        size = text.size();
        rootEnd = new int(-1);
        root = newNode(-1, rootEnd);
        activeNode = root;
        activeEdge = -1;
        activeLength = 0;
        remainingSuffixCount = 0;
        leafEnd = -1;

        for (int i = 0; i < size; i++) {
            extendSuffixTree(i);
        }
        setSuffixIndexByDFS(root, 0);
    }

    ~SuffixTree() {
        // Clean up the dynamically allocated memory
    }

    void print(int i, int j) {
        for (int k = i; k <= j; k++) {
            cout << text[k];
        }
    }

    void printTree(SuffixTreeNode* n, int labelHeight) {
        if (n == nullptr) return;

        if (n->start != -1) {
            print(n->start, *(n->end));
        }

        for (auto& it : n->children) {
            if (n->start != -1) cout << " [" << n->suffixIndex << "]";
            cout << endl;
            printTree(it.second, labelHeight + edgeLength(it.second));
        }
    }

    void printTree() {
        printTree(root, 0);
    }

    bool search(string pattern) {
        SuffixTreeNode* currentNode = root;
        int i = 0;
        while (i < pattern.size()) {
            if (currentNode->children.find(pattern[i]) != currentNode->children.end()) {
                currentNode = currentNode->children[pattern[i]];
                int j = currentNode->start;
                while (j <= *(currentNode->end) && i < pattern.size()) {
                    if (text[j] != pattern[i]) return false;
                    j++;
                    i++;
                }
                if (i == pattern.size()) return true;
            } else {
                return false;
            }
        }
        return true;
    }
};

int main() {
    string text = "banana";
    SuffixTree tree(text);

    cout << "Suffix Tree for " << text << ":\n";
    tree.printTree();

    string pattern = "nan";
    cout << "\nSearching for pattern \"" << pattern << "\": " << (tree.search(pattern) ? "Found" : "Not Found") << endl;

    return 0;
}
```
- **설명**:
  - `SuffixTree` 클래스를 사용하여 접미사 트리를 구축하고, 특정 패턴을 검색합니다.
  - 패턴이 문자열에 포함되어 있는지 확인합니다.

이로써 14주차 강의가 마무리됩니다. 학생들은 트라이와 접미사 트리의 기본 개념과 구현 방법을 이해하고, 이를 활용하여 다양한 프로그램을 작성하는 능력을 기르게 됩니다.