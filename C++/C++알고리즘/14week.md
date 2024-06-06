### 14주차 강의 계획안

#### 강의 주제: 문자열 알고리즘
- 문자열 검색 알고리즘 (KMP, Rabin-Karp)
- 트라이(Trie) 자료구조

---

### 강의 내용

#### 1. 문자열 검색 알고리즘
- **개념**: 주어진 텍스트 내에서 패턴을 검색하는 알고리즘
- **응용**: 텍스트 편집기, 웹 검색 엔진 등

**KMP 알고리즘 (Knuth-Morris-Pratt Algorithm)**
- **개념**: 접두사와 접미사의 일치를 이용해 검색 시간을 줄이는 알고리즘
- **시간 복잡도**: O(n + m) (n은 텍스트 길이, m은 패턴 길이)

**예제**: KMP 알고리즘
```cpp
#include <iostream>
#include <vector>
using namespace std;

void computeLPSArray(string pat, int M, vector<int>& lps) {
    int length = 0;
    lps[0] = 0;
    int i = 1;
    while (i < M) {
        if (pat[i] == pat[length]) {
            length++;
            lps[i] = length;
            i++;
        } else {
            if (length != 0) {
                length = lps[length - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
}

void KMPSearch(string pat, string txt) {
    int M = pat.length();
    int N = txt.length();

    vector<int> lps(M);
    computeLPSArray(pat, M, lps);

    int i = 0;
    int j = 0;
    while (i < N) {
        if (pat[j] == txt[i]) {
            j++;
            i++;
        }

        if (j == M) {
            cout << "Found pattern at index " << i - j << endl;
            j = lps[j - 1];
        } else if (i < N && pat[j] != txt[i]) {
            if (j != 0) {
                j = lps[j - 1];
            } else {
                i++;
            }
        }
    }
}

int main() {
    string txt = "ABABDABACDABABCABAB";
    string pat = "ABABCABAB";
    KMPSearch(pat, txt);
    return 0;
}
```

**Rabin-Karp 알고리즘**
- **개념**: 해시를 이용해 패턴과 텍스트의 서브스트링을 비교하는 알고리즘
- **시간 복잡도**: 평균 O(n + m), 최악 O(nm)

**예제**: Rabin-Karp 알고리즘
```cpp
#include <iostream>
using namespace std;

#define d 256

void search(string pat, string txt, int q) {
    int M = pat.length();
    int N = txt.length();
    int i, j;
    int p = 0;
    int t = 0;
    int h = 1;

    for (i = 0; i < M - 1; i++)
        h = (h * d) % q;

    for (i = 0; i < M; i++) {
        p = (d * p + pat[i]) % q;
        t = (d * t + txt[i]) % q;
    }

    for (i = 0; i <= N - M; i++) {
        if (p == t) {
            for (j = 0; j < M; j++) {
                if (txt[i + j] != pat[j])
                    break;
            }
            if (j == M)
                cout << "Pattern found at index " << i << endl;
        }
        if (i < N - M) {
            t = (d * (t - txt[i] * h) + txt[i + M]) % q;
            if (t < 0)
                t = (t + q);
        }
    }
}

int main() {
    string txt = "GEEKS FOR GEEKS";
    string pat = "GEEK";
    int q = 101;
    search(pat, txt, q);
    return 0;
}
```

#### 2. 트라이(Trie) 자료구조
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

---

### 과제

#### 과제 1: KMP 알고리즘을 이용한 문자열 검색
사용자로부터 텍스트와 패턴을 입력받아 KMP 알고리즘을 사용해 패턴을 검색하는 프로그램을 작성하세요.

**코드 예시**:
```cpp
#include <iostream>
#include <vector>
using namespace std;

void computeLPSArray(string pat, int M, vector<int>& lps) {
    int length = 0;
    lps[0] = 0;
    int i = 1;
    while (i < M) {
        if (pat[i] == pat[length]) {
            length++;
            lps[i] = length;
            i++;
        } else {
            if (length != 0) {
                length = lps[length - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
}

void KMPSearch(string pat, string txt) {
    int M = pat.length();
    int N = txt.length();

    vector<int> lps(M);
    computeLPSArray(pat, M, lps);

    int i = 0;
    int j = 0;
    while (i < N) {
        if (pat[j] == txt[i]) {
            j++;
            i++;
        }

        if (j == M) {
            cout << "Found pattern at index " << i - j << endl;
            j = lps[j - 1];
        } else if (i < N && pat[j] != txt[i]) {
            if (j != 0) {
                j = lps[j - 1];
            } else {
                i++;
            }
        }
    }
}

int main() {
    string txt, pat;
    cout << "Enter the text: ";
    cin >> txt;
    cout << "Enter the pattern: ";
    cin >> pat;
    KMPSearch(pat, txt);
    return 0;
}
```

**해설**:
1. 사용자로부터 텍스트와 패턴을 입력받습니다.
2. `KMPSearch` 함수를 사용해 패턴을 검색하고, 결과를 출력합니다.

#### 과제 2: 트라이 자료구조를 이용한 문자열 검색
사용자로부터 단어 리스트와 검색할 단어를 입력받아 트라이 자료구조를 사용해 단어를 검색하는 프로그램을 작성하세요.

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
    for (int i = 0; i < ALPHABET_SIZE; i

++)
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

---

### 퀴즈

#### 퀴즈 1: KMP 알고리즘의 시간 복잡도는 무엇인가요?
1. O(n)
2. O(m)
3. O(n + m)
4. O(n * m)

**정답**: 3. O(n + m)

#### 퀴즈 2: Rabin-Karp 알고리즘에서 사용되는 해시 함수의 목적은 무엇인가요?
1. 패턴을 압축하기 위해
2. 패턴과 텍스트의 서브스트링을 비교하기 위해
3. 텍스트를 정렬하기 위해
4. 텍스트를 분할하기 위해

**정답**: 2. 패턴과 텍스트의 서브스트링을 비교하기 위해

#### 퀴즈 3: 트라이 자료구조의 주요 특징은 무엇인가요?
1. 정렬된 배열을 사용한다.
2. 문자열의 접두사를 효율적으로 저장하고 검색할 수 있다.
3. 문자열의 접미사를 효율적으로 저장하고 검색할 수 있다.
4. 문자열을 압축 저장할 수 있다.

**정답**: 2. 문자열의 접두사를 효율적으로 저장하고 검색할 수 있다.

이 계획안은 14주차에 필요한 문자열 알고리즘의 개념과 실습을 통해 학습할 수 있도록 구성되었습니다. 과제와 퀴즈를 통해 학습한 내용을 복습하고, 이해도를 높일 수 있습니다.