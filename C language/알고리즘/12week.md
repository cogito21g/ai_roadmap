### 알고리즘 교육과정 - 12주차: 문자열 알고리즘

**강의 목표:**
문자열 알고리즘의 기본 개념과 원리를 이해하고, 이를 활용한 다양한 알고리즘을 학습합니다. KMP 알고리즘, 보이어-무어 알고리즘, 접미사 배열과 트리 등 문자열 관련 알고리즘을 효율적으로 구현하는 방법을 학습합니다.

**강의 구성:**

#### 12. 문자열 알고리즘

**강의 내용:**
- 문자열 검색 알고리즘
- KMP 알고리즘
- 보이어-무어 알고리즘
- 접미사 배열과 트리

**실습:**
- 문자열 알고리즘 구현 및 성능 분석

### 문자열 검색 알고리즘

**강의 내용:**
- 문자열 검색의 기본 개념
  - 텍스트 내에서 특정 패턴을 찾는 문제
- 문자열 검색 알고리즘의 응용
  - 텍스트 편집기, 검색 엔진 등

**실습:**
- 문자열 검색 알고리즘 이해를 위한 간단한 예제

### KMP 알고리즘

**강의 내용:**
- KMP 알고리즘의 개념
  - 패턴 내에서 부분 일치를 활용하여 검색을 효율적으로 수행
- KMP 알고리즘의 시간 복잡도
  - O(n + m) (텍스트 길이 n, 패턴 길이 m)

**실습:**
- KMP 알고리즘 구현 예제
  ```c
  #include <stdio.h>
  #include <string.h>

  void computeLPSArray(char* pat, int M, int* lps) {
      int len = 0;
      lps[0] = 0;
      int i = 1;
      while (i < M) {
          if (pat[i] == pat[len]) {
              len++;
              lps[i] = len;
              i++;
          } else {
              if (len != 0) {
                  len = lps[len - 1];
              } else {
                  lps[i] = 0;
                  i++;
              }
          }
      }
  }

  void KMPSearch(char* pat, char* txt) {
      int M = strlen(pat);
      int N = strlen(txt);

      int lps[M];
      computeLPSArray(pat, M, lps);

      int i = 0;
      int j = 0;
      while (i < N) {
          if (pat[j] == txt[i]) {
              j++;
              i++;
          }

          if (j == M) {
              printf("Found pattern at index %d\n", i - j);
              j = lps[j - 1];
          } else if (i < N && pat[j] != txt[i]) {
              if (j != 0)
                  j = lps[j - 1];
              else
                  i++;
          }
      }
  }

  int main() {
      char txt[] = "ABABDABACDABABCABAB";
      char pat[] = "ABABCABAB";
      KMPSearch(pat, txt);
      return 0;
  }
  ```

### 보이어-무어 알고리즘

**강의 내용:**
- 보이어-무어 알고리즘의 개념
  - 패턴을 오른쪽에서 왼쪽으로 비교하며 불일치 시 점프를 통해 검색 속도 향상
- 보이어-무어 알고리즘의 시간 복잡도
  - 평균 O(n/m) (텍스트 길이 n, 패턴 길이 m)

**실습:**
- 보이어-무어 알고리즘 구현 예제
  ```c
  #include <stdio.h>
  #include <string.h>

  #define NO_OF_CHARS 256

  void badCharHeuristic(char* str, int size, int badchar[NO_OF_CHARS]) {
      int i;
      for (i = 0; i < NO_OF_CHARS; i++)
          badchar[i] = -1;
      for (i = 0; i < size; i++)
          badchar[(int) str[i]] = i;
  }

  void search(char* txt, char* pat) {
      int m = strlen(pat);
      int n = strlen(txt);

      int badchar[NO_OF_CHARS];
      badCharHeuristic(pat, m, badchar);

      int s = 0;
      while (s <= (n - m)) {
          int j = m - 1;

          while (j >= 0 && pat[j] == txt[s + j])
              j--;

          if (j < 0) {
              printf("Patterns occur at shift = %d\n", s);
              s += (s + m < n) ? m - badchar[txt[s + m]] : 1;
          } else {
              s += (j - badchar[txt[s + j]] > 1) ? j - badchar[txt[s + j]] : 1;
          }
      }
  }

  int main() {
      char txt[] = "ABAAABCD";
      char pat[] = "ABC";
      search(txt, pat);
      return 0;
  }
  ```

### 접미사 배열과 트리

**강의 내용:**
- 접미사 배열의 개념
  - 문자열의 모든 접미사를 사전순으로 정렬한 배열
- 접미사 배열의 응용
  - 문자열 검색, 문자열 비교 등
- 접미사 트리의 개념
  - 모든 접미사를 포함하는 트리 구조

**실습:**
- 접미사 배열 생성 예제
  ```c
  #include <stdio.h>
  #include <stdlib.h>
  #include <string.h>

  struct suffix {
      int index;
      char* suff;
  };

  int cmp(const void* a, const void* b) {
      struct suffix* a1 = (struct suffix*) a;
      struct suffix* b1 = (struct suffix*) b;
      return strcmp(a1->suff, b1->suff);
  }

  int* buildSuffixArray(char* txt, int n) {
      struct suffix* suffixes = (struct suffix*) malloc(n * sizeof(struct suffix));
      for (int i = 0; i < n; i++) {
          suffixes[i].index = i;
          suffixes[i].suff = (txt + i);
      }

      qsort(suffixes, n, sizeof(struct suffix), cmp);

      int* suffixArr = (int*) malloc(n * sizeof(int));
      for (int i = 0; i < n; i++)
          suffixArr[i] = suffixes[i].index;

      free(suffixes);
      return suffixArr;
  }

  void printArr(int arr[], int n) {
      for (int i = 0; i < n; i++)
          printf("%d ", arr[i]);
      printf("\n");
  }

  int main() {
      char txt[] = "banana";
      int n = strlen(txt);
      int* suffixArr = buildSuffixArray(txt, n);
      printf("Suffix Array for %s:\n", txt);
      printArr(suffixArr, n);
      free(suffixArr);
      return 0;
  }
  ```

**과제:**
- 다양한 문자열 알고리즘 문제를 해결하는 알고리즘을 구현하고, 성능을 비교
- 각 알고리즘의 시간 복잡도를 분석하고, 실제 실행 시간을 측정하여 비교

**퀴즈 및 해설:**

1. **KMP 알고리즘의 시간 복잡도는 무엇인가요?**
   - KMP 알고리즘의 시간 복잡도는 O(n + m)입니다. 여기서 n은 텍스트의 길이, m은 패턴의 길이입니다.

2. **보이어-무어 알고리즘의 기본 원리는 무엇인가요?**
   - 보이어-무어 알고리즘은 패턴을 오른쪽에서 왼쪽으로 비교하며 불일치 시 점프를 통해 검색 속도를 향상시키는 방법입니다. 이는 패턴의 마지막 문자부터 비교하며, 불일치 시 점프 길이를 계산하여 다음 비교 위치를 결정합니다.

3. **접미사 배열의 응용 사례는 무엇인가요?**
   - 접미사 배열은 문자열 검색, 문자열 비교, 문자열의 부분 문자열 찾기, 문자열의 반복 패턴 찾기 등 다양한 문자열 처리 문제에 사용됩니다.

**해설:**
1. **KMP 알고리즘의 시간 복잡도**는 O(n + m)입니다. 이는 패턴 내에서 부분 일치를 활용하여 검색을 효율적으로 수행하기 때문입니다.
2. **보이어-무어 알고리즘의 기본 원리**는 패턴을 오른쪽에서 왼쪽으로 비교하며 불일치 시 점프를 통해 검색 속도를 향상시키는 방법입니다. 이는 불일치 시 최대한 많이 점프하여 검색을 빠르게 진행합니다.
3. **접미사 배열의 응용 사례**는 문자열 검색, 문자열 비교, 문자열의 부분 문자열 찾기, 문자열의 반복 패턴 찾기 등 다양한 문자열 처리 문제에 사용됩니다. 접미사 배열은 문자열의 모든 접미사를 사전순으로 정렬하여 다양한 문제를 효율적으로 해결할 수 있습니다.

---
