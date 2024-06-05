### C 언어 20주차 심화 교육과정 - 7주차: 문자열 처리

#### 7주차: 문자열 처리

**강의 목표:**
7주차의 목표는 문자열의 개념과 배열을 사용한 문자열 처리 방법을 이해하고, 표준 라이브러리 함수를 활용하여 문자열을 다루는 능력을 기르는 것입니다.

**강의 구성:**

##### 1. 문자열과 배열
- **강의 내용:**
  - 문자열의 개념
    - 문자열은 문자들의 배열이며, 문자열의 끝은 null 문자 (`'\0'`)로 표시
  - 문자열 초기화 및 배열과의 관계
    - 문자열 선언: `char str[] = "Hello";`
    - 문자열 초기화: `char str[6] = {'H', 'e', 'l', 'l', 'o', '\0'};`
  - 문자열 입출력
    - `scanf`와 `printf`를 사용한 문자열 입출력
    - `gets`와 `puts` 함수 사용법
- **실습:**
  - 문자열 선언 및 초기화 예제 작성
    ```c
    #include <stdio.h>

    int main() {
        char str1[] = "Hello";
        char str2[6] = {'H', 'e', 'l', 'l', 'o', '\0'};

        printf("str1: %s\n", str1);
        printf("str2: %s\n", str2);

        return 0;
    }
    ```

  - 문자열 입출력 예제 작성
    ```c
    #include <stdio.h>

    int main() {
        char str[100];

        printf("Enter a string: ");
        gets(str);
        printf("You entered: ");
        puts(str);

        return 0;
    }
    ```

##### 2. 문자열 함수
- **강의 내용:**
  - 표준 라이브러리 문자열 함수
    - `strlen`: 문자열 길이 계산
    - `strcpy`: 문자열 복사
    - `strcat`: 문자열 연결
    - `strcmp`: 문자열 비교
  - 각 함수의 사용법 및 예제
    - `strlen` 예제: 문자열의 길이를 계산하여 출력
    - `strcpy` 예제: 문자열을 다른 변수에 복사
    - `strcat` 예제: 두 문자열을 연결하여 새로운 문자열 생성
    - `strcmp` 예제: 두 문자열을 비교하여 결과 출력
- **실습:**
  - `strlen`을 사용한 문자열 길이 계산 예제 작성
    ```c
    #include <stdio.h>
    #include <string.h>

    int main() {
        char str[] = "Hello, World!";
        int length = strlen(str);

        printf("Length of the string: %d\n", length);

        return 0;
    }
    ```

  - `strcpy`를 사용한 문자열 복사 예제 작성
    ```c
    #include <stdio.h>
    #include <string.h>

    int main() {
        char src[] = "Hello";
        char dest[20];

        strcpy(dest, src);

        printf("Source: %s\n", src);
        printf("Destination: %s\n", dest);

        return 0;
    }
    ```

  - `strcat`을 사용한 문자열 연결 예제 작성
    ```c
    #include <stdio.h>
    #include <string.h>

    int main() {
        char str1[20] = "Hello, ";
        char str2[] = "World!";

        strcat(str1, str2);

        printf("Resulting string: %s\n", str1);

        return 0;
    }
    ```

  - `strcmp`를 사용한 문자열 비교 예제 작성
    ```c
    #include <stdio.h>
    #include <string.h>

    int main() {
        char str1[] = "Hello";
        char str2[] = "World";

        int result = strcmp(str1, str2);

        if (result == 0) {
            printf("Strings are equal\n");
        } else if (result < 0) {
            printf("str1 is less than str2\n");
        } else {
            printf("str1 is greater than str2\n");
        }

        return 0;
    }
    ```

##### 3. 문자열의 기타 처리 방법
- **강의 내용:**
  - 문자열 검색
    - `strchr`: 특정 문자 검색
    - `strstr`: 특정 문자열 검색
  - 문자열 변환
    - `atoi`: 문자열을 정수로 변환
    - `atof`: 문자열을 실수로 변환
- **실습:**
  - `strchr`을 사용한 문자 검색 예제 작성
    ```c
    #include <stdio.h>
    #include <string.h>

    int main() {
        char str[] = "Hello, World!";
        char *ptr;

        ptr = strchr(str, 'W');

        if (ptr != NULL) {
            printf("Character found at position: %ld\n", ptr - str);
        } else {
            printf("Character not found\n");
        }

        return 0;
    }
    ```

  - `atoi`와 `atof`를 사용한 문자열 변환 예제 작성
    ```c
    #include <stdio.h>
    #include <stdlib.h>

    int main() {
        char intStr[] = "123";
        char floatStr[] = "456.78";

        int intValue = atoi(intStr);
        float floatValue = atof(floatStr);

        printf("Integer value: %d\n", intValue);
        printf("Float value: %.2f\n", floatValue);

        return 0;
    }
    ```

**과제:**
7주차 과제는 다음과 같습니다.
- 사용자로부터 두 개의 문자열을 입력받아 두 문자열을 비교하는 프로그램 작성
- 문자열을 입력받아 문자열의 길이와 첫 번째 출현하는 특정 문자의 위치를 출력하는 프로그램 작성
- 두 개의 문자열을 입력받아 첫 번째 문자열의 끝에 두 번째 문자열을 연결하여 출력하는 프로그램 작성

**퀴즈 및 해설:**

1. **문자열의 정의는 무엇인가요?**
   - 문자열은 문자들의 배열이며, 문자열의 끝은 null 문자 (`'\0'`)로 표시됩니다.

2. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    char str[] = "Hello, World!";
    printf("%c\n", str[7]);
    ```
   - 출력 결과는 `W`입니다. 배열의 인덱스 7에 있는 문자는 `W`입니다.

3. **`strlen` 함수의 역할은 무엇인가요?**
   - `strlen` 함수는 문자열의 길이를 계산하여 반환합니다. null 문자는 길이에 포함되지 않습니다.

4. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    char str1[10] = "Hello";
    char str2[10];
    strcpy(str2, str1);
    printf("%s\n", str2);
    ```
   - 출력 결과는 `Hello`입니다. `strcpy` 함수는 `str1`의 문자열을 `str2`로 복사합니다.

5. **`strcmp` 함수의 반환 값이 0인 경우는 무엇을 의미하나요?**
   - `strcmp` 함수의 반환 값이 0이면 두 문자열이 같다는 의미입니다.

**해설:**
1. 문자열은 문자들의 배열이며, 문자열의 끝은 null 문자 (`'\0'`)로 표시됩니다.
2. 배열의 인덱스 7에 있는 문자는 `W`이므로 출력 결과는 `W`입니다.
3. `strlen` 함수는 문자열의 길이를 계산하여 반환합니다. null 문자는 길이에 포함되지 않습니다.
4. `strcpy` 함수는 `str1`의 문자열을 `str2`로 복사하므로, `str2`의 값은 `Hello`입니다.
5. `strcmp` 함수의 반환 값이 0이면 두 문자열이 같다는 의미입니다.

이 7주차 강의는 학생들이 문자열의 개념을 이해하고, 표준 라이브러리 함수를 활용하여 문자열을 효과적으로 처리하는 능력을 기를 수 있도록 도와줍니다.