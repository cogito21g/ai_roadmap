### C 언어 20주차 심화 교육과정 - 9주차: 파일 입출력

#### 9주차: 파일 입출력

**강의 목표:**
9주차의 목표는 파일 입출력의 개념과 사용법을 이해하고, 텍스트 및 이진 파일을 처리하는 방법을 배우는 것입니다.

**강의 구성:**

##### 1. 파일 포인터 (FILE)
- **강의 내용:**
  - 파일 입출력의 개념
    - 파일의 종류: 텍스트 파일과 이진 파일
  - FILE 포인터의 개념
    - FILE 포인터는 파일과의 연결을 관리하는 구조체
  - 표준 파일 스트림
    - 표준 입력(stdin), 표준 출력(stdout), 표준 오류(stderr)
- **실습:**
  - 간단한 파일 포인터 예제 작성
    ```c
    #include <stdio.h>

    int main() {
        FILE *file;
        file = fopen("example.txt", "r");

        if (file == NULL) {
            printf("File could not be opened\n");
            return 1;
        }

        printf("File opened successfully\n");
        fclose(file);
        return 0;
    }
    ```

##### 2. 파일 열기, 읽기, 쓰기, 닫기
- **강의 내용:**
  - 파일 열기 모드
    - "r": 읽기 모드
    - "w": 쓰기 모드
    - "a": 추가 모드
    - "r+": 읽기 및 쓰기 모드
    - "w+": 읽기 및 쓰기 모드 (기존 파일 덮어쓰기)
    - "a+": 읽기 및 추가 모드
  - 파일 읽기
    - `fscanf`, `fgets`, `fgetc` 함수 사용법
  - 파일 쓰기
    - `fprintf`, `fputs`, `fputc` 함수 사용법
  - 파일 닫기
    - `fclose` 함수 사용법
- **실습:**
  - 파일 쓰기 예제 작성
    ```c
    #include <stdio.h>

    int main() {
        FILE *file;
        file = fopen("example.txt", "w");

        if (file == NULL) {
            printf("File could not be opened\n");
            return 1;
        }

        fprintf(file, "Hello, World!\n");
        fputs("This is an example.\n", file);

        fclose(file);
        return 0;
    }
    ```

  - 파일 읽기 예제 작성
    ```c
    #include <stdio.h>

    int main() {
        FILE *file;
        char buffer[100];

        file = fopen("example.txt", "r");

        if (file == NULL) {
            printf("File could not be opened\n");
            return 1;
        }

        while (fgets(buffer, 100, file) != NULL) {
            printf("%s", buffer);
        }

        fclose(file);
        return 0;
    }
    ```

##### 3. 이진 파일 입출력
- **강의 내용:**
  - 이진 파일의 개념
    - 텍스트 파일과의 차이점
  - 이진 파일 입출력 함수
    - `fwrite`와 `fread` 함수 사용법
    - 구조체 데이터를 이진 파일로 저장하고 읽기
- **실습:**
  - 이진 파일 쓰기 예제 작성
    ```c
    #include <stdio.h>

    struct Student {
        char name[50];
        int age;
        float gpa;
    };

    int main() {
        FILE *file;
        struct Student student = {"John Doe", 20, 3.5};

        file = fopen("student.dat", "wb");

        if (file == NULL) {
            printf("File could not be opened\n");
            return 1;
        }

        fwrite(&student, sizeof(struct Student), 1, file);

        fclose(file);
        return 0;
    }
    ```

  - 이진 파일 읽기 예제 작성
    ```c
    #include <stdio.h>

    struct Student {
        char name[50];
        int age;
        float gpa;
    };

    int main() {
        FILE *file;
        struct Student student;

        file = fopen("student.dat", "rb");

        if (file == NULL) {
            printf("File could not be opened\n");
            return 1;
        }

        fread(&student, sizeof(struct Student), 1, file);

        printf("Name: %s\n", student.name);
        printf("Age: %d\n", student.age);
        printf("GPA: %.2f\n", student.gpa);

        fclose(file);
        return 0;
    }
    ```

**과제:**
9주차 과제는 다음과 같습니다.
- 텍스트 파일에서 학생들의 이름과 성적을 읽어 평균 성적을 계산하고 출력하는 프로그램 작성
- 이진 파일에 여러 책 정보를 저장하고, 저장된 책 정보를 읽어 출력하는 프로그램 작성
- 텍스트 파일에서 특정 단어를 검색하여 해당 단어의 빈도를 출력하는 프로그램 작성

**퀴즈 및 해설:**

1. **파일 포인터의 역할은 무엇인가요?**
   - 파일 포인터는 파일과의 연결을 관리하는 구조체입니다. 파일 입출력 작업에서 파일 포인터를 통해 파일을 읽거나 쓸 수 있습니다.

2. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    FILE *file;
    file = fopen("example.txt", "w");
    fprintf(file, "Hello, World!\n");
    fclose(file);
    file = fopen("example.txt", "r");
    char buffer[20];
    fgets(buffer, 20, file);
    printf("%s", buffer);
    fclose(file);
    ```
   - 출력 결과는 `Hello, World!\n`입니다. 첫 번째 `fopen`은 파일을 쓰기 모드로 열고, `fprintf`로 문자열을 파일에 씁니다. 두 번째 `fopen`은 파일을 읽기 모드로 열고, `fgets`로 문자열을 읽어 출력합니다.

3. **이진 파일 입출력 함수 `fwrite`와 `fread`의 역할은 무엇인가요?**
   - `fwrite` 함수는 데이터를 이진 파일에 씁니다. `fread` 함수는 이진 파일에서 데이터를 읽습니다. 이 두 함수는 구조체와 같은 복잡한 데이터를 파일로 저장하거나 읽는 데 사용됩니다.

4. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    struct Point {
        int x;
        int y;
    };

    FILE *file;
    struct Point p = {10, 20};
    file = fopen("point.dat", "wb");
    fwrite(&p, sizeof(struct Point), 1, file);
    fclose(file);
    file = fopen("point.dat", "rb");
    fread(&p, sizeof(struct Point), 1, file);
    printf("x: %d, y: %d\n", p.x, p.y);
    fclose(file);
    ```
   - 출력 결과는 `x: 10, y: 20`입니다. `fwrite` 함수는 구조체 `p`를 이진 파일에 저장하고, `fread` 함수는 저장된 구조체를 읽어 `p`에 복원합니다.

5. **파일을 열 때 `fopen` 함수가 반환하는 값은 무엇인가요?**
   - `fopen` 함수는 파일이 성공적으로 열리면 파일 포인터를 반환하고, 파일을 열지 못하면 `NULL`을 반환합니다. `NULL`을 반환하면 파일 열기에 실패한 것을 의미합니다.

**해설:**
1. 파일 포인터는 파일과의 연결을 관리하는 구조체로, 파일 입출력 작업에서 파일을 읽거나 쓸 때 사용됩니다.
2. 첫 번째 `fopen`은 파일을 쓰기 모드로 열고, `fprintf`로 문자열을 파일에 씁니다. 두 번째 `fopen`은 파일을 읽기 모드로 열고, `fgets`로 문자열을 읽어 출력하므로 출력 결과는 `Hello, World!\n`입니다.
3. `fwrite` 함수는 데이터를 이진 파일에 쓰고, `fread` 함수는 이진 파일에서 데이터를 읽습니다.
4. `fwrite` 함수는 구조체 `p`를 이진 파일에 저장하고, `fread` 함수는 저장된 구조체를 읽어 `p`에 복원하므로 출력 결과는 `x: 10, y: 20`입니다.
5. `fopen` 함수는 파일이 성공적으로 열리면 파일 포인터를 반환하고, 실패하면 `NULL`을 반환합니다.

이 9주차 강의는 학생들이 파일 입출력의 개념을 이해하고, 텍스트 및 이진 파일을 처리하는 능력을 기를 수 있도록 도와줍니다.