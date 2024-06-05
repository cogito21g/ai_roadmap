### C 언어 20주차 심화 교육과정 - 8주차: 구조체

#### 8주차: 구조체

**강의 목표:**
8주차의 목표는 구조체의 개념과 사용법을 이해하고, 구조체 배열과 구조체 포인터를 활용하여 복잡한 데이터를 관리하는 능력을 기르는 것입니다.

**강의 구성:**

##### 1. 구조체 정의 및 사용
- **강의 내용:**
  - 구조체의 개념
    - 여러 데이터 타입을 하나로 묶어 새로운 데이터 타입을 정의하는 방법
  - 구조체 선언 및 정의
    - 구조체 선언: `struct StructName { dataType member1; dataType member2; ... };`
    - 구조체 변수 선언 및 초기화
  - 구조체 멤버 접근
    - 점(.) 연산자를 사용하여 구조체 멤버에 접근
- **실습:**
  - 구조체 선언 및 정의 예제 작성
    ```c
    #include <stdio.h>

    struct Student {
        char name[50];
        int age;
        float gpa;
    };

    int main() {
        struct Student student1 = {"John Doe", 20, 3.5};

        printf("Name: %s\n", student1.name);
        printf("Age: %d\n", student1.age);
        printf("GPA: %.2f\n", student1.gpa);

        return 0;
    }
    ```

##### 2. 구조체 배열
- **강의 내용:**
  - 구조체 배열의 개념
    - 동일한 구조체 타입의 여러 변수를 배열로 관리
  - 구조체 배열 선언 및 초기화
    - 구조체 배열 선언: `struct StructName arrayName[arraySize];`
    - 구조체 배열 초기화: 배열의 각 요소에 구조체 변수 할당
  - 구조체 배열 요소 접근
    - 점(.) 연산자를 사용하여 각 요소의 멤버에 접근
- **실습:**
  - 구조체 배열 선언 및 초기화 예제 작성
    ```c
    #include <stdio.h>

    struct Student {
        char name[50];
        int age;
        float gpa;
    };

    int main() {
        struct Student students[3] = {
            {"Alice", 21, 3.8},
            {"Bob", 22, 3.2},
            {"Charlie", 20, 3.6}
        };

        for (int i = 0; i < 3; i++) {
            printf("Student %d\n", i + 1);
            printf("Name: %s\n", students[i].name);
            printf("Age: %d\n", students[i].age);
            printf("GPA: %.2f\n\n", students[i].gpa);
        }

        return 0;
    }
    ```

##### 3. 구조체와 포인터
- **강의 내용:**
  - 구조체 포인터의 개념
    - 구조체 변수의 주소를 저장하는 포인터
  - 구조체 포인터 선언 및 초기화
    - 구조체 포인터 선언: `struct StructName *pointerName;`
    - 구조체 포인터 초기화: `pointerName = &structVariable;`
  - 구조체 포인터를 통한 멤버 접근
    - 화살표(->) 연산자를 사용하여 구조체 포인터의 멤버에 접근
- **실습:**
  - 구조체 포인터 예제 작성
    ```c
    #include <stdio.h>

    struct Student {
        char name[50];
        int age;
        float gpa;
    };

    int main() {
        struct Student student1 = {"John Doe", 20, 3.5};
        struct Student *ptr = &student1;

        printf("Name: %s\n", ptr->name);
        printf("Age: %d\n", ptr->age);
        printf("GPA: %.2f\n", ptr->gpa);

        return 0;
    }
    ```

  - 구조체 배열과 포인터를 사용한 예제 작성
    ```c
    #include <stdio.h>

    struct Student {
        char name[50];
        int age;
        float gpa;
    };

    int main() {
        struct Student students[3] = {
            {"Alice", 21, 3.8},
            {"Bob", 22, 3.2},
            {"Charlie", 20, 3.6}
        };

        struct Student *ptr;
        for (int i = 0; i < 3; i++) {
            ptr = &students[i];
            printf("Student %d\n", i + 1);
            printf("Name: %s\n", ptr->name);
            printf("Age: %d\n", ptr->age);
            printf("GPA: %.2f\n\n", ptr->gpa);
        }

        return 0;
    }
    ```

**과제:**
8주차 과제는 다음과 같습니다.
- 여러 도서 정보를 저장하는 구조체를 정의하고, 구조체 배열을 사용하여 도서 목록을 출력하는 프로그램 작성
- 학생들의 성적을 관리하는 프로그램 작성: 학생 구조체를 정의하고, 구조체 포인터를 사용하여 성적을 업데이트하고 출력
- 복소수의 덧셈을 수행하는 프로그램 작성: 복소수를 저장하는 구조체를 정의하고, 구조체 포인터를 사용하여 덧셈을 구현

**퀴즈 및 해설:**

1. **구조체의 정의는 무엇인가요?**
   - 구조체는 여러 데이터 타입을 하나로 묶어 새로운 데이터 타입을 정의하는 방법입니다. 구조체는 연관된 데이터를 하나의 단위로 관리할 수 있게 해줍니다.

2. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    struct Point {
        int x;
        int y;
    };

    int main() {
        struct Point p1 = {10, 20};
        printf("x: %d, y: %d\n", p1.x, p1.y);
        return 0;
    }
    ```
   - 출력 결과는 `x: 10, y: 20`입니다. 구조체 `Point`의 멤버 `x`와 `y`에 접근하여 값을 출력합니다.

3. **구조체 배열의 선언 방법은 무엇인가요?**
   - 구조체 배열의 선언 방법은 `struct StructName arrayName[arraySize];`입니다. 예를 들어, `struct Student students[3];`는 `Student` 구조체의 배열을 선언합니다.

4. **다음 코드의 출력 결과는 무엇인가요?**
    ```c
    struct Point {
        int x;
        int y;
    };

    int main() {
        struct Point p1 = {10, 20};
        struct Point *ptr = &p1;
        printf("x: %d, y: %d\n", ptr->x, ptr->y);
        return 0;
    }
    ```
   - 출력 결과는 `x: 10, y: 20`입니다. `ptr`은 `p1`의 주소를 가리키고, `ptr->x`와 `ptr->y`를 통해 구조체 멤버에 접근합니다.

5. **구조체 포인터를 사용하여 구조체 멤버에 접근하는 방법은 무엇인가요?**
   - 구조체 포인터를 사용하여 구조체 멤버에 접근하는 방법은 화살표(->) 연산자를 사용하는 것입니다. 예를 들어, `ptr->member`는 포인터 `ptr`이 가리키는 구조체의 `member`에 접근합니다.

**해설:**
1. 구조체는 여러 데이터 타입을 하나로 묶어 새로운 데이터 타입을 정의하는 방법입니다.
2. `p1` 구조체의 멤버 `x`와 `y`에 접근하여 값을 출력하므로, 출력 결과는 `x: 10, y: 20`입니다.
3. 구조체 배열은 `struct StructName arrayName[arraySize];`와 같이 선언합니다.
4. `ptr`은 `p1`의 주소를 가리키며, `ptr->x`와 `ptr->y`를 통해 멤버에 접근하므로 출력 결과는 `x: 10, y: 20`입니다.
5. 구조체 포인터를 사용하여 구조체 멤버에 접근하려면 화살표(->) 연산자를 사용합니다.

이 8주차 강의는 학생들이 구조체의 개념을 이해하고, 구조체 배열과 포인터를 활용하여 복잡한 데이터를 관리하는 능력을 기를 수 있도록 도와줍니다.