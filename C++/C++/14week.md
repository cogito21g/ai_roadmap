### 14주차: 파일 입출력

#### 강의 목표
- 파일 입출력의 기본 개념 이해
- ifstream, ofstream을 사용한 파일 읽기와 쓰기
- 이진 파일 입출력 방법 이해 및 사용

#### 강의 내용

##### 1. 파일 스트림 (ifstream, ofstream)
- **파일 쓰기 (ofstream)**

```cpp
#include <iostream>
#include <fstream>
using namespace std;

int main() {
    ofstream outFile("example.txt");

    if (!outFile) {
        cerr << "File could not be opened!" << endl;
        return 1;
    }

    outFile << "Hello, World!" << endl;
    outFile << 123 << endl;

    outFile.close();
    return 0;
}
```

- **파일 읽기 (ifstream)**

```cpp
#include <iostream>
#include <fstream>
using namespace std;

int main() {
    ifstream inFile("example.txt");

    if (!inFile) {
        cerr << "File could not be opened!" << endl;
        return 1;
    }

    string str;
    int num;

    inFile >> str >> num;

    cout << "Read from file: " << str << " " << num << endl;

    inFile.close();
    return 0;
}
```

##### 2. 파일 읽기와 쓰기
- **라인 단위 파일 읽기**

```cpp
#include <iostream>
#include <fstream>
using namespace std;

int main() {
    ifstream inFile("example.txt");

    if (!inFile) {
        cerr << "File could not be opened!" << endl;
        return 1;
    }

    string line;
    while (getline(inFile, line)) {
        cout << line << endl;
    }

    inFile.close();
    return 0;
}
```

- **파일에 데이터 추가하기 (append)**

```cpp
#include <iostream>
#include <fstream>
using namespace std;

int main() {
    ofstream outFile("example.txt", ios::app);  // append 모드로 파일 열기

    if (!outFile) {
        cerr << "File could not be opened!" << endl;
        return 1;
    }

    outFile << "Appending a new line." << endl;

    outFile.close();
    return 0;
}
```

##### 3. 이진 파일 입출력
- **이진 파일 쓰기**

```cpp
#include <iostream>
#include <fstream>
using namespace std;

int main() {
    ofstream outFile("binary.dat", ios::binary);

    if (!outFile) {
        cerr << "File could not be opened!" << endl;
        return 1;
    }

    int num = 1234;
    outFile.write(reinterpret_cast<char*>(&num), sizeof(num));

    outFile.close();
    return 0;
}
```

- **이진 파일 읽기**

```cpp
#include <iostream>
#include <fstream>
using namespace std;

int main() {
    ifstream inFile("binary.dat", ios::binary);

    if (!inFile) {
        cerr << "File could not be opened!" << endl;
        return 1;
    }

    int num;
    inFile.read(reinterpret_cast<char*>(&num), sizeof(num));

    cout << "Read from binary file: " << num << endl;

    inFile.close();
    return 0;
}
```

#### 과제

1. **텍스트 파일 쓰기**
   - 사용자로부터 문자열을 입력받아 텍스트 파일에 저장하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <fstream>
using namespace std;

int main() {
    ofstream outFile("user_input.txt");

    if (!outFile) {
        cerr << "File could not be opened!" << endl;
        return 1;
    }

    string userInput;
    cout << "Enter a string to write to the file: ";
    getline(cin, userInput);

    outFile << userInput << endl;

    outFile.close();
    return 0;
}
```

2. **텍스트 파일 읽기**
   - 텍스트 파일에서 데이터를 읽어와서 화면에 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <fstream>
using namespace std;

int main() {
    ifstream inFile("user_input.txt");

    if (!inFile) {
        cerr << "File could not be opened!" << endl;
        return 1;
    }

    string line;
    while (getline(inFile, line)) {
        cout << line << endl;
    }

    inFile.close();
    return 0;
}
```

3. **이진 파일 쓰기와 읽기**
   - 사용자로부터 정수를 입력받아 이진 파일에 저장하고, 저장된 데이터를 다시 읽어와서 화면에 출력하는 프로그램을 작성하세요.

```cpp
#include <iostream>
#include <fstream>
using namespace std;

int main() {
    // 이진 파일 쓰기
    ofstream outFile("binary.dat", ios::binary);

    if (!outFile) {
        cerr << "File could not be opened!" << endl;
        return 1;
    }

    int num;
    cout << "Enter an integer to write to the binary file: ";
    cin >> num;

    outFile.write(reinterpret_cast<char*>(&num), sizeof(num));
    outFile.close();

    // 이진 파일 읽기
    ifstream inFile("binary.dat", ios::binary);

    if (!inFile) {
        cerr << "File could not be opened!" << endl;
        return 1;
    }

    int readNum;
    inFile.read(reinterpret_cast<char*>(&readNum), sizeof(readNum));

    cout << "Read from binary file: " << readNum << endl;

    inFile.close();
    return 0;
}
```

#### 퀴즈

1. **파일 입출력에 대한 설명 중 맞는 것은?**
   - A) ifstream은 파일을 쓰기 위해 사용된다.
   - B) ofstream은 파일을 읽기 위해 사용된다.
   - C) fstream은 파일을 읽기와 쓰기 모두에 사용될 수 있다.
   - D) ios::binary는 텍스트 파일을 읽기 위해 사용된다.

2. **파일 스트림을 사용한 파일 읽기 중 틀린 것은?**
   - A) ifstream 객체를 사용하여 파일을 읽을 수 있다.
   - B) getline() 함수를 사용하여 파일의 한 줄을 읽을 수 있다.
   - C) 파일이 존재하지 않을 경우 예외를 던진다.
   - D) 파일 스트림 객체를 사용하여 파일을 연다.

3. **이진 파일 입출력에 대한 설명 중 맞는 것은?**
   - A) 이진 파일은 텍스트 파일과 같은 방식으로 읽고 쓸 수 있다.
   - B) 이진 파일 입출력은 텍스트 데이터를 처리하기 위해 사용된다.
   - C) 이진 파일 입출력은 read()와 write() 함수를 사용한다.
   - D) 이진 파일은 항상 .bin 확장자를 가져야 한다.

4. **파일 스트림 객체가 하는 일은?**
   - A) 파일을 열고 닫는 역할을 한다.
   - B) 메모리를 할당하고 해제하는 역할을 한다.
   - C) 프로그램의 흐름을 제어하는 역할을 한다.
   - D) 네트워크 통신을 관리하는 역할을 한다.

#### 퀴즈 해설

1. **파일 입출력에 대한 설명 중 맞는 것은?**
   - **정답: C) fstream은 파일을 읽기와 쓰기 모두에 사용될 수 있다.**
     - 해설: fstream 객체는 파일을 읽기와 쓰기 모두에 사용할 수 있습니다.

2. **파일 스트림을 사용한 파일 읽기 중 틀린 것은?**
   - **정답: C) 파일이 존재하지 않을 경우 예외를 던진다.**
     - 해설: 파일이 존재하지 않을 경우 예외를 던지지 않으며, 대신 파일 스트림 객체의 상태를 검사하여 파일이 열렸는지 확인해야 합니다.

3. **이진 파일 입출력에 대한 설명 중 맞는 것은?**
   - **정답: C) 이진 파일 입출력은 read()와 write() 함수를 사용한다.**
     - 해설: 이진 파일 입출력은 텍스트 파일과 달리 read()와 write() 함수를 사용하여 데이터를 처리합니다.

4. **파일 스트림 객체가 하는 일은?**
   - **정답: A) 파일을 열고 닫는 역할을 한다.**
     - 해설: 파일 스트림 객체는 파일을 열고 닫는 역할을 하며, 파일 입출력 작업을 수행합니다.

다음 주차 강의 내용을 요청하시면, 15주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.