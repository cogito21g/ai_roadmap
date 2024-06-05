### 27주차: Python 프로그래밍

#### 강의 목표
- Python 기초 문법 이해 및 사용
- Python 표준 라이브러리 및 패키지 사용
- Python을 이용한 데이터 분석 및 시각화

#### 강의 내용

##### 1. Python 기초
- **Python 기본 문법**

```python
# 변수와 데이터 타입
x = 5
y = 3.14
name = "Alice"

# 리스트
numbers = [1, 2, 3, 4, 5]

# 튜플
coordinates = (10, 20)

# 딕셔너리
student = {"name": "John", "age": 21}

# 조건문
if x > 0:
    print("x is positive")
else:
    print("x is non-positive")

# 반복문
for number in numbers:
    print(number)

# 함수
def add(a, b):
    return a + b

result = add(3, 4)
print(result)
```

- **클래스와 객체지향 프로그래밍**

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

# 객체 생성 및 사용
person1 = Person("Alice", 30)
person1.greet()
```

##### 2. Python 표준 라이브러리 및 패키지 사용
- **파일 입출력**

```python
# 파일 쓰기
with open("example.txt", "w") as file:
    file.write("Hello, World!\n")

# 파일 읽기
with open("example.txt", "r") as file:
    content = file.read()
    print(content)
```

- **시간과 날짜 처리**

```python
import datetime

# 현재 시간과 날짜
now = datetime.datetime.now()
print(now)

# 특정 날짜 생성
specific_date = datetime.datetime(2022, 1, 1)
print(specific_date)

# 날짜 차이 계산
delta = now - specific_date
print(f"Days since {specific_date}: {delta.days}")
```

- **정규 표현식**

```python
import re

pattern = r"\b[A-Za-z]+\b"
text = "Hello, World! 123"
matches = re.findall(pattern, text)
print(matches)
```

##### 3. Python을 이용한 데이터 분석 및 시각화
- **Pandas를 이용한 데이터 분석**

```python
import pandas as pd

# 데이터프레임 생성
data = {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
df = pd.DataFrame(data)
print(df)

# 데이터 불러오기 및 저장
df.to_csv("data.csv", index=False)
df2 = pd.read_csv("data.csv")
print(df2)
```

- **Matplotlib를 이용한 데이터 시각화**

```python
import matplotlib.pyplot as plt

# 데이터
x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 35]

# 라인 그래프
plt.plot(x, y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Line Graph Example")
plt.show()
```

- **Seaborn을 이용한 고급 시각화**

```python
import seaborn as sns
import pandas as pd

# 데이터 생성
data = pd.DataFrame({
    "x": [1, 2, 3, 4, 5],
    "y": [10, 20, 25, 30, 35]
})

# Scatter plot
sns.scatterplot(data=data, x="x", y="y")
plt.show()
```

#### 과제

1. **Python 기본 문법을 사용하여 간단한 계산기를 구현하세요.**

```python
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        return "Error: Division by zero"
    return a / b

print("Simple Calculator")
a = float(input("Enter first number: "))
b = float(input("Enter second number: "))
operation = input("Enter operation (+, -, *, /): ")

if operation == '+':
    print(f"Result: {add(a, b)}")
elif operation == '-':
    print(f"Result: {subtract(a, b)}")
elif operation == '*':
    print(f"Result: {multiply(a, b)}")
elif operation == '/':
    print(f"Result: {divide(a, b)}")
else:
    print("Invalid operation")
```

2. **Pandas를 사용하여 데이터프레임을 생성하고, 특정 열의 평균을 계산하세요.**

```python
import pandas as pd

# 데이터프레임 생성
data = {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
df = pd.DataFrame(data)

# 특정 열의 평균 계산
average_age = df["age"].mean()
print(f"Average age: {average_age}")
```

3. **Matplotlib를 사용하여 막대 그래프를 생성하세요.**

```python
import matplotlib.pyplot as plt

# 데이터
categories = ["A", "B", "C", "D"]
values = [4, 7, 1, 8]

# 막대 그래프
plt.bar(categories, values)
plt.xlabel("Categories")
plt.ylabel("Values")
plt.title("Bar Graph Example")
plt.show()
```

4. **Seaborn을 사용하여 히스토그램을 생성하세요.**

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 생성
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]

# 히스토그램
sns.histplot(data, bins=4, kde=True)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram Example")
plt.show()
```

#### 퀴즈

1. **Python의 기본 데이터 타입이 아닌 것은?**
   - A) 리스트
   - B) 튜플
   - C) 딕셔너리
   - D) 배열

2. **Python에서 파일을 읽기 모드로 열 때 사용하는 메서드는?**
   - A) open("filename", "r")
   - B) read("filename")
   - C) file("filename")
   - D) load("filename")

3. **Pandas 라이브러리에서 데이터프레임을 생성할 때 사용하는 함수는?**
   - A) pd.DataFrame()
   - B) pd.read_csv()
   - C) pd.Series()
   - D) pd.create_df()

4. **Matplotlib에서 라인 그래프를 생성하는 함수는?**
   - A) plt.bar()
   - B) plt.plot()
   - C) plt.hist()
   - D) plt.scatter()

#### 퀴즈 해설

1. **Python의 기본 데이터 타입이 아닌 것은?**
   - **정답: D) 배열**
     - 해설: Python의 기본 데이터 타입에는 리스트, 튜플, 딕셔너리가 있지만, 배열은 기본 데이터 타입이 아닙니다. 배열은 numpy 라이브러리를 통해 사용할 수 있습니다.

2. **Python에서 파일을 읽기 모드로 열 때 사용하는 메서드는?**
   - **정답: A) open("filename", "r")**
     - 해설: 파일을 읽기 모드로 열 때는 `open("filename", "r")`을 사용합니다.

3. **Pandas 라이브러리에서 데이터프레임을 생성할 때 사용하는 함수는?**
   - **정답: A) pd.DataFrame()**
     - 해설: Pandas에서 데이터프레임을 생성할 때는 `pd.DataFrame()` 함수를 사용합니다.

4. **Matplotlib에서 라인 그래프를 생성하는 함수는?**
   - **정답: B) plt.plot()**
     - 해설: Matplotlib에서 라인 그래프를 생성할 때는 `plt.plot()` 함수를 사용합니다.

다음 주차 강의 내용을 요청하시면, 28주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.