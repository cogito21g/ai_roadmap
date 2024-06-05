### 5주차 강의 상세 계획: 양자 알고리즘 3 - 쇼어 알고리즘

#### 강의 목표
- 쇼어 알고리즘의 개념과 원리 이해
- 쇼어 알고리즘의 주요 응용 분야 학습
- 쇼어 알고리즘을 구현하는 방법 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 쇼어 알고리즘의 개념 (20분)

##### 쇼어 알고리즘이란?
- **정의**: 정수의 소인수 분해를 효율적으로 수행하는 양자 알고리즘.
- **장점**: 고전 알고리즘의 지수 시간 복잡도에 비해 다항 시간 복잡도를 가짐.

##### 쇼어 알고리즘의 기본 원리
- **양자 병렬성**: 양자 중첩과 얽힘을 이용해 병렬로 연산 수행.
- **주기성 탐지**: 고유 주기를 찾는 과정에서 양자 푸리에 변환(QFT) 사용.

#### 1.2 쇼어 알고리즘의 주요 응용 분야 (20분)
- **암호 해독**: RSA 암호화 시스템의 보안 기반을 약화시킴.
- **정수론**: 수학적 연구에서의 소인수 분해 문제 해결.
- **최적화 문제**: 주기성을 가지는 다양한 최적화 문제에 응용 가능.

#### 1.3 쇼어 알고리즘의 단계 (20분)

##### 기본 개념
- **양자 상태 준비**: 초기 상태에서 모든 가능한 값을 중첩 상태로 준비.
- **모듈러 지수 연산**: 주어진 함수를 양자 병렬성으로 계산.
- **양자 푸리에 변환**: QFT를 사용하여 주기성을 탐지.
- **고유 주기 찾기**: 측정을 통해 주기 정보를 추출하고 소인수 분해 수행.

##### 쇼어 알고리즘의 단계
1. **입력 및 초기화**: 소인수 분해할 수 \(N\)과 초기 상태 준비.
2. **모듈러 지수 연산**: 함수 \(f(x) = a^x \mod N\) 계산.
3. **양자 푸리에 변환**: QFT를 사용하여 주기 탐지.
4. **고유 주기 찾기**: 주기를 이용해 소인수 분해 수행.
5. **소인수 분해 완료**: 고유 주기를 이용해 \(N\)의 소인수 추출.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 쇼어 알고리즘 구현 실습

##### 필요 라이브러리 설치
```bash
pip install qiskit
```

##### 쇼어 알고리즘 구현 코드 (Python)
```python
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
from qiskit.visualization import plot_histogram

def qpe_amod15(a):
    n_count = 8
    qc = QuantumCircuit(4 + n_count, n_count)
    for q in range(n_count):
        qc.h(q)  # 초기 상태 준비

    qc.x(3+n_count)  # |1> 상태 준비

    for q in range(n_count):  # 제어-모듈러 지수 연산
        qc.append(c_amod15(a, 2**q), [q] + [i+n_count for i in range(4)])

    qc.append(qft_dagger(n_count), range(n_count))  # QFT 적용
    qc.measure(range(n_count), range(n_count))
    return qc

def c_amod15(a, power):
    U = QuantumCircuit(4)
    for iteration in range(power):
        if a == 2:
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        elif a == 7:
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
        elif a == 8:
            U.swap(1, 2)
            U.swap(0, 1)
            U.swap(2, 3)
        elif a == 11:
            U.swap(2, 3)
            U.swap(0, 1)
            U.swap(1, 2)
        elif a == 13:
            U.swap(1, 2)
            U.swap(0, 1)
            U.swap(2, 3)
        elif a == 14:
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
    U = U.to_gate()
    U.name = "%i^%i mod 15" % (a, power)
    c_U = U.control()
    return c_U

def qft_dagger(n):
    qc = QuantumCircuit(n)
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
    qc.name = "QFT†"
    return qc

# 15를 소인수 분해할 숫자
a = 7
qc = qpe_amod15(a)
qc.draw('mpl')

# 시뮬레이터 백엔드 설정
simulator = Aer.get_backend('qasm_simulator')

# 양자 회로 컴파일 및 실행
compiled_circuit = transpile(qc, simulator)
qobj = assemble(compiled_circuit)
result = simulator.run(qobj).result()

# 결과 시각화
counts = result.get_counts()
plot_histogram(counts)
```

### 준비 자료
- **강의 자료**: 쇼어 알고리즘의 개념과 원리 슬라이드 (PDF)
- **참고 코드**: 쇼어 알고리즘 구현 예제 코드 (Python)

### 과제
- **이론 정리**: 쇼어 알고리즘의 개념과 주요 응용 분야 요약.
- **코드 실습**: 제공된 쇼어 알고리즘 코드를 실행하고, 다른 입력값으로 실습.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 쇼어 알고리즘의 개념과 원리를 이해하고, 주요 응용 분야를 학습하며, 실제 쇼어 알고리즘을 구현하는 경험을 쌓을 수 있도록 유도합니다.

---

