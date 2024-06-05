### 45주차: 시계열 분석 (Time Series Analysis)

#### 강의 목표
- 시계열 데이터의 기본 개념 이해
- 시계열 데이터 전처리 및 탐색적 데이터 분석
- 시계열 예측 모델 구현

#### 강의 내용

##### 1. 시계열 데이터의 기본 개념
- **시계열 데이터 개요**
  - 정의: 시간의 흐름에 따라 순차적으로 관측된 데이터
  - 주요 특성: 트렌드, 계절성, 순환성, 잔차

- **시계열 데이터의 주요 기법**
  - 이동 평균 (Moving Average)
  - 분해 (Decomposition)
  - 자기상관 함수 (Autocorrelation Function, ACF)
  - 부분 자기상관 함수 (Partial Autocorrelation Function, PACF)

##### 2. 시계열 데이터 전처리 및 탐색적 데이터 분석
- **데이터 로드 및 시각화**

```python
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_csv('path_to_your_time_series_data.csv', index_col='date', parse_dates=True)

# 데이터 시각화
df.plot(figsize=(10, 6))
plt.show()
```

- **이동 평균 및 분해**

```python
# 이동 평균 계산
df['moving_average'] = df['value'].rolling(window=12).mean()

# 이동 평균 시각화
df[['value', 'moving_average']].plot(figsize=(10, 6))
plt.show()

# 시계열 분해
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df['value'], model='additive', period=12)
result.plot()
plt.show()
```

- **자기상관 함수 (ACF) 및 부분 자기상관 함수 (PACF)**

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ACF 플롯
plot_acf(df['value'], lags=50)
plt.show()

# PACF 플롯
plot_pacf(df['value'], lags=50)
plt.show()
```

##### 3. 시계열 예측 모델 구현
- **ARIMA 모델**
  - 설명: 자기회귀 적분 이동 평균 (Autoregressive Integrated Moving Average) 모델
  - 구현 예제:

```python
from statsmodels.tsa.arima_model import ARIMA

# ARIMA 모델 학습
model = ARIMA(df['value'], order=(5, 1, 0))
model_fit = model.fit(disp=0)

# 모델 요약
print(model_fit.summary())

# 예측
forecast = model_fit.forecast(steps=12)
plt.plot(df['value'])
plt.plot(pd.date_range(start=df.index[-1], periods=12, freq='M'), forecast[0])
plt.show()
```

- **SARIMA 모델**
  - 설명: 계절성 자기회귀 적분 이동 평균 (Seasonal ARIMA) 모델
  - 구현 예제:

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMA 모델 학습
model = SARIMAX(df['value'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit(disp=False)

# 모델 요약
print(model_fit.summary())

# 예측
forecast = model_fit.get_forecast(steps=12)
forecast_ci = forecast.conf_int()

plt.plot(df['value'])
plt.plot(pd.date_range(start=df.index[-1], periods=12, freq='M'), forecast.predicted_mean, label='Forecast')
plt.fill_between(pd.date_range(start=df.index[-1], periods=12, freq='M'), forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='g', alpha=0.3)
plt.legend()
plt.show()
```

- **Prophet 모델**
  - 설명: Facebook에서 개발한 시계열 예측 모델
  - 구현 예제:

```python
from fbprophet import Prophet

# 데이터 준비
df_prophet = df.reset_index().rename(columns={'date': 'ds', 'value': 'y'})

# Prophet 모델 학습
model = Prophet()
model.fit(df_prophet)

# 예측
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

# 시각화
fig = model.plot(forecast)
plt.show()
```

#### 과제

1. **시계열 데이터 전처리 및 시각화**
   - 주어진 시계열 데이터를 로드하고, 이동 평균 및 분해를 통해 데이터의 트렌드와 계절성을 시각화합니다.

2. **ARIMA 모델 구현**
   - 주어진 시계열 데이터를 사용하여 ARIMA 모델을 구현하고, 모델을 학습 및 평가합니다.

3. **SARIMA 모델 구현**
   - 주어진 시계열 데이터를 사용하여 SARIMA 모델을 구현하고, 모델을 학습 및 평가합니다.

4. **Prophet 모델 구현**
   - 주어진 시계열 데이터를 사용하여 Prophet 모델을 구현하고, 모델을 학습 및 평가합니다.

#### 퀴즈

1. **시계열 데이터의 주요 특성이 아닌 것은?**
   - A) 트렌드
   - B) 계절성
   - C) 순환성
   - D) 분산성

2. **이동 평균의 주요 목적은 무엇인가?**
   - A) 데이터의 계절성을 분석하기 위해
   - B) 데이터의 추세를 부드럽게 만들기 위해
   - C) 데이터의 분산을 계산하기 위해
   - D) 데이터의 주기를 확인하기 위해

3. **ARIMA 모델에서 'I'는 무엇을 의미하는가?**
   - A) 자기회귀
   - B) 이동 평균
   - C) 적분
   - D) 계절성

4. **Prophet 모델의 주요 장점은 무엇인가?**
   - A) 단순한 구조로 인해 구현이 용이하다
   - B) 비계절성 데이터에 특화되어 있다
   - C) 자동화된 하이퍼파라미터 튜닝을 지원한다
   - D) 페이스북에서 개발되어 신뢰성이 높다

#### 퀴즈 해설

1. **시계열 데이터의 주요 특성이 아닌 것은?**
   - **정답: D) 분산성**
     - 해설: 시계열 데이터의 주요 특성에는 트렌드, 계절성, 순환성이 있으며, 분산성은 포함되지 않습니다.

2. **이동 평균의 주요 목적은 무엇인가?**
   - **정답: B) 데이터의 추세를 부드럽게 만들기 위해**
     - 해설: 이동 평균은 데이터의 추세를 부드럽게 만들어 시각적으로 더 쉽게 이해할 수 있도록 합니다.

3. **ARIMA 모델에서 'I'는 무엇을 의미하는가?**
   - **정답: C) 적분**
     - 해설: ARIMA 모델에서 'I'는 적분(Integration)을 의미하며, 데이터의 비정상성을 제거하기 위해 사용됩니다.

4. **Prophet 모델의 주요 장점은 무엇인가?**
   - **정답: C) 자동화된 하이퍼파라미터 튜닝을 지원한다**
     - 해설: Prophet 모델은 자동화된 하이퍼파라미터 튜닝을 지원하여 사용자가 쉽게 모델을 최적화할 수 있도록 합니다.

더 깊이 있는 학습이 필요하면, 구체적인 주제나 알고 싶은 내용을 요청해 주세요.