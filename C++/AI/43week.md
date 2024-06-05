### 43주차: 추천 시스템 (Recommender Systems) 기초

#### 강의 목표
- 추천 시스템의 기본 개념 이해
- 협업 필터링과 콘텐츠 기반 필터링 이해
- 간단한 추천 시스템 구현

#### 강의 내용

##### 1. 추천 시스템의 기본 개념
- **추천 시스템 개요**
  - 정의: 사용자에게 맞춤형 아이템을 추천하는 시스템
  - 주요 응용 분야: 전자상거래, 스트리밍 서비스, 뉴스 제공

- **추천 시스템의 유형**
  - 협업 필터링 (Collaborative Filtering)
  - 콘텐츠 기반 필터링 (Content-Based Filtering)
  - 하이브리드 필터링 (Hybrid Filtering)

##### 2. 협업 필터링
- **사용자 기반 협업 필터링 (User-Based Collaborative Filtering)**
  - 설명: 유사한 취향을 가진 사용자가 선호하는 아이템을 추천
  - 구현 예제:

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 사용자-아이템 행렬 생성
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 4],
    'item_id': [1, 2, 3, 2, 3, 1, 3, 2],
    'rating': [5, 4, 3, 4, 2, 3, 5, 4]
}
df = pd.DataFrame(data)

# 사용자-아이템 매트릭스 변환
user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)

# 사용자 유사도 계산
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# 특정 사용자의 아이템 추천
def recommend_items(user_id, user_item_matrix, user_similarity_df, top_n=2):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:top_n+1]
    similar_users_ratings = user_item_matrix.loc[similar_users].mean(axis=0)
    user_ratings = user_item_matrix.loc[user_id]
    recommendations = similar_users_ratings[user_ratings == 0].sort_values(ascending=False)
    return recommendations

recommendations = recommend_items(1, user_item_matrix, user_similarity_df)
print(recommendations)
```

- **아이템 기반 협업 필터링 (Item-Based Collaborative Filtering)**
  - 설명: 유사한 아이템을 사용자가 선호하는 경우 해당 아이템을 추천
  - 구현 예제:

```python
# 아이템 유사도 계산
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# 특정 아이템의 유사 아이템 추천
def recommend_similar_items(item_id, item_similarity_df, top_n=2):
    similar_items = item_similarity_df[item_id].sort_values(ascending=False).index[1:top_n+1]
    return similar_items

similar_items = recommend_similar_items(1, item_similarity_df)
print(similar_items)
```

##### 3. 콘텐츠 기반 필터링
- **콘텐츠 기반 필터링 개요**
  - 설명: 아이템의 속성 정보를 사용하여 유사한 아이템을 추천

- **콘텐츠 기반 필터링 구현 예제**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 아이템 데이터 생성
item_data = {
    'item_id': [1, 2, 3, 4],
    'description': [
        'This is a great action movie',
        'A thrilling adventure of a hero',
        'Romantic comedy that will make you laugh',
        'An epic tale of love and war'
    ]
}
items_df = pd.DataFrame(item_data)

# TF-IDF 벡터화
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(items_df['description'])

# 아이템 유사도 계산
item_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
item_similarity_df = pd.DataFrame(item_similarity, index=items_df['item_id'], columns=items_df['item_id'])

# 특정 아이템의 유사 아이템 추천
def recommend_content_based_items(item_id, item_similarity_df, top_n=2):
    similar_items = item_similarity_df[item_id].sort_values(ascending=False).index[1:top_n+1]
    return similar_items

content_based_recommendations = recommend_content_based_items(1, item_similarity_df)
print(content_based_recommendations)
```

#### 과제

1. **사용자 기반 협업 필터링 구현**
   - 주어진 사용자-아이템 행렬을 사용하여 사용자 기반 협업 필터링을 구현하고, 특정 사용자에게 아이템을 추천합니다.

2. **아이템 기반 협업 필터링 구현**
   - 주어진 사용자-아이템 행렬을 사용하여 아이템 기반 협업 필터링을 구현하고, 특정 아이템과 유사한 아이템을 추천합니다.

3. **콘텐츠 기반 필터링 구현**
   - 주어진 아이템 데이터를 사용하여 콘텐츠 기반 필터링을 구현하고, 특정 아이템과 유사한 아이템을 추천합니다.

#### 퀴즈

1. **추천 시스템의 주요 유형이 아닌 것은?**
   - A) 협업 필터링
   - B) 콘텐츠 기반 필터링
   - C) 하이브리드 필터링
   - D) 지도 학습 필터링

2. **협업 필터링에서 사용자 기반 협업 필터링의 주요 특징은 무엇인가?**
   - A) 유사한 속성을 가진 아이템을 추천
   - B) 유사한 취향을 가진 사용자가 선호하는 아이템을 추천
   - C) 아이템의 속성 정보를 사용하여 추천
   - D) 사용자 행동 데이터를 사용하지 않음

3. **콘텐츠 기반 필터링의 주요 특징은 무엇인가?**
   - A) 유사한 취향을 가진 사용자가 선호하는 아이템을 추천
   - B) 사용자 행동 데이터를 사용하여 추천
   - C) 아이템의 속성 정보를 사용하여 추천
   - D) 협업 필터링과 결합하여 추천

4. **TF-IDF의 주요 목적은 무엇인가?**
   - A) 단어의 빈도를 계산하기 위해
   - B) 단어의 중요도를 계산하기 위해
   - C) 텍스트를 요약하기 위해
   - D) 텍스트의 차원을 축소하기 위해

#### 퀴즈 해설

1. **추천 시스템의 주요 유형이 아닌 것은?**
   - **정답: D) 지도 학습 필터링**
     - 해설: 추천 시스템의 주요 유형에는 협업 필터링, 콘텐츠 기반 필터링, 하이브리드 필터링이 있으며, 지도 학습 필터링은 추천 시스템의 주요 유형이 아닙니다.

2. **협업 필터링에서 사용자 기반 협업 필터링의 주요 특징은 무엇인가?**
   - **정답: B) 유사한 취향을 가진 사용자가 선호하는 아이템을 추천**
     - 해설: 사용자 기반 협업 필터링은 유사한 취향을 가진 사용자가 선호하는 아이템을 추천하는 방법입니다.

3. **콘텐츠 기반 필터링의 주요 특징은 무엇인가?**
   - **정답: C) 아이템의 속성 정보를 사용하여 추천**
     - 해설: 콘텐츠 기반 필터링은 아이템의 속성 정보를 사용하여 유사한 아이템을 추천하는 방법입니다.

4. **TF-IDF의 주요 목적은 무엇인가?**
   - **정답: B) 단어의 중요도를 계산하기 위해**
     - 해설: TF-IDF는 문서에서 단어의 중요도를 계산하는 방법입니다.

다음 주차 강의 내용을 요청하시면, 44주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.