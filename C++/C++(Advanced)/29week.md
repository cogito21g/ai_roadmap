### 29주차: 데이터베이스 관리

#### 강의 목표
- SQL 기초 이해 및 사용
- 고급 SQL 및 최적화 이해 및 사용
- NoSQL 데이터베이스 (MongoDB, Redis 등) 이해 및 사용

#### 강의 내용

##### 1. SQL 기초
- **데이터베이스와 테이블 생성**

```sql
-- 데이터베이스 생성
CREATE DATABASE mydatabase;

-- 데이터베이스 사용
USE mydatabase;

-- 테이블 생성
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    age INT
);
```

- **데이터 삽입, 조회, 수정, 삭제**

```sql
-- 데이터 삽입
INSERT INTO users (name, age) VALUES ('Alice', 30), ('Bob', 25);

-- 데이터 조회
SELECT * FROM users;

-- 데이터 수정
UPDATE users SET age = 26 WHERE name = 'Bob';

-- 데이터 삭제
DELETE FROM users WHERE name = 'Alice';
```

##### 2. 고급 SQL 및 최적화
- **조인 (JOIN)과 서브쿼리 (Subquery)**

```sql
-- 조인 예제
SELECT orders.id, customers.name, orders.amount
FROM orders
INNER JOIN customers ON orders.customer_id = customers.id;

-- 서브쿼리 예제
SELECT name FROM users
WHERE age = (SELECT MAX(age) FROM users);
```

- **인덱스와 성능 최적화**

```sql
-- 인덱스 생성
CREATE INDEX idx_name ON users (name);

-- 인덱스 사용 조회
EXPLAIN SELECT * FROM users WHERE name = 'Bob';
```

- **트랜잭션과 잠금**

```sql
-- 트랜잭션 시작
START TRANSACTION;

-- 여러 작업 실행
UPDATE accounts SET balance = balance - 100 WHERE name = 'Alice';
UPDATE accounts SET balance = balance + 100 WHERE name = 'Bob';

-- 트랜잭션 커밋
COMMIT;

-- 트랜잭션 롤백
ROLLBACK;
```

##### 3. NoSQL 데이터베이스
- **MongoDB 기본 사용법**

```javascript
// MongoDB 데이터베이스와 컬렉션 생성
use mydatabase;
db.createCollection("users");

// 데이터 삽입
db.users.insertMany([
    { name: "Alice", age: 30 },
    { name: "Bob", age: 25 }
]);

// 데이터 조회
db.users.find();

// 데이터 수정
db.users.updateOne({ name: "Bob" }, { $set: { age: 26 } });

// 데이터 삭제
db.users.deleteOne({ name: "Alice" });
```

- **Redis 기본 사용법**

```bash
# Redis 서버 시작
redis-server

# Redis 클라이언트 접속
redis-cli

# 문자열 데이터 저장 및 조회
SET user:1 "Alice"
GET user:1

# 해시 데이터 저장 및 조회
HSET user:2 name "Bob" age 25
HGETALL user:2
```

#### 과제

1. **SQL을 사용하여 데이터베이스를 생성하고, 사용자 정보를 포함한 테이블을 작성하세요. 데이터를 삽입하고 조회하는 SQL 쿼리를 작성하세요.**

```sql
-- 데이터베이스 생성 및 사용
CREATE DATABASE mydatabase;
USE mydatabase;

-- 테이블 생성
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    age INT
);

-- 데이터 삽입
INSERT INTO users (name, age) VALUES ('Alice', 30), ('Bob', 25);

-- 데이터 조회
SELECT * FROM users;
```

2. **고급 SQL 쿼리를 작성하여 사용자 테이블에서 나이가 가장 많은 사용자를 조회하세요.**

```sql
-- 최대 나이 사용자 조회
SELECT name FROM users
WHERE age = (SELECT MAX(age) FROM users);
```

3. **MongoDB를 사용하여 사용자 정보를 저장하고 조회하는 JavaScript 코드를 작성하세요.**

```javascript
// MongoDB 데이터베이스와 컬렉션 생성
use mydatabase;
db.createCollection("users");

// 데이터 삽입
db.users.insertMany([
    { name: "Alice", age: 30 },
    { name: "Bob", age: 25 }
]);

// 데이터 조회
db.users.find();
```

4. **Redis를 사용하여 사용자 정보를 저장하고 조회하는 명령어를 작성하세요.**

```bash
# Redis 클라이언트 접속
redis-cli

# 문자열 데이터 저장 및 조회
SET user:1 "Alice"
GET user:1

# 해시 데이터 저장 및 조회
HSET user:2 name "Bob" age 25
HGETALL user:2
```

#### 퀴즈

1. **SQL에서 테이블을 생성할 때 기본 키를 설정하는 방법은?**
   - A) PRIMARY KEY(name)
   - B) KEY(name)
   - C) PRIMARY(name)
   - D) AUTO_INCREMENT(name)

2. **SQL 조인에서 두 테이블을 결합하는 키워드는?**
   - A) JOIN
   - B) UNION
   - C) CONNECT
   - D) LINK

3. **MongoDB에서 데이터를 조회할 때 사용하는 메서드는?**
   - A) find()
   - B) search()
   - C) get()
   - D) fetch()

4. **Redis에서 문자열 데이터를 저장하는 명령어는?**
   - A) PUT
   - B) SET
   - C) STORE
   - D) SAVE

#### 퀴즈 해설

1. **SQL에서 테이블을 생성할 때 기본 키를 설정하는 방법은?**
   - **정답: A) PRIMARY KEY(name)**
     - 해설: 기본 키를 설정할 때는 PRIMARY KEY 키워드를 사용합니다.

2. **SQL 조인에서 두 테이블을 결합하는 키워드는?**
   - **정답: A) JOIN**
     - 해설: SQL에서 조인을 수행할 때는 JOIN 키워드를 사용합니다.

3. **MongoDB에서 데이터를 조회할 때 사용하는 메서드는?**
   - **정답: A) find()**
     - 해설: MongoDB에서 데이터를 조회할 때는 find() 메서드를 사용합니다.

4. **Redis에서 문자열 데이터를 저장하는 명령어는?**
   - **정답: B) SET**
     - 해설: Redis에서 문자열 데이터를 저장할 때는 SET 명령어를 사용합니다.

다음 주차 강의 내용을 요청하시면, 30주차 강의 내용, 과제 및 퀴즈를 상세히 제공해드리겠습니다.