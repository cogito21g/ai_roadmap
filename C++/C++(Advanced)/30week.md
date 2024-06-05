### 30주차: 클라우드 컴퓨팅

#### 강의 목표
- 클라우드 서비스 소개 (AWS, Azure, GCP)
- 클라우드 인프라 설정 이해 및 사용
- 클라우드 상에서의 애플리케이션 배포 및 관리 이해 및 사용

#### 강의 내용

##### 1. 클라우드 서비스 소개
- **Amazon Web Services (AWS)**

```markdown
AWS는 아마존이 제공하는 클라우드 컴퓨팅 서비스입니다. 주요 서비스로는 EC2(가상 서버), S3(저장소), RDS(데이터베이스) 등이 있습니다.
```

- **Microsoft Azure**

```markdown
Azure는 마이크로소프트가 제공하는 클라우드 컴퓨팅 서비스입니다. 주요 서비스로는 Virtual Machines, Blob Storage, SQL Database 등이 있습니다.
```

- **Google Cloud Platform (GCP)**

```markdown
GCP는 구글이 제공하는 클라우드 컴퓨팅 서비스입니다. 주요 서비스로는 Compute Engine, Cloud Storage, BigQuery 등이 있습니다.
```

##### 2. 클라우드 인프라 설정
- **AWS EC2 인스턴스 설정**

```markdown
1. AWS Management Console에 로그인합니다.
2. EC2 대시보드로 이동합니다.
3. "Launch Instance" 버튼을 클릭합니다.
4. AMI(Amazon Machine Image)를 선택합니다.
5. 인스턴스 유형을 선택합니다.
6. 키 페어를 생성하거나 기존 키 페어를 선택합니다.
7. 인스턴스를 시작합니다.
```

- **Azure Virtual Machines 설정**

```markdown
1. Azure Portal에 로그인합니다.
2. Virtual Machines 메뉴로 이동합니다.
3. "Add" 버튼을 클릭합니다.
4. 이미지와 크기를 선택합니다.
5. 인증 방식(SSH 키 또는 비밀번호)을 설정합니다.
6. 가상 머신을 생성합니다.
```

- **GCP Compute Engine 설정**

```markdown
1. GCP Console에 로그인합니다.
2. Compute Engine 메뉴로 이동합니다.
3. "Create Instance" 버튼을 클릭합니다.
4. 머신 유형과 이미지를 선택합니다.
5. 인스턴스를 생성합니다.
```

##### 3. 클라우드 상에서의 애플리케이션 배포 및 관리
- **AWS Elastic Beanstalk을 사용한 애플리케이션 배포**

```markdown
1. AWS Management Console에 로그인합니다.
2. Elastic Beanstalk 서비스로 이동합니다.
3. "Create New Application" 버튼을 클릭합니다.
4. 애플리케이션 이름과 환경을 설정합니다.
5. 플랫폼과 버전을 선택합니다.
6. 코드 패키지를 업로드하고 배포합니다.
```

- **Azure App Services를 사용한 애플리케이션 배포**

```markdown
1. Azure Portal에 로그인합니다.
2. App Services 메뉴로 이동합니다.
3. "Create" 버튼을 클릭합니다.
4. 애플리케이션 이름과 리소스 그룹을 설정합니다.
5. 배포할 코드를 업로드하고 배포합니다.
```

- **GCP App Engine을 사용한 애플리케이션 배포**

```markdown
1. GCP Console에 로그인합니다.
2. App Engine 메뉴로 이동합니다.
3. "Create Application" 버튼을 클릭합니다.
4. 언어와 런타임 환경을 선택합니다.
5. 코드를 업로드하고 배포합니다.
```

#### 과제

1. **AWS EC2 인스턴스를 생성하고 웹 서버를 설정하세요.**

```markdown
1. AWS Management Console에 로그인합니다.
2. EC2 대시보드로 이동합니다.
3. "Launch Instance" 버튼을 클릭합니다.
4. Ubuntu AMI를 선택합니다.
5. t2.micro 인스턴스 유형을 선택합니다.
6. 키 페어를 생성하거나 기존 키 페어를 선택합니다.
7. 인스턴스를 시작합니다.
8. SSH를 통해 인스턴스에 접속합니다.
9. Apache 웹 서버를 설치하고 실행합니다.
```

2. **Azure Virtual Machines를 사용하여 웹 서버를 설정하세요.**

```markdown
1. Azure Portal에 로그인합니다.
2. Virtual Machines 메뉴로 이동합니다.
3. "Add" 버튼을 클릭합니다.
4. Ubuntu 이미지를 선택합니다.
5. B1s 인스턴스 크기를 선택합니다.
6. 인증 방식(SSH 키 또는 비밀번호)을 설정합니다.
7. 가상 머신을 생성합니다.
8. SSH를 통해 가상 머신에 접속합니다.
9. Nginx 웹 서버를 설치하고 실행합니다.
```

3. **GCP Compute Engine을 사용하여 웹 서버를 설정하세요.**

```markdown
1. GCP Console에 로그인합니다.
2. Compute Engine 메뉴로 이동합니다.
3. "Create Instance" 버튼을 클릭합니다.
4. Ubuntu 이미지를 선택합니다.
5. f1-micro 머신 유형을 선택합니다.
6. 인스턴스를 생성합니다.
7. SSH를 통해 인스턴스에 접속합니다.
8. Apache 웹 서버를 설치하고 실행합니다.
```

4. **AWS Elastic Beanstalk을 사용하여 간단한 Node.js 애플리케이션을 배포하세요.**

```javascript
// app.js
const express = require('express');
const app = express();
const port = process.env.PORT || 3000;

app.get('/', (req, res) => {
    res.send('Hello, Elastic Beanstalk!');
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});

// 배포 절차
1. AWS Management Console에 로그인합니다.
2. Elastic Beanstalk 서비스로 이동합니다.
3. "Create New Application" 버튼을 클릭합니다.
4. 애플리케이션 이름과 환경을 설정합니다.
5. 플랫폼으로 Node.js를 선택합니다.
6. 코드를 zip 파일로 업로드하고 배포합니다.
```

#### 퀴즈

1. **AWS의 가상 서버 서비스는 무엇인가?**
   - A) S3
   - B) RDS
   - C) EC2
   - D) Lambda

2. **Azure에서 가상 머신을 생성할 때 사용하는 포털은?**
   - A) AWS Management Console
   - B) GCP Console
   - C) Azure Portal
   - D) IBM Cloud Dashboard

3. **GCP에서 가상 머신 인스턴스를 생성할 때 사용하는 서비스는?**
   - A) Compute Engine
   - B) App Engine
   - C) Cloud Functions
   - D) BigQuery

4. **AWS Elastic Beanstalk을 사용하여 배포할 수 있는 애플리케이션 종류가 아닌 것은?**
   - A) Node.js
   - B) Python
   - C) Ruby
   - D) Adobe Flash

#### 퀴즈 해설

1. **AWS의 가상 서버 서비스는 무엇인가?**
   - **정답: C) EC2**
     - 해설: AWS의 가상 서버 서비스는 EC2 (Elastic Compute Cloud)입니다.

2. **Azure에서 가상 머신을 생성할 때 사용하는 포털은?**
   - **정답: C) Azure Portal**
     - 해설: Azure에서 가상 머신을 생성할 때는 Azure Portal을 사용합니다.

3. **GCP에서 가상 머신 인스턴스를 생성할 때 사용하는 서비스는?**
   - **정답: A) Compute Engine**
     - 해설: GCP에서 가상 머신 인스턴스를 생성할 때는 Compute Engine을 사용합니다.

4. **AWS Elastic Beanstalk을 사용하여 배포할 수 있는 애플리케이션 종류가 아닌 것은?**
   - **정답: D) Adobe Flash**
     - 해설: AWS Elastic Beanstalk은 Node.js, Python, Ruby 등 다양한 애플리케이션을 배포할 수 있지만, Adobe Flash는 지원하지 않습니다.

이로써 30주차 클라우드 컴퓨팅 과정을 마쳤습니다. 추가로 학습하고 싶은 주제나 더 알고 싶은 내용이 있다면 언제든지 말씀해 주세요.