Git은 버전 관리 시스템으로, 소프트웨어 개발 프로젝트의 변경 사항을 추적하고 여러 명의 개발자가 협업할 수 있게 해줍니다. 기본적인 Git 사용법을 단계별로 설명하겠습니다.

### 1. Git 설치
먼저 Git을 설치해야 합니다. [Git 공식 웹사이트](https://git-scm.com/)에서 운영체제에 맞는 설치 파일을 다운로드하고 설치하세요.

### 2. Git 설정
설치 후, 사용자 정보를 설정합니다.
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 3. Git 리포지토리 초기화
새로운 Git 리포지토리를 생성하거나 기존 리포지토리를 클론할 수 있습니다.

#### 새로운 리포지토리 생성
```bash
mkdir my_project
cd my_project
git init
```

#### 기존 리포지토리 클론
```bash
git clone https://github.com/username/repository.git
cd repository
```

### 4. 파일 추가 및 커밋
파일을 추가하고 커밋하여 변경 사항을 저장할 수 있습니다.

#### 파일 상태 확인
```bash
git status
```

#### 파일 추가
```bash
git add filename  # 특정 파일 추가
git add .         # 모든 변경된 파일 추가
```

#### 커밋
```bash
git commit -m "Commit message"
```

### 5. 원격 리포지토리 설정
원격 리포지토리를 설정하고, 로컬 변경 사항을 푸시할 수 있습니다.

#### 원격 리포지토리 추가
```bash
git remote add origin https://github.com/username/repository.git
```

#### 푸시
```bash
git push origin master
```

### 6. 브랜치 사용
브랜치를 사용하여 독립된 작업 공간을 만들 수 있습니다.

#### 브랜치 생성 및 이동
```bash
git checkout -b new-branch
```

#### 브랜치 목록 보기
```bash
git branch
```

#### 브랜치 이동
```bash
git checkout branch-name
```

#### 브랜치 병합
```bash
git checkout master
git merge new-branch
```

### 7. 변경 사항 가져오기
원격 리포지토리의 변경 사항을 로컬 리포지토리에 가져올 수 있습니다.

#### 풀
```bash
git pull origin master
```

### 8. 로그 확인
커밋 로그를 확인할 수 있습니다.

```bash
git log
```

### 9. 파일 무시
`.gitignore` 파일을 사용하여 Git이 특정 파일이나 디렉토리를 무시하도록 설정할 수 있습니다.

#### .gitignore 예시
```plaintext
# Python
__pycache__/
*.pyc

# JavaScript
node_modules/

# Logs
*.log

# OS-specific
.DS_Store
```

### 10. 충돌 해결
다른 브랜치와 병합할 때 충돌이 발생할 수 있습니다. 충돌을 해결하고 다시 커밋하면 됩니다.

1. 충돌 파일 수정
2. 수정된 파일 추가
3. 커밋

```bash
git add filename
git commit
```

이 기본적인 Git 사용법을 통해 버전 관리를 효과적으로 수행할 수 있습니다. 더 자세한 내용은 [Git 공식 문서](https://git-scm.com/doc)에서 확인할 수 있습니다.