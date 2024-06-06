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

----

효과적인 Git 커밋 메시지는 변경 사항을 명확하고 간결하게 설명하는 데 매우 중요합니다. 좋은 커밋 메시지 템플릿은 일관성을 유지하고, 나중에 변경 사항을 추적하는 데 도움이 됩니다. 아래는 일반적인 Git 커밋 메시지 템플릿을 제공합니다.

### Git 커밋 메시지 템플릿

```
<타입>: <제목>

<본문>

<푸터>
```

#### 타입 (Type)
커밋의 목적을 명확히 하기 위해 타입을 지정합니다. 일반적으로 사용하는 타입은 다음과 같습니다:

- **feat**: 새로운 기능 추가
- **fix**: 버그 수정
- **docs**: 문서 수정
- **style**: 코드 포맷팅, 세미콜론 누락 등 코드 변경이 없는 경우
- **refactor**: 코드 리팩토링
- **perf**: 성능 개선
- **test**: 테스트 추가나 수정
- **chore**: 빌드 과정 또는 보조 도구 수정
- **ci**: CI 설정 파일이나 스크립트 변경

#### 제목 (Subject)
- 제목은 변경 사항을 요약합니다.
- 50자 이내로 작성합니다.
- 첫 글자는 대문자로 작성합니다.
- 마침표나 특수문자로 끝내지 않습니다.

#### 본문 (Body)
- 변경 이유와 주요 변경 사항을 설명합니다.
- 72자마다 줄 바꿈을 합니다.
- 필요에 따라 여러 문단으로 나눌 수 있습니다.

#### 푸터 (Footer)
- 고친 이슈나 참고할 이슈 번호를 추가합니다.
- 예: `Fixes #123`, `Closes #456`

### 예시

#### 새로운 기능 추가
```
feat: Add user authentication

Added a new user authentication module using JWT. This allows users to securely log in and access protected routes. Updated the user model and added new routes for login and signup.

- Added JWT-based authentication
- Created login and signup routes
- Updated user model to store hashed passwords

Fixes #45
```

#### 버그 수정
```
fix: Correct header layout on mobile

Fixed the header layout issue on mobile devices by updating the CSS flex properties. The header now properly aligns and resizes on different screen sizes.

- Updated CSS flex properties for header
- Tested on various mobile devices

Closes #78
```

#### 문서 수정
```
docs: Update README with setup instructions

Added detailed setup instructions to the README file, including environment setup, installation steps, and how to run the project locally.

- Added environment setup section
- Detailed installation steps
- Instructions for running the project locally
```

#### 코드 스타일 수정
```
style: Reformat code according to style guide

Reformatted the entire codebase to adhere to the new style guide. This includes changes to indentation, spacing, and line length.

- Applied new indentation rules
- Adjusted spacing and line length
```

#### 코드 리팩토링
```
refactor: Simplify the user service logic

Refactored the user service to simplify the logic and improve readability. Removed redundant code and optimized database queries.

- Removed redundant code
- Optimized database queries
- Improved code readability
```

이 템플릿을 사용하여 커밋 메시지를 작성하면, 변경 사항을 명확하게 기록할 수 있으며, 팀원들과의 협업이 더욱 원활해집니다. 커밋 메시지는 프로젝트의 변경 이력을 추적하는 중요한 도구이므로, 일관되고 명확하게 작성하는 것이 좋습니다.