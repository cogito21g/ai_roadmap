### 23주차: 가상화 및 컨테이너 기술

**강의 목표:** 가상화와 컨테이너 기술의 개념을 이해하고, 하이퍼바이저와 Docker를 사용하여 가상 환경을 설정하고 관리하는 방법을 학습합니다.

**강의 내용:**

1. **가상화 개념**
   - 가상화란 무엇인가?
     - 물리적 하드웨어 자원을 추상화하여 여러 가상 컴퓨터를 생성하는 기술
   - 가상화의 장점
     - 자원 효율성, 비용 절감, 유연성 증가
   - 가상화 유형
     - 하드웨어 가상화, 운영체제 수준 가상화

2. **하이퍼바이저**
   - 하이퍼바이저란 무엇인가?
     - 가상화를 지원하는 소프트웨어 레이어
   - 하이퍼바이저의 종류
     - Type 1 하이퍼바이저 (네이티브 하이퍼바이저)
     - Type 2 하이퍼바이저 (호스트형 하이퍼바이저)
   - 주요 하이퍼바이저 예시
     - VMware ESXi, Microsoft Hyper-V, KVM, VirtualBox

3. **컨테이너 기술 (Docker)**
   - 컨테이너란 무엇인가?
     - 애플리케이션과 그 종속성을 격리하여 패키징하고 실행하는 기술
   - 컨테이너와 가상 머신의 차이점
     - 경량화, 빠른 시작 시간, 효율적인 자원 사용
   - Docker 소개
     - Docker의 개념과 구성 요소 (이미지, 컨테이너, Dockerfile, Docker Hub)
   - Docker 설치 및 기본 사용법
     - Docker 설치, 이미지 생성 및 관리, 컨테이너 실행

**실습:**

1. **하이퍼바이저 설치 및 가상 머신 설정**

**VirtualBox를 사용한 가상 머신 설정:**

```sh
# VirtualBox 설치 (Linux 예시)
sudo apt update
sudo apt install virtualbox

# 가상 머신 생성 및 설정
VBoxManage createvm --name "MyVM" --register
VBoxManage modifyvm "MyVM" --memory 2048 --acpi on --boot1 dvd --nic1 bridged
VBoxManage createhd --filename "MyVM.vdi" --size 20000
VBoxManage storagectl "MyVM" --name "SATA Controller" --add sata --controller IntelAhci
VBoxManage storageattach "MyVM" --storagectl "SATA Controller" --port 0 --device 0 --type hdd --medium "MyVM.vdi"
VBoxManage storagectl "MyVM" --name "IDE Controller" --add ide
VBoxManage storageattach "MyVM" --storagectl "IDE Controller" --port 0 --device 0 --type dvddrive --medium /path/to/iso
VBoxManage startvm "MyVM"
```

2. **Docker 설치 및 기본 사용법**

**Docker 설치 (Linux 예시):**

```sh
# Docker 설치
sudo apt update
sudo apt install docker.io

# Docker 서비스 시작 및 활성화
sudo systemctl start docker
sudo systemctl enable docker
```

**Docker 기본 명령어:**

```sh
# Docker 버전 확인
docker --version

# Docker 이미지 다운로드
docker pull ubuntu

# Docker 이미지 목록 확인
docker images

# Docker 컨테이너 실행
docker run -it ubuntu /bin/bash

# 실행 중인 컨테이너 목록 확인
docker ps

# 모든 컨테이너 목록 확인
docker ps -a

# 컨테이너 중지
docker stop <컨테이너 ID>

# 컨테이너 삭제
docker rm <컨테이너 ID>

# Docker 이미지 삭제
docker rmi <이미지 ID>
```

**Dockerfile을 사용한 이미지 생성:**

**Dockerfile 작성 예제:**

```Dockerfile
# 베이스 이미지 설정
FROM ubuntu:latest

# 메타데이터 설정
LABEL maintainer="yourname@example.com"
LABEL version="1.0"
LABEL description="A simple Dockerfile example"

# 패키지 업데이트 및 설치
RUN apt-get update && apt-get install -y \
    curl \
    vim

# 컨테이너 실행 시 기본 명령어
CMD ["bash"]
```

**Docker 이미지 빌드 및 실행:**

```sh
# Docker 이미지 빌드
docker build -t myubuntu:1.0 .

# Docker 이미지 목록 확인
docker images

# Docker 컨테이너 실행
docker run -it myubuntu:1.0
```

**과제:**

1. **고급 Docker 사용법**
   - Docker Compose를 사용하여 다중 컨테이너 애플리케이션 설정 및 관리
   - Docker Volume과 Network를 사용하여 컨테이너 간 데이터 공유 및 통신 설정

2. **Dockerfile 작성 및 최적화**
   - Dockerfile을 작성하여 웹 서버(Nginx 또는 Apache)를 설정하고, 최적화된 이미지 생성
   - 이미지를 빌드하고, 컨테이너를 실행하여 웹 서버가 정상적으로 동작하는지 확인

**퀴즈 및 해설:**

1. **가상화의 주요 장점은 무엇인가요?**
   - 가상화의 주요 장점은 자원 효율성, 비용 절감, 유연성 증가입니다. 가상화는 여러 가상 컴퓨터를 단일 물리적 하드웨어에서 실행할 수 있어 자원 활용도를 극대화하고, 하드웨어 비용을 절감할 수 있습니다.

2. **컨테이너와 가상 머신의 차이점은 무엇인가요?**
   - 컨테이너는 애플리케이션과 그 종속성을 격리하여 패키징하고 실행하는 기술로, 경량화되어 있으며 빠른 시작 시간을 갖습니다. 반면, 가상 머신은 전체 운영체제를 포함하여 실행되므로 더 많은 자원을 사용하고 시작 시간이 더 깁니다.

3. **Dockerfile의 역할은 무엇인가요?**
   - Dockerfile은 Docker 이미지를 빌드하기 위한 명령어와 설정을 포함한 텍스트 파일입니다. Dockerfile을 사용하여 애플리케이션 환경을 정의하고, 일관된 이미지 빌드를 자동화할 수 있습니다.

**해설:**

1. **가상화의 주요 장점**은 자원 효율성, 비용 절감, 유연성 증가입니다. 가상화는 여러 가상 컴퓨터를 단일 물리적 하드웨어에서 실행할 수 있어 자원 활용도를 극대화하고, 하드웨어 비용을 절감할 수 있습니다. 또한, 가상 환경을 쉽게 생성, 제거 및 관리할 수 있어 유연성이 증가합니다.

2. **컨테이너와 가상 머신의 차이점**은 컨테이너는 애플리케이션과 그 종속성을 격리하여 패키징하고 실행하는 기술로, 경량화되어 있으며 빠른 시작 시간을 갖는 반면, 가상 머신은 전체 운영체제를 포함하여 실행되므로 더 많은 자원을 사용하고 시작 시간이 더 깁니다. 컨테이너는 운영체제 수준에서 격리되고, 가상 머신은 하드웨어 수준에서 격리됩니다.

3. **Dockerfile의 역할**은 Docker 이미지를 빌드하기 위한 명령어와 설정을 포함한 텍스트 파일입니다. Dockerfile을 사용하여 애플리케이션 환경을 정의하고, 일관된 이미지 빌드를 자동화할 수 있습니다. Dockerfile은 베이스 이미지 설정, 패키지 설치, 환경 변수 설정 등 다양한 명령어를 포함할 수 있습니다.

이로써 23주차 강의를 마무리합니다. 다음 주차에는 클라우드 컴퓨팅 및 DevOps에 대해 학습합니다. 추가 질문이 있거나 특정 주제에 대해 더 알고 싶다면 언제든지 문의해 주세요.