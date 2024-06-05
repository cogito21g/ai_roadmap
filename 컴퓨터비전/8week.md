### 8주차 강의 상세 계획: 이미지 스타일 변환

#### 강의 목표
- Neural Style Transfer (NST)의 기본 개념과 원리 이해
- PyTorch를 이용한 Neural Style Transfer 구현 및 실습

#### 강의 구성
- **이론 강의**: 1시간
- **코드 구현 실습**: 1시간

---

### 1. 이론 강의 (1시간)

#### 1.1 Neural Style Transfer (NST)의 기본 개념 (20분)

##### Neural Style Transfer란?
- **정의**: 한 이미지의 스타일을 다른 이미지에 적용하여 새로운 이미지를 생성하는 기법.
- **주요 개념**:
  - **콘텐츠 이미지**: 스타일이 적용될 대상 이미지.
  - **스타일 이미지**: 콘텐츠 이미지에 적용될 스타일을 제공하는 이미지.
  - **결과 이미지**: 콘텐츠 이미지의 구조를 유지하면서 스타일 이미지의 스타일을 적용한 이미지.

##### Neural Style Transfer의 주요 응용 분야
- **예술**: 다양한 예술 스타일을 적용한 이미지 생성.
- **디자인**: 제품 디자인, 패션 디자인에서의 스타일 변환.
- **엔터테인먼트**: 영화, 게임에서의 스타일 변환.

#### 1.2 Neural Style Transfer의 원리 (40분)

##### VGG 네트워크
- **VGG 네트워크**: 스타일 변환을 위해 사전 학습된 VGG19 네트워크 사용.
- **레이어 선택**: 콘텐츠와 스타일을 추출하기 위해 특정 레이어 선택.

##### 손실 함수
- **콘텐츠 손실**: 결과 이미지와 콘텐츠 이미지의 차이를 최소화.
  - \( L_{content} = \sum (F_{content} - F_{generated})^2 \)
- **스타일 손실**: 결과 이미지와 스타일 이미지의 스타일 차이를 최소화.
  - 그램 행렬(Gram Matrix)을 사용하여 스타일 표현.
  - \( L_{style} = \sum (G_{style} - G_{generated})^2 \)

##### 학습 과정
- **입력 이미지**: 초기화된 결과 이미지.
- **최적화**: 결과 이미지의 픽셀 값을 조정하여 콘텐츠 손실과 스타일 손실을 최소화.

---

### 2. 코드 구현 실습 (1시간)

#### 2.1 PyTorch를 이용한 Neural Style Transfer 구현

##### 필요 라이브러리 설치
```bash
pip install torch torchvision matplotlib
```

##### Neural Style Transfer 구현 코드 (Python 3.10 및 PyTorch)
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt

# 이미지 로드 및 전처리 함수
def load_image(img_path, transform=None, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')
    if max_size:
        scale = max_size / max(image.size)
        size = tuple((np.array(image.size) * scale).astype(int))
        image = image.resize(size, Image.ANTIALIAS)
    if shape:
        image = image.resize(shape, Image.ANTIALIAS)
    if transform:
        image = transform(image).unsqueeze(0)
    return image

# 이미지 저장 함수
def save_image(image, path):
    image = image.clone().squeeze()
    image = unloader(image)
    image.save(path)

# 하이퍼파라미터 설정
image_size = 512 if torch.cuda.is_available() else 128
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 콘텐츠 이미지와 스타일 이미지 로드
content_img = load_image('path_to_your_content_image.jpg', transform)
style_img = load_image('path_to_your_style_image.jpg', transform, shape=[content_img.size(2), content_img.size(3)])

# 결과 이미지를 위한 텐서 생성
generated_img = content_img.clone().requires_grad_(True)

# VGG 모델 로드
model = models.vgg19(pretrained=True).features
for param in model.parameters():
    param.requires_grad_(False)

# 모델을 GPU로 이동
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
content_img = content_img.to(device)
style_img = style_img.to(device)
generated_img = generated_img.to(device)

# 스타일 손실 계산을 위한 그램 행렬 함수
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# 콘텐츠 레이어와 스타일 레이어 정의
content_layers = ['conv4_2']
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
content_weight = 1e3
style_weight = 1e-2

# 모델에서 콘텐츠와 스타일 특성 추출 함수
def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

# 콘텐츠와 스타일 특성 추출
content_features = get_features(content_img, model, content_layers)
style_features = get_features(style_img, model, style_layers)

# 스타일 그램 행렬 계산
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# 옵티마이저 설정
optimizer = optim.Adam([generated_img], lr=0.003)
criterion = nn.MSELoss()

# 학습
num_steps = 2000
for step in range(num_steps):
    generated_features = get_features(generated_img, model, content_layers + style_layers)
    
    # 콘텐츠 손실 계산
    content_loss = criterion(generated_features[content_layers[0]], content_features[content_layers[0]])
    
    # 스타일 손실 계산
    style_loss = 0
    for layer in style_layers:
        generated_feature = generated_features[layer]
        _, d, h, w = generated_feature.size()
        generated_gram = gram_matrix(generated_feature)
        style_gram = style_grams[layer]
        style_loss += criterion(generated_gram, style_gram)
    style_loss *= style_weight / len(style_layers)
    
    # 총 손실 계산
    total_loss = content_weight * content_loss + style_loss
    
    # 역전파 및 최적화
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if step % 50 == 0:
        print(f'Step {step}, Total loss: {total_loss.item()}')
        save_image(generated_img, 'generated_image.jpg')

# 최종 결과 이미지 저장 및 시각화
save_image(generated_img, 'final_generated_image.jpg')
generated_image = Image.open('final_generated_image.jpg')
plt.imshow(generated_image)
plt.title('Generated Image')
plt.show()
```

### 준비 자료
- **강의 자료**: Neural Style Transfer (NST) 개요 슬라이드 (PDF)
- **참고 코드**: NST 구현 예제 코드 (Python)

### 과제
- **이론 정리**: Neural Style Transfer의 기본 개념과 학습 과정 정리.
- **코드 실습**: NST 구현 코드 실행 및 결과 분석.
- **과제 제출**: 다음 주차 강의 전까지 이메일로 제출.

이 강의 계획안을 통해 학생들이 Neural Style Transfer의 기본 개념과 원리를 이해하고, PyTorch를 이용하여 NST를 구현하며, 실제 데이터를 통해 스타일 변환 결과를 생성하고 분석하는 경험을 쌓을 수 있도록 유도합니다.