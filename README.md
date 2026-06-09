# AI-project 

## 1. 프로젝트 개요

### 주제 
YOLO11 기반 도로 파손 부위(Pothole, Crack, Manhole) 탐지 및 데이터 증강 성능 비교 연구
Road Damage Detection and Augmentation Analysis using YOLO11

### 문제 정의
* 도로 표면의 **포트홀(Pothole)**, **균열(Crack)** 및 관리되지 않은 **맨홀(Manhole)** 영역은 차량 파손 및 심각한 대형 교통사고를 유발하는 주된 위험. 
* 기존의 인력 중심 수동 도로 점검 방식은 시간과 비용 소모가 크고, 실시간 노면 변화에 신속하게 대응하기 어렵다는 근본적인 한계가 존재.

### 연구 목적 및 개발 목적
* 최신 실시간 객체 탐지 알고리즘인 `YOLO11 (Nano & Small)` 모델을 활용하여 주행 중 도로 파손 부위를 정확히 분류하고 탐지하는 자동화 시스템을 구축하고자 합
* 데이터 전처리(Crop) 및 가혹 주행 환경을 모사한 **데이터 증강(Data Augmentation)** 기법이 객체 탐지 성능 지표(`mAP50`, `mAP50-95`) 개선에 미치는 영향력을 체계적으로 비교 분석하여 최적의 파이프라인을 제안

### 프로젝트 파이프라인 아키텍처


## 2. 필요한 라이브러리 및 환경
본 프로젝트는 구글 코랩(Google Colab - Tesla T4 GPU) 환경에서 `Ultralytics` 라이브러리를 기반으로 구현함
* **핵심 프레임워크:** `Ultralytics YOLO11`
* **의존성 라이브러리:** `Torch`, `OpenCV-Python`, `PyYAML`, `Matplotlib`

### 개발 환경 구축 및 의존성 설치
구글 코랩 환경 필수 패키지 및 라이브러리 일괄 설치
pip install ultralytics python-dotenv opencv-python

## 3.데이터 세트
데이터셋 가공 및 분할 구조
원본 데이터 가공: 도로 전경 데이터에서 이미지 품질을 높이기 위해 파손 의심 영역을 정밀하게 크롭(Crop) 및 라벨링하여 데이터 완성도를 고도화
데이터셋 분할: 원본 데이터 총 2,009장을 Train(80%) : Validation(20%) 비율로 분할하여 독립적인 검증 환경을 구축

Train Dataset: 1,607장 (이미지 및 라벨 텍스트 매칭 완료)
Validation Dataset: 402장

데이터 구성 정의
path: /content/drive/MyDrive/프로젝트 4조/Road/dataset
train: images/train
val: images/val

nc: 3
names: ['Pothole', 'Crack', 'Manhole']

### 수립된 데이터셋 로컬 디렉토리 구조

```text
/content/drive/MyDrive/프로젝트 4조/Road/dataset/
├── data.yaml
├── images/
│   ├── train/ (1,607장)
│   └── val/   (402장)
└── labels/
    ├── train/ (1,607개 .txt)
    └── val/   (402개 .txt)
```

## 4. 모델 설명 및 개발 내용
1차 실험 [Baseline]:가장 가벼운 YOLO11n (Nano)모델을 기반으로 하는 분석 처리 없이 기본 전 처리 데이터셋만 사용하여 50 Epochs 모델 학습을 연구하고 기준 지표를 제공

python
from ultralytics import YOLO

Baseline: 전처리(Crop) 데이터셋 기반 YOLO11n 기본 학습
model_1 = YOLO("yolo11n.yaml")
model_1.train(
    data="data.yaml", 
    epochs=50, 
    batch=16, 
    imgsz=640, 
    name="yolo11n_baseline"
)

2차 실험 [Augmentation]:약간의 상황이 발생할 수 있는 노면의 조도 및 스윙을 모사하기 위해 YOLO11n모델에 광학적 변형(명도 hsv_v=0.6, 채도 hsv_s=0.5) 및 각도 변형(회전 degrees=10.0, 전위 이동 translate=0.2) 가능성을 적용하여 50 Epochs 확장을 진행

### Augmentation: 조도 변화 및 가혹 주행 환경 모사를 위한 증강 파라미터 적용
model_2 = YOLO("yolo11n.yaml")
model_2.train(
    data="data.yaml", 
    epochs=50, 
    batch=16, 
    imgsz=640, 
    hsv_v=0.6,       # 명도 변형 확장
    hsv_s=0.5,       # 채도 변형 확장
    degrees=10.0,    # 미세 회전 강제
    translate=0.2,   # 평행 이동 변형
    name="yolo11n_augmentation"
    
3차 실험 [Scale-up]:복합적인 환경과 이온 처리 성능을 극대화하기 위해 사용자가 더 확장된 YOLO11s (Small)모델로 규모를 확장하고, 하이퍼파라미터 최적화와 함께 100 Epochs의 장기적 성능을 수행

비디오 인퍼런스 및 추적(Tracking) 알고리즘
연속적인 프레임 환경에서의 탐지 누락을 방지하기 위해 학습 완료된 가중치 파일(best.pt)을 활용하여 실제 도로 주행 영상(맑은날.mp4)에 인퍼런스를 연동
프레임 연속성을 고려한 ByteTrack 추적 알고리즘을 결합하여 동일 파손 영역의 중복 카운트를 방지하고 추적 안정성을 확보

### ByteTrack 알고리즘 연동 실시간 주행 영상 객체 추적 추론
best_model = YOLO("/content/drive/MyDrive/프로젝트 4조/Road/weights/best.pt")
results = best_model.track(
    source="/content/drive/MyDrive/프로젝트 4조/맑은날.mp4", 
    show=True, 
    tracker="bytetrack.yaml"
)

## 5. 실험 결과 및 분석
결과 요약: 2차 실험(YOLO11n + 기본 증강) 환경에서는 에포크 제한으로 일시적인 지표 둔화가 관찰되었으나, **모델 구조를 확장하고 학습 에포크를 100회로 고도화한 3차 실험(YOLO11s)**이 모든 지표에서 가장 뛰어난 성능을 기록

### 모델별 최종 평가 메트릭 비교 테이블

| 실험 조건 (Model Architecture) | Epochs | Batch | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1차: YOLO11n (Baseline)** | 50 | 16 | 0.518 | 0.467 | 0.468 | 0.201 |
| **2차: YOLO11n (+Augmentation)** | 50 | 16 | 0.503 | 0.439 | 0.437 | 0.186 |
| **3차: YOLO11s (Scale-up)** | **100** | **16** | **0.547** | **0.494** | **0.492** | **0.217** |

## 6. 한계점 및 향후 계획 (Limitations & Future Work)

### 연구 한계점
* **기상 및 주행 환경의 편향성 (Environmental Constraints):** 현재 파이프라인의 검증 인퍼런스는 '맑은 날' 비디오 스트림만을 기준으로 테스트되었습니다. 따라서 흐린 날, 우천 시, 혹은 야간 주행 상황과 같이 조도가 급격히 저하되거나 노면 반사가 심한 가혹 환경에서의 탐지 신뢰성을 보장하기 위해 추가적인 데이터 확보와 검증이 필요합니다.
* **엣지 디바이스 연산 오버헤드 (Edge Device Deployment):** YOLO11s 모델 스케일업을 통해 객체 탐지 정밀도(mAP)는 크게 향상시켰으나, 실제 차량용 블랙박스나 저전력 임베디드 엣지 디바이스(Edge Device)에 탑재하여 실시간(Real-time) 환경으로 구동하기에는 여전히 연산 자원 최적화 측면의 과제가 남아있습니다.

### 향후 연구 계획
* **도메인 적응 및 전처리 모듈 도입 (Domain Adaptation & GAN):** 조도 변화 및 기상 악화 환경에 유연하게 대응할 수 있도록 환경 변화를 극복하는 **도메인 적응(Domain Adaptation)** 기법을 연구할 예정입니다. 더불어, GAN이나 오토인코더(Autoencoder)를 활용해 노면의 노이즈와 흔들림을 상쇄하는 전처리 모듈을 파이프라인에 추가하고자 합니다.
* **모델 경량화 및 고속 가속화 (Model Compression):** 차량 내장형 내비게이션이나 임베디드 장비 환경에서 실시간 제어가 가능하도록, 학습된 모델 가중치를 엔비디아(NVIDIA) 가속 포맷인 **TensorRT**로 변환하고 **양자화(Quantization, INT8/FP16)** 최적화 공정을 적용하는 경량화 파이프라인 연구를 이어갈 계획입니다.
