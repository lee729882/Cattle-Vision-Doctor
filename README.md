# 🐄 Vision Doctor
> **축사 CCTV 오염 탐지 및 화각 복구 시스템** <br>
> Livestock CCTV Contamination Detection and Angle Recovery System

<p align="left">
  <img src="https://img.shields.io/badge/YOLOv11-00FFFF?style=flat-square&logo=YOLO&logoColor=black"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=OpenCV&logoColor=white"/>
</p>

## 📋 프로젝트 소개 (About the Project)
**Vision Doctor**는 축사 환경에서 발생하는 CCTV 렌즈의 오염(거미줄, 먼지, 분변 등)을 자동으로 탐지하고, 화각의 변형을 감지하여 알림 및 복구 기준을 제공하는 스마트 비전 AI 시스템입니다. 

## 🛠 기술 스택 (Tech Stack)
- **Deep Learning Model**: YOLOv11 (Nano - `yolo11n.pt`)
- **Framework**: PyTorch
- **Computer Vision**: OpenCV (Python)

## 📈 현재 진행 상황 (Current Progress)
- [x] **Cattle(소) 탐지 베이스라인 모델 학습 완료**
  - 추출된 고품질 소 라벨링 데이터를 활용하여 베이스라인 모델 구축.
- [x] **오염 물질(Contamination) 클래스 학습 및 검증 완료**
  - YOLOv11n 기반 전이 학습 결과, 전 클래스에 대해 **mAP50 0.91 (91%)** 라는 매우 뛰어난 정확도를 달성했습니다.
  - **데이터 구축 전략**: 실제 환경의 오염 데이터 부족 문제를 해결하기 위해, 프로그래밍 방식으로 투명 거미줄 마스크를 생성하고 이를 정상 데이터에 합성하여 5:5 비율의 고품질 밸런스 데이터셋을 직접 구축하는 공학적 접근법을 적용했습니다.
- [ ] 화각 이탈 및 왜곡 탐지 알고리즘 로직 구현 (진행 예정)

## 📂 프로젝트 구조 (Project Structure)
> **전문가 수준(Expert Level)으로 최적화된 폴더 구조입니다.**
```text
Cattle-Vision Doctor/
├── models/
│   └── best_cow_contamination.pt    # [최종 가중치] 소 ID 및 오염 탐지 통합 마스터 모델
├── output/                          # [추론 결과물] 학습 성능이 증명된 최종 프레젠테이션용 데모 사진 모음
├── utils/                           # [스크립트] 데이터 증강, 분할, 커스텀 검증 등 파이썬 도구 모음
├── data/
│   ├── samples/                     # 샘플용 합성 데이터 이미지
│   └── data.yaml                    # YOLO 학습 환경 설정 파일
└── runs/                            # 학습 및 검증 로그 (TensorBoard 호환)
```

## 🚀 시작하기 (Getting Started)

### 1. 환경 설정 (Environment Setup)
이 프로젝트를 실행하려면 가상환경을 활성화하고 필수 패키지를 설치해야 합니다.
```bash
# 가상환경 활성화 (Conda)
conda activate cattle-env

# 패키지 설치
pip install ultralytics torch torchvision opencv-python tensorboard
```

### 2. 추론 테스트 실행 (Run Inference / Prediction)
터미널에 아래 명령어를 입력하여 완성된 마스터 모델로 추론을 실행해 볼 수 있습니다:
```bash
python utils/test_inference.py
```

---
© 2026 Vision Doctor Project. All Rights Reserved.