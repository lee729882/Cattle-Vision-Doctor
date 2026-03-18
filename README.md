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
  - 추출된 고품질 소(Cattle) 라벨링 데이터를 활용하여 베이스라인 모델(`best.pt`) 구축.
  - 교차 검증(Validation) 및 테스트 수행 결과, 우수한 손실률(Loss) 감소와 함께 **최상위 mAP(정확도) 지표 달성 완료**.
- [ ] 오염 물질(Contamination) 클래스 라벨링 및 전이 학습 (진행 예정)
- [ ] 화각 이탈 및 왜곡 탐지 알고리즘 로직 구현 (진행 예정)

## 🚀 시작하기 (Getting Started)

### 1. 환경 설정 (Environment Setup)
이 프로젝트를 실행하려면 가상환경을 활성화하고 필수 패키지를 설치해야 합니다.
```bash
# 가상환경 활성화 (Conda)
conda activate cattle-env

# 패키지 설치
pip install ultralytics torch torchvision opencv-python
```

### 2. 추론 테스트 실행 (Run Inference / Prediction)
학습된 모델(`best.pt`)이 준비되면, 깃허브에 동봉된 `samples/` 폴더 안의 이미지들로 AI가 소를 얼마나 잘 인식하는지 직접 테스트해 볼 수 있습니다!

터미널에 아래 명령어를 입력하여 추론(Predict) 스크립트를 실행합니다:
```bash
# AI 추론 예측(Predict) 실행
python test_inference.py
```
*(실행이 완료되면, 소의 위치에 초록색 바운딩 박스와 AI의 확신도(Confidence Score)가 적힌 결과 사진들이 자동으로 `runs/val_results/` 폴더에 저장됩니다.)*

---
© 2026 Vision Doctor Project. All Rights Reserved.