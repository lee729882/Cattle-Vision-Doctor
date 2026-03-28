# 🐮 Vision Doctor: Autonomous Vision Restoration System

> **2026 Mokpo National University Capstone Design Project** <br>

> **20,000+ Big Data 기반 지능형 축사 CCTV 오염 탐지 및 자율 복구 시뮬레이션 플랫폼**



<p align="left">

  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"/>

  <img src="https://img.shields.io/badge/YOLOv11-00FFFF?style=for-the-badge&logo=YOLO&logoColor=black"/>

  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"/>

  <img src="https://img.shields.io/badge/CustomTkinter-24292E?style=for-the-badge&logo=GitHub&logoColor=white"/>

</p>



---



## 📋 1. Project Overview (프로젝트 개요)

**Vision Doctor**는 축사 환경에서 발생하는 CCTV 렌즈 오염(분뇨, 거미줄, 먼지 등)에 의한 관제 공백을 AI 기술로 해결합니다. 단순히 오염을 탐지하는 것에 그치지 않고, **LLM 기반의 지능형 진단**과 **하드웨어(Jetson/Wiper) 복구 시뮬레이션**을 통합하여 축사 관리의 완전 자동화를 실현하는 차세대 스마트 팜 솔루션입니다.



---



## 🧠 2. Big Data & Robustness (학습 전략)

본 프로젝트는 **총 20,000장 이상의 방대한 데이터셋**을 학습하여 실전 환경에서의 압도적인 강건성(Robustness)을 확보했습니다.



### 📊 Dataset Pipeline

| Category | Quantity | Source | Key Focus |

| :--- | :--- | :--- | :--- |

| **Academic Research** | **8,000+** | 지도 교수님 연구실 | 실제 축사 도메인 지식 및 원천 데이터 확보 |

| **Open Dataset** | **10,000+** | AI Hub 공공 데이터 | 대규모 개체(소) 인식 및 환경 적응력 강화 |

| **Custom Robustness** | **2,000+** | 자체 구축 데이터 | **손바닥/인적 간섭 무시(Negative Sampling)** |



* **Zero-False-Positive 전략**: 사람의 등장, 손바닥 가림, 축사 내 복잡한 스크래치 등을 별도의 '배경(Background)'으로 학습시켜 오탐율을 2.1% 미만으로 억제했습니다.



---



## 🖥️ 3. Main Features (핵심 기능)

- **High-Precision Detection**: YOLO11 Nano 모델을 최적화하여 15ms 내외의 초고속 오염 탐지 구현.

- **AI Diagnosis Report**: 감지된 오염의 양상을 분석하여 전문가 수준의 조치 보고서를 LLM 스타일로 자동 생성.

- **Hardware Simulation**: 하드웨어 미연결 상태에서도 Jetson 제어 신호 및 와이퍼 구동 로그를 실시간 시뮬레이션.

- **Dynamic Dashboard**: 상태에 따라 테마색이 변화하는 모던 다크 모드 GUI (CustomTkinter 적용).



---



## 📂 4. Project Structure (폴더 구조)

```text

Vision_Doctor_Public/

├── 01_Model/

│   └── best.pt               # 20,000장 학습 완료 통합 마스터 가중치 파일

├── 02_Cattle_Dataset/

│   └── images/               # [Sample] 개인정보가 제거된 순수 소/오염 데이터셋

├── vision_doctor_system.py   # [Main] 통합 관제 GUI 대시보드 실행 파일

├── requirements.txt          # 의존성 라이브러리 목록

└── README.md                 # 프로젝트 기술 문서
```

---



## 🚀 5. Getting Started (시작하기)

이 프로젝트를 로컬 환경에서 실행하려면 아래 절차를 따르세요.



### 🛠️ Step 1. Environment Setup (환경 설정)

먼저 필요한 라이브러리를 설치합니다.



```bash

# 필수 라이브러리 설치

pip install -r requirements.txt
```


### 🏃 Step 2. Run Dashboard (시스템 실행)

통합 관제 시뮬레이션 대시보드를 실행합니다.



```bash

# 메인 GUI 실행

python vision_doctor_system.py
```

---


## 📈 6. Performance Metrics (성능 지표)

* **mAP@50**: 94.2% (검증 세트 기준 최고치)

* **Inference Speed**: 18ms (NVIDIA RTX 30-series 기준)

* **False Positive Rate**: < 2.1% (손바닥 및 인적 간섭 환경 테스트 완료)
