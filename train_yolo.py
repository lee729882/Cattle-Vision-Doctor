import torch
from ultralytics import YOLO

def main():
    # 1. 모델 로드 (YOLO11 nano 모델을 기본으로 사용)
    # 다른 크기의 모델이 필요하다면 'yolo11s.pt', 'yolo11m.pt' 등으로 변경하세요.
    model = YOLO("yolo11n.pt")  

    # 2. GPU(CUDA) 사용 가능 여부 확인
    # GPU가 사용 가능하면 '0'번 장치를, 아니면 'cpu'를 사용하도록 자동 설정합니다.
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[*] 학습에 사용할 장치(Device): {'GPU (CUDA)' if device == '0' else 'CPU'}")

    # 3. 모델 학습 설정 및 시작
    results = model.train(
        data="data/data.yaml",  # 클래스 매핑 및 데이터셋 경로가 포함된 yaml 파일
        epochs=20,              # 베이스라인 테스트용 20 Epoch
        imgsz=640,              # 이미지 크기 (기본값: 640)
        device=device,          # 자동 할당된 장치 사용
        project="runs/detect",  # 결과를 저장할 최상위 디렉토리
        name="baseline_test",   # 저장될 하위 폴더 이름 (runs/detect/baseline_test/)
        exist_ok=True,          # 덮어쓰기 허용 (또는 False로 두면 baseline_test2, 3 자동 생성)
        # batch=16,             # GPU VRAM이 부족하면 배치 사이즈를 낮춰주세요 (예: 8, 4)
    )

    print("[*] 학습이 완료되었습니다!")
    print("[*] 학습 결과, Weights 파일 및 차트들은 'runs/yolov11_train' 폴더를 확인해주세요.")

if __name__ == '__main__':
    # Windows 환경에서 PyTorch DataLoader의 다중 프로세싱(Workers) 충돌을 방지하려면
    # 반드시 if __name__ == '__main__': 아래에서 실행 코드를 구현해야 합니다.
    main()
