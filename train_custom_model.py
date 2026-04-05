import os
import torch
from ultralytics import YOLO

def main():
    print("=" * 50)
    print("🔪 Vision-Doctor: Custom Weapon YOLO Training Script")
    print("=" * 50)
    
    # 데이터셋 YAML 파일 경로 설정
    dataset_yaml = os.path.abspath("weapon_dataset/data.yaml")
    
    # 1. 환경 및 경로 검사
    if not os.path.exists(dataset_yaml):
        print(f"\n[오류] 데이터셋 파일을 찾을 수 없습니다: {dataset_yaml}")
        print("Roboflow 등에서 데이터셋을 다운로드한 후, 폴더명을 'weapon_dataset'으로 변경하여")
        print("새로운 'datasets' 폴더 안에 넣어주세요.")
        print("최종 경로 예시: c:\\Users\\kim11\\test4\\Vision-Doctor\\datasets\\weapon_dataset\\data.yaml\n")
        return

    # CUDA(GPU) 사용 가능 여부 확인
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"[정보] 학습에 사용할 연산 장치: {'GPU (CUDA)' if device == 0 else 'CPU (시간이 오래 걸리므로 가급적 GPU를 권장합니다)'}")
    
    # 2. 모델 로드 (가장 가성비 좋고 무기 감지에 유리한 yolov8s.pt 바탕으로 학습)
    print("\n[시스템] 사전 학습된 모델(yolov8s.pt)을 불러옵니다...")
    model = YOLO("yolov8s.pt")  
    
    # 3. 모델 학습 시작
    print("\n🚀本格的に モデル学習を開始します！(이 작업은 사양에 따라 몇십 분 ~ 몇 시간이 소요될 수 있습니다)")
    try:
        results = model.train(
            data=dataset_yaml,        # 학습 데이터셋 설정 파일 경로
            epochs=50,                # 전체 데이터셋을 반복 학습할 횟수 (데이터셋 양에 따라 30~100 사이 권장)
            imgsz=640,                # 이미지 리사이징 크기
            batch=16,                 # 한번에 처리할 이미지 수 (Out Of Memory 에러 발생 시 8로 낮추세요)
            device=device,            # GPU / CPU 선택
            project="runs/detect",    # 학습 결과 저장 최상위 폴더
            name="baseline_test",     # 세션명 (webcam_inference.py 에서 이 이름으로 모델을 찾습니다)
            exist_ok=True             # 기존 폴더가 있으면 덮어쓰기 허용
        )
        print("\n✅ 모델 학습이 완벽하게 완료되었습니다!")
        print("최적의 결과물(AI 뇌)은 다음 경로에 안전하게 저장되었습니다:")
        print("-> runs/detect/baseline_test/weights/best.pt")
        print("\n이제 launcher.py를 통해 CCTV 기능을 다시 실행하시면 새로 학습된 '나만의 보안 모델'이 자동 적용됩니다!")
        
    except Exception as e:
        print(f"\n[에러 발생] 학습 중 다음 문제가 발생했습니다:\n{e}")

if __name__ == "__main__":
    # Windows 환경에서 PyTorch 멀티프로세싱 충돌 방지
    import multiprocessing
    multiprocessing.freeze_support()
    main()
