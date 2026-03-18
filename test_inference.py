from ultralytics import YOLO
import os

def run_inference():
    # 1. 학습이 완료된 베스트 모델 가중치 파일 로드 (YOLO의 자동 생성 경로 대응)
    weights_path = "runs/detect/runs/detect/baseline_test/weights/best.pt"
    
    # 예비 경로 확인
    if not os.path.exists(weights_path):
        weights_path = "runs/detect/baseline_test/weights/best.pt"
    
    if not os.path.exists(weights_path):
        print(f"[!] 아직 모델 폴더에 파일이 없습니다: {weights_path}")
        print("[!] 먼저 터미널에서 'python train_yolo.py'를 끝까지 기다려서 돌려주세요! 🚀")
        return

    # 베스트 모델 불러오기
    model = YOLO(weights_path)

    print("\n[*] 검증용 데이터(val_split.txt)로 Inference(추론)를 시작합니다...")
    
    # 2. val_split.txt 안에 있는 사진 1600여장을 가져와 AI 박스 예측 수행
    # YOLO는 사진 목록이 담긴 txt 파일 경로를 바로 넣어주어도 알아서 다 처리해줍니다.
    # 프로젝트 폴더를 'runs', 이름을 'val_results'로 주어 [runs/val_results] 안에 저장되게 합니다.
    results = model.predict(
        source="data/val_split.txt",  
        save=True,                    # 박스가 그려진 이미지 최종 저장 여부
        project="runs",               
        name="val_results",           # 실제 결과 폴더 이름 (runs/val_results/)
        conf=0.25,                    # 25% 이상 확신하는 물체만 그리기 (필요 시 조절)
        show_conf=True,               # 박스 위에 AI가 확신하는 백분율(Confidence Score) 점수 표시
        show_labels=True,             # 박스 위에 어느 클래스인지 이름/번호 표시
        exist_ok=True                 # 폴더 경로 덮어쓰기 허용
    )

    print("\n[*] Inference(추론)이 성공적으로 완료되었습니다! 🎨")
    print("[*] 텍스트가 덮어씌워진 화려한 결과 사진들은 'runs/val_results/' 폴더 안에서 확인하실 수 있습니다!")

if __name__ == '__main__':
    # Windows 멀티프로세싱 에러 방지
    run_inference()
