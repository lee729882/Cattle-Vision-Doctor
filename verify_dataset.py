import os

def check_split(split_file):
    if not os.path.exists(split_file):
        print(f"\n[!] 파일이 없습니다: {split_file}")
        return
        
    with open(split_file, 'r', encoding='utf-8') as f:
        # 빈 줄 제거
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    print(f"\n==============================================")
    print(f"[*] '{split_file}' 데이터셋 검증 (총 {len(lines)}장)")
    print(f"==============================================")
    
    missing_images = []
    missing_labels = []
    
    for img_path in lines:
        # 1. 실제 폴더에 해당 이미지가 존재하는지 확인
        if not os.path.exists(img_path):
            missing_images.append(img_path)
            continue # 이미지가 없으면 라벨 확인도 건너뜀
            
        # 2. 이미지에 대응되는 라벨(.txt) 경로 유추
        label_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        
        # 3. 라벨 파일(.txt)이 실제로 존재하는지 확인
        if not os.path.exists(label_path):
            # 라벨 파일이 없으면 배경 이미지(Background)로 간주됨
            missing_labels.append(label_path)
            
    print(f"  - 경로 상 실제 존재하는 이미지: {len(lines) - len(missing_images)}장")
    print(f"  - (에러) 누락되거나 경로가 틀린 이미지: {len(missing_images)}장")
    print(f"  - (경고) 라벨(.txt) 파일이 없는 이미지: {len(missing_labels)}장 (YOLO에 의해 빈 화면으로 학습/평가됨)")
    
    if missing_images:
        print("\n[!] 🚨 1. 경로가 존재하지 않는 이미지 리스트 (최대 10개만 출력):")
        for p in missing_images[:10]:
            print(f"    - {p}")
        if len(missing_images) > 10:
            print(f"    ... 외 {len(missing_images) - 10}개 더 있음")
            
    if missing_labels:
        print("\n[!] ⚠️ 2. 해당하는 라벨 파일(.txt)을 찾을 수 없는 경로 리스트 (최대 10개만 출력):")
        for p in missing_labels[:10]:
            print(f"    - {p}")
        if len(missing_labels) > 10:
            print(f"    ... 외 {len(missing_labels) - 10}개 더 있음")

if __name__ == '__main__':
    # 검증(val) 데이터와 학습(train) 데이터 모두 테스트합니다.
    check_split('data/val_split.txt')
    check_split('data/train_split.txt')
