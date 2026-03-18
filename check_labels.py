import cv2
import random
import os

def check_labels():
    split_file = 'data/train_split.txt'
    if not os.path.exists(split_file):
        print(f"[!] 파일이 없습니다: {split_file}")
        return
        
    with open(split_file, 'r', encoding='utf-8') as f:
        # 빈 줄과 줄바꿈 문자 제거
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    if len(lines) == 0:
        print("[!] 훈련 데이터 경로 파일이 비어있습니다.")
        return
        
    # 라벨 파일(.txt)이 실제로 존재하는 이미지만 필터링 (배경 이미지 8063장 제외)
    labeled_lines = []
    for line in lines:
        label_path = line.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        if os.path.exists(label_path):
            labeled_lines.append(line)
            
    if len(labeled_lines) == 0:
        print("[!] 라벨 파일이 존재하는 이미지가 하나도 없습니다.")
        return
        
    # 랜덤으로 3장 뽑기 (이제는 무조건 소가 있는 타겟 이미지만 뽑힙니다)
    sampled_lines = random.sample(labeled_lines, min(3, len(labeled_lines)))
    
    # 결과를 저장할 폴더 생성
    out_dir = "label_checks"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[*] 뽑힌 3장의 이미지 시각화를 시작합니다. 결과는 '{out_dir}/' 폴더에 저장됩니다.")
    
    for idx, img_path in enumerate(sampled_lines):
        # 1. 이미지 읽기
        img = cv2.imread(img_path)
        if img is None:
            print(f"[!] 이미지를 읽을 수 없습니다: {img_path}")
            continue
            
        # 이미지 크기(가로, 세로) 가져오기
        h, w, _ = img.shape
        
        # 2. 이미지 경로를 바탕으로 라벨(.txt) 경로 유추
        # 예: data/images/train/img1.jpg -> data/labels/train/img1.txt
        label_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        
        # 3. 라벨 읽고 그리기
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = f.readlines()
                
            for label in labels:
                parts = label.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_c = float(parts[1])  # 정규화된 x 중심
                    y_c = float(parts[2])  # 정규화된 y 중심
                    box_w = float(parts[3]) # 정규화된 너비
                    box_h = float(parts[4]) # 정규화된 높이
                    
                    # YOLO 정규화 좌표를 실제 이미지 픽셀 좌표로 변환
                    x1 = int((x_c - box_w / 2) * w)
                    y1 = int((y_c - box_h / 2) * h)
                    x2 = int((x_c + box_w / 2) * w)
                    y2 = int((y_c + box_h / 2) * h)
                    
                    # 4. 박스 그리기
                    # BGR 기준 형광 연두색(Neon Green) = (0, 255, 0), 두께 2
                    color = (0, 255, 0)
                    thickness = 2
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                    
                    # 클래스 ID 텍스트 삽입 (보기 쉽도록)
                    text_color = (0, 255, 255) # 형광 노란색(Cyan-yellow)
                    cv2.putText(img, f"Class {cls_id}", (x1, max(y1-10, 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        else:
            print(f"[!] 매칭되는 라벨 파일이 없습니다: {label_path}")
            
        # 결과 이미지 저장
        out_path = os.path.join(out_dir, f"check_{idx + 1}.jpg")
        cv2.imwrite(out_path, img)
        print(f"  -> 저장 완료: {out_path} (원본 이미지: {os.path.basename(img_path)})")

if __name__ == '__main__':
    check_labels()
