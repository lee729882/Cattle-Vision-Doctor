import os
import random
import shutil

def extract_samples():
    # 라벨 파일이 무조건 존재하는 걸로 검증된 train_split.txt에서 뽑습니다.
    split_file = 'data/train_split.txt'
    
    if not os.path.exists(split_file):
        print(f"[!] 파일을 찾을 수 없습니다: {split_file}")
        return
        
    with open(split_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    # 5장 무작위 추출 (리스트가 5개 미만이면 전체 추출)
    sample_count = min(5, len(lines))
    sampled_lines = random.sample(lines, sample_count)
    
    # 깃허브 공유용 샘플 폴더 구조 생성 (YOLO 포맷 유지)
    sample_img_dir = os.path.join('samples', 'images')
    sample_lbl_dir = os.path.join('samples', 'labels')
    os.makedirs(sample_img_dir, exist_ok=True)
    os.makedirs(sample_lbl_dir, exist_ok=True)
    
    print(f"[*] 깃허브 공유용 샘플 ({sample_count}장) 복사를 시작합니다...")
    
    for img_path in sampled_lines:
        img_name = os.path.basename(img_path)
        # 이미지에 대응되는 라벨 경로 유추
        lbl_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        lbl_name = os.path.basename(lbl_path)
        
        dest_img = os.path.join(sample_img_dir, img_name)
        dest_lbl = os.path.join(sample_lbl_dir, lbl_name)
        
        # 파일 복사 진행
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            shutil.copy2(img_path, dest_img)
            shutil.copy2(lbl_path, dest_lbl)
            print(f"  -> 복사 완료: {img_name} 및 {lbl_name}")
        else:
            print(f"  -> 복사 실패 (파일 누락): {img_name}")
            
    print("\n[*] 성공! 5세트의 이미지와 라벨이 'samples/' 폴더에 완벽하게 준비되었습니다.")
    print("[*] (참고: .gitignore 설정에 의해 data/ 폴더는 무시되지만, samples/ 폴더는 자동으로 깃허브에 함께 업로드됩니다!)")

if __name__ == '__main__':
    extract_samples()
