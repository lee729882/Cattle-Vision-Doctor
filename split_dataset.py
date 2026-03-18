import random
import yaml
import os

def main():
    # 1. 기존 데이터셋(train.txt) 경로 읽기
    txt_path = 'data/train.txt'
    if not os.path.exists(txt_path):
        print(f"[!] {txt_path} 가 존재하지 않습니다. 스크립트 실행 위치를 확인해주세요 (c:\Cattle-Vision Doctor).")
        return
        
    with open(txt_path, 'r', encoding='utf-8') as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]
        
    # 🌟 라벨 파일(.txt)이 실제로 존재하는 타겟 이미지만 필터링 (배경 제거) 🌟
    lines = []
    for line in raw_lines:
        label_path = line.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        if os.path.exists(label_path):
            lines.append(line + '\n')
            
    print(f"\n[*] 원본 {len(raw_lines)}장 중에서 라벨이 존재하는 {len(lines)}장만 추출완료!")
    
    if len(lines) == 0:
        print("[!] 에러: 유효한 라벨 파일이 전혀 없습니다.")
        return
    
    # 리스트 셔플 (랜덤성 부여)
    random.seed(42)  # 재현성을 위한 시드 고정
    random.shuffle(lines)
    
    # 8:2 비율 계산
    split_idx = int(len(lines) * 0.8)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
    
    # 2. 결과 파일 저장
    with open('data/train_split.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open('data/val_split.txt', 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
        
    print(f"[*] 데이터 분할 완료: Train {len(train_lines)}장, Val {len(val_lines)}장")

    # 3. data.yaml 업데이트
    yaml_path = 'data/data.yaml'
    with open(yaml_path, 'r', encoding='utf-8') as f:
        # PyYAML로 안전하게 파싱합니다.
        data = yaml.safe_load(f)
        
    # 'names' 딕셔너리 업데이트
    if 'names' in data:
        # 기존 80개 클래스 (0~79번 인덱스)
        # 80번 인덱스(81번째 아이템)로 'contamination' 추가
        data['names'][80] = 'contamination'
            
    # 전체 클래스 개수(nc) 81로 명시
    data['nc'] = 81
    
    # train/val 경로를 분할된 텍스트 파일로 업데이트
    data['train'] = 'data/train_split.txt'
    data['val'] = 'data/val_split.txt'
    
    # 위 항목들을 다시 data.yaml에 덮어 쓰기
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        
    print("[*] data.yaml 파일 업데이트가 성공적으로 완료되었습니다!")

if __name__ == '__main__':
    main()
