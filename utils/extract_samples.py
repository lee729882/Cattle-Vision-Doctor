import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.append(project_root)
os.chdir(project_root)
import os
import random
import shutil

def extract_samples():
    # ?Όλ²¨ ?μΌ??λ¬΄μ΅°κ±?μ΅΄μ¬?λ” κ±Έλ΅ κ²€μ¦λ train_split.txt?μ„ λ½‘μµ?λ‹¤.
    split_file = 'data/train_split.txt'
    
    if not os.path.exists(split_file):
        print(f"[!] ?μΌ??μ°Ύμ„ ???†μµ?λ‹¤: {split_file}")
        return
        
    with open(split_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    # 5??λ¬΄μ‘??μ¶”μ¶ (λ¦¬μ¤?Έκ? 5κ°?λ―Έλ§?΄λ©΄ ?„μ²΄ μ¶”μ¶)
    sample_count = min(5, len(lines))
    sampled_lines = random.sample(lines, sample_count)
    
    # κΉƒν—λΈ?κ³µμ ???ν” ?΄λ” κµ¬μ΅° ?μ„± (YOLO ?¬λ§· ? μ?)
    sample_img_dir = os.path.join('samples', 'images')
    sample_lbl_dir = os.path.join('samples', 'labels')
    os.makedirs(sample_img_dir, exist_ok=True)
    os.makedirs(sample_lbl_dir, exist_ok=True)
    
    print(f"[*] κΉƒν—λΈ?κ³µμ ???ν” ({sample_count}?? λ³µμ‚¬λ¥??μ‘?©λ‹??..")
    
    for img_path in sampled_lines:
        img_name = os.path.basename(img_path)
        # ?΄λ?μ§€???€?‘λ???Όλ²¨ κ²½λ΅ ? μ¶”
        lbl_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        lbl_name = os.path.basename(lbl_path)
        
        dest_img = os.path.join(sample_img_dir, img_name)
        dest_lbl = os.path.join(sample_lbl_dir, lbl_name)
        
        # ?μΌ λ³µμ‚¬ μ§„ν–‰
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            shutil.copy2(img_path, dest_img)
            shutil.copy2(lbl_path, dest_lbl)
            print(f"  -> λ³µμ‚¬ ?„λ£: {img_name} λ°?{lbl_name}")
        else:
            print(f"  -> λ³µμ‚¬ ?¤ν¨ (?μΌ ?„λ½): {img_name}")
            
    print("\n[*] ?±κ³µ! 5?ΈνΈ???΄λ?μ§€?€ ?Όλ²¨??'samples/' ?΄λ”???„λ²½?κ² μ¤€λΉ„λ?μµ?λ‹¤.")
    print("[*] (μ°Έκ³ : .gitignore ?¤μ •???ν•΄ data/ ?΄λ”??λ¬΄μ‹?μ?λ§? samples/ ?΄λ”???λ™?Όλ΅ κΉƒν—λΈμ— ?¨κ» ?…λ΅?λ©?λ‹¤!)")

if __name__ == '__main__':
    extract_samples()

