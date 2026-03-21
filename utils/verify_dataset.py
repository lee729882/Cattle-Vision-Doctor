import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.append(project_root)
os.chdir(project_root)
import os

def check_split(split_file):
    if not os.path.exists(split_file):
        print(f"\n[!] ?Œì¼???†ìŠµ?ˆë‹¤: {split_file}")
        return
        
    with open(split_file, 'r', encoding='utf-8') as f:
        # ë¹?ì¤??œê±°
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    print(f"\n==============================================")
    print(f"[*] '{split_file}' ?°ì´?°ì…‹ ê²€ì¦?(ì´?{len(lines)}??")
    print(f"==============================================")
    
    missing_images = []
    missing_labels = []
    
    for img_path in lines:
        # 1. ?¤ì œ ?´ë”???´ë‹¹ ?´ë?ì§€ê°€ ì¡´ì¬?˜ëŠ”ì§€ ?•ì¸
        if not os.path.exists(img_path):
            missing_images.append(img_path)
            continue # ?´ë?ì§€ê°€ ?†ìœ¼ë©??¼ë²¨ ?•ì¸??ê±´ë„ˆ?€
            
        # 2. ?´ë?ì§€???€?‘ë˜???¼ë²¨(.txt) ê²½ë¡œ ? ì¶”
        label_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        
        # 3. ?¼ë²¨ ?Œì¼(.txt)???¤ì œë¡?ì¡´ì¬?˜ëŠ”ì§€ ?•ì¸
        if not os.path.exists(label_path):
            # ?¼ë²¨ ?Œì¼???†ìœ¼ë©?ë°°ê²½ ?´ë?ì§€(Background)ë¡?ê°„ì£¼??            missing_labels.append(label_path)
            
    print(f"  - ê²½ë¡œ ???¤ì œ ì¡´ì¬?˜ëŠ” ?´ë?ì§€: {len(lines) - len(missing_images)}??)
    print(f"  - (?ëŸ¬) ?„ë½?˜ê±°??ê²½ë¡œê°€ ?€ë¦??´ë?ì§€: {len(missing_images)}??)
    print(f"  - (ê²½ê³ ) ?¼ë²¨(.txt) ?Œì¼???†ëŠ” ?´ë?ì§€: {len(missing_labels)}??(YOLO???˜í•´ ë¹??”ë©´?¼ë¡œ ?™ìŠµ/?‰ê???")
    
    if missing_images:
        print("\n[!] ?š¨ 1. ê²½ë¡œê°€ ì¡´ì¬?˜ì? ?ŠëŠ” ?´ë?ì§€ ë¦¬ìŠ¤??(ìµœë? 10ê°œë§Œ ì¶œë ¥):")
        for p in missing_images[:10]:
            print(f"    - {p}")
        if len(missing_images) > 10:
            print(f"    ... ??{len(missing_images) - 10}ê°????ˆìŒ")
            
    if missing_labels:
        print("\n[!] ? ï¸ 2. ?´ë‹¹?˜ëŠ” ?¼ë²¨ ?Œì¼(.txt)??ì°¾ì„ ???†ëŠ” ê²½ë¡œ ë¦¬ìŠ¤??(ìµœë? 10ê°œë§Œ ì¶œë ¥):")
        for p in missing_labels[:10]:
            print(f"    - {p}")
        if len(missing_labels) > 10:
            print(f"    ... ??{len(missing_labels) - 10}ê°????ˆìŒ")

if __name__ == '__main__':
    # ê²€ì¦?val) ?°ì´?°ì? ?™ìŠµ(train) ?°ì´??ëª¨ë‘ ?ŒìŠ¤?¸í•©?ˆë‹¤.
    check_split('data/val_split.txt')
    check_split('data/train_split.txt')

