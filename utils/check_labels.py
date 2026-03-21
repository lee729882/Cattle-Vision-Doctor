import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.append(project_root)
os.chdir(project_root)
import cv2
import random
import os

def check_labels():
    split_file = 'data/train_split.txt'
    if not os.path.exists(split_file):
        print(f"[!] ?Œì¼???†ìŠµ?ˆë‹¤: {split_file}")
        return
        
    with open(split_file, 'r', encoding='utf-8') as f:
        # ë¹?ì¤„ê³¼ ì¤„ë°”ê¿?ë¬¸ì ?œê±°
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    if len(lines) == 0:
        print("[!] ?ˆë ¨ ?°ì´??ê²½ë¡œ ?Œì¼??ë¹„ì–´?ˆìŠµ?ˆë‹¤.")
        return
        
    # ?¼ë²¨ ?Œì¼(.txt)???¤ì œë¡?ì¡´ì¬?˜ëŠ” ?´ë?ì§€ë§??„í„°ë§?(ë°°ê²½ ?´ë?ì§€ 8063???œì™¸)
    labeled_lines = []
    for line in lines:
        label_path = line.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        if os.path.exists(label_path):
            labeled_lines.append(line)
            
    if len(labeled_lines) == 0:
        print("[!] ?¼ë²¨ ?Œì¼??ì¡´ì¬?˜ëŠ” ?´ë?ì§€ê°€ ?˜ë‚˜???†ìŠµ?ˆë‹¤.")
        return
        
    # ?œë¤?¼ë¡œ 3??ë½‘ê¸° (?´ì œ??ë¬´ì¡°ê±??Œê? ?ˆëŠ” ?€ê²??´ë?ì§€ë§?ë½‘í™?ˆë‹¤)
    sampled_lines = random.sample(labeled_lines, min(3, len(labeled_lines)))
    
    # ê²°ê³¼ë¥??€?¥í•  ?´ë” ?ì„±
    out_dir = "label_checks"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[*] ë½‘íŒ 3?¥ì˜ ?´ë?ì§€ ?œê°?”ë? ?œì‘?©ë‹ˆ?? ê²°ê³¼??'{out_dir}/' ?´ë”???€?¥ë©?ˆë‹¤.")
    
    for idx, img_path in enumerate(sampled_lines):
        # 1. ?´ë?ì§€ ?½ê¸°
        img = cv2.imread(img_path)
        if img is None:
            print(f"[!] ?´ë?ì§€ë¥??½ì„ ???†ìŠµ?ˆë‹¤: {img_path}")
            continue
            
        # ?´ë?ì§€ ?¬ê¸°(ê°€ë¡? ?¸ë¡œ) ê°€?¸ì˜¤ê¸?        h, w, _ = img.shape
        
        # 2. ?´ë?ì§€ ê²½ë¡œë¥?ë°”íƒ•?¼ë¡œ ?¼ë²¨(.txt) ê²½ë¡œ ? ì¶”
        # ?? data/images/train/img1.jpg -> data/labels/train/img1.txt
        label_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        
        # 3. ?¼ë²¨ ?½ê³  ê·¸ë¦¬ê¸?        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = f.readlines()
                
            for label in labels:
                parts = label.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_c = float(parts[1])  # ?•ê·œ?”ëœ x ì¤‘ì‹¬
                    y_c = float(parts[2])  # ?•ê·œ?”ëœ y ì¤‘ì‹¬
                    box_w = float(parts[3]) # ?•ê·œ?”ëœ ?ˆë¹„
                    box_h = float(parts[4]) # ?•ê·œ?”ëœ ?’ì´
                    
                    # YOLO ?•ê·œ??ì¢Œí‘œë¥??¤ì œ ?´ë?ì§€ ?½ì? ì¢Œí‘œë¡?ë³€??                    x1 = int((x_c - box_w / 2) * w)
                    y1 = int((y_c - box_h / 2) * h)
                    x2 = int((x_c + box_w / 2) * w)
                    y2 = int((y_c + box_h / 2) * h)
                    
                    # 4. ë°•ìŠ¤ ê·¸ë¦¬ê¸?                    # BGR ê¸°ì? ?•ê´‘ ?°ë‘??Neon Green) = (0, 255, 0), ?ê»˜ 2
                    color = (0, 255, 0)
                    thickness = 2
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                    
                    # ?´ë˜??ID ?ìŠ¤???½ì… (ë³´ê¸° ?½ë„ë¡?
                    text_color = (0, 255, 255) # ?•ê´‘ ?¸ë???Cyan-yellow)
                    cv2.putText(img, f"Class {cls_id}", (x1, max(y1-10, 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        else:
            print(f"[!] ë§¤ì¹­?˜ëŠ” ?¼ë²¨ ?Œì¼???†ìŠµ?ˆë‹¤: {label_path}")
            
        # ê²°ê³¼ ?´ë?ì§€ ?€??        out_path = os.path.join(out_dir, f"check_{idx + 1}.jpg")
        cv2.imwrite(out_path, img)
        print(f"  -> ?€???„ë£Œ: {out_path} (?ë³¸ ?´ë?ì§€: {os.path.basename(img_path)})")

if __name__ == '__main__':
    check_labels()

