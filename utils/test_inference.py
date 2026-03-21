import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.append(project_root)
os.chdir(project_root)
from ultralytics import YOLO
import os

def run_inference():
    # 1. ?™ìŠµ???„ë£Œ??ë² ìŠ¤??ëª¨ë¸ ê°€ì¤‘ì¹˜ ?Œì¼ ë¡œë“œ (YOLO???ë™ ?ì„± ê²½ë¡œ ?€??
    weights_path = "runs/detect/runs/detect/baseline_test/weights/best.pt"
    
    # ?ˆë¹„ ê²½ë¡œ ?•ì¸
    if not os.path.exists(weights_path):
        weights_path = "runs/detect/baseline_test/weights/best.pt"
    
    if not os.path.exists(weights_path):
        print(f"[!] ?„ì§ ëª¨ë¸ ?´ë”???Œì¼???†ìŠµ?ˆë‹¤: {weights_path}")
        print("[!] ë¨¼ì? ?°ë??ì—??'python train_yolo.py'ë¥??ê¹Œì§€ ê¸°ë‹¤?¤ì„œ ?Œë ¤ì£¼ì„¸?? ??")
        return

    # ë² ìŠ¤??ëª¨ë¸ ë¶ˆëŸ¬?¤ê¸°
    model = YOLO(weights_path)

    print("\n[*] ê²€ì¦ìš© ?°ì´??val_split.txt)ë¡?Inference(ì¶”ë¡ )ë¥??œì‘?©ë‹ˆ??..")
    
    # 2. val_split.txt ?ˆì— ?ˆëŠ” ?¬ì§„ 1600?¬ì¥??ê°€?¸ì? AI ë°•ìŠ¤ ?ˆì¸¡ ?˜í–‰
    # YOLO???¬ì§„ ëª©ë¡???´ê¸´ txt ?Œì¼ ê²½ë¡œë¥?ë°”ë¡œ ?£ì–´ì£¼ì–´???Œì•„????ì²˜ë¦¬?´ì¤?ˆë‹¤.
    # ?„ë¡œ?íŠ¸ ?´ë”ë¥?'runs', ?´ë¦„??'val_results'ë¡?ì£¼ì–´ [runs/val_results] ?ˆì— ?€?¥ë˜ê²??©ë‹ˆ??
    results = model.predict(
        source="data/val_split.txt",  
        save=True,                    # ë°•ìŠ¤ê°€ ê·¸ë ¤ì§??´ë?ì§€ ìµœì¢… ?€???¬ë?
        project="runs",               
        name="val_results",           # ?¤ì œ ê²°ê³¼ ?´ë” ?´ë¦„ (runs/val_results/)
        conf=0.25,                    # 25% ?´ìƒ ?•ì‹ ?˜ëŠ” ë¬¼ì²´ë§?ê·¸ë¦¬ê¸?(?„ìš” ??ì¡°ì ˆ)
        show_conf=True,               # ë°•ìŠ¤ ?„ì— AIê°€ ?•ì‹ ?˜ëŠ” ë°±ë¶„??Confidence Score) ?ìˆ˜ ?œì‹œ
        show_labels=True,             # ë°•ìŠ¤ ?„ì— ?´ëŠ ?´ë˜?¤ì¸ì§€ ?´ë¦„/ë²ˆí˜¸ ?œì‹œ
        exist_ok=True                 # ?´ë” ê²½ë¡œ ??–´?°ê¸° ?ˆìš©
    )

    print("\n[*] Inference(ì¶”ë¡ )???±ê³µ?ìœ¼ë¡??„ë£Œ?˜ì—ˆ?µë‹ˆ?? ?¨")
    print("[*] ?ìŠ¤?¸ê? ??–´?Œì›Œì§??”ë ¤??ê²°ê³¼ ?¬ì§„?¤ì? 'runs/val_results/' ?´ë” ?ˆì—???•ì¸?˜ì‹¤ ???ˆìŠµ?ˆë‹¤!")

if __name__ == '__main__':
    # Windows ë©€?°í”„ë¡œì„¸???ëŸ¬ ë°©ì?
    run_inference()

