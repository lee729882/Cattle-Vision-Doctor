import os
import faulthandler
faulthandler.enable()
# [ALARM] [세그폴트 원천 차단 - 최강력]
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['GOMP_SPINCOUNT'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CUDA 강제 비활성화
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
os.environ['TORCH_CUDA_ARCH_LIST'] = ''

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import cv2
print("[OK] cv2 import 완료")

import torch
print("[OK] torch import 완료")

# torch CUDA 런타임 완전 차단
try:
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    torch.cuda.get_device_name = lambda *args, **kwargs: 'cpu'
    torch.cuda.current_device = lambda: -1
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.is_available = lambda: False
    if hasattr(torch._C, '_cuda_getDeviceCount'):
        torch._C._cuda_getDeviceCount = lambda: 0
    print("[OK] CUDA 런타임 완전 차단 (monkey-patch 적용)")
except Exception:
    pass

import subprocess

# OpenCV 멀티스레딩 차단
try:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# torch 스레드 1개 고정
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

import random
import threading
import time
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import sqlite3
import requests
import json

# [MAINT] 서보 하드웨어 라이브러리
HARDWARE_AVAILABLE = False
try:
    from adafruit_extended_bus import ExtendedI2C as I2C
    from adafruit_pca9685 import PCA9685
    from adafruit_motor import servo as servo_lib
    HARDWARE_AVAILABLE = True
    print("[OK] 서보 하드웨어 라이브러리 로드 완료")
except ImportError:
    print("[WARNING] 서보 하드웨어 라이브러리 없음 (소프트웨어 전용 모드)")

try:
    import winsound
except ImportError:
    class winsound:
        @staticmethod
        def Beep(frequency, duration):
            print(f"[SOUND] [시스템 알람] 사이렌 작동: {frequency}Hz, {duration}ms")

# ==========================================
# 1. 스타일 및 글로벌 프리미엄 테마 설정
# ==========================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

COLOR_BG = "#0b0f19"
COLOR_SIDEBAR = "#0f172a"
COLOR_CARD = "#1e293b"
COLOR_ACCENT = "#3b82f6"
COLOR_ACCENT_HOVER = "#2563eb"
COLOR_TEXT_MAIN = "#f9fafb"
COLOR_TEXT_SUB = "#94a3b8"
COLOR_SUCCESS = "#10b981"
COLOR_DANGER = "#f43f5e"
COLOR_WARNING = "#f59e0b"

NVIDIA_API_KEY = "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"

# 텔레그램 봇 설정
TELEGRAM_BOT_TOKEN = "8566455811:AAHxZoTSgW8heolAfMoNNsfkJ7Jeb8Kkafs"
TELEGRAM_CHAT_ID = "8726768542"

base_dir = os.path.dirname(os.path.abspath(__file__))

# AI 엔진 로드 — [ALARM] Scanner / CCTV 완전 분리 인스턴스 (NCNN 공유 세그폴트 방지)
yolo_on_cuda = False
yolo_model_scanner = None      # Scanner 전용 독립 인스턴스
yolo_model_cctv_native = None  # CCTV 전용 독립 인스턴스

PT_MODEL_PATH = os.path.join(base_dir, "01_Model", "best.pt")
CCTV_MODEL_PATH = os.path.join(base_dir, "01_Model", "best2.pt")
NCNN_MODEL_PATH = os.path.join(base_dir, "01_Model", "best_ncnn_model")

# ── Scanner 모델 로드 ───────────────────────────────────────────────────────
try:
    if os.path.isdir(NCNN_MODEL_PATH):
        yolo_model_scanner = YOLO(NCNN_MODEL_PATH, task='detect')
        print("[CHECK] Scanner 모델 로드 완료 (NCNN)")
    else:
        yolo_model_scanner = YOLO(PT_MODEL_PATH)
        print("[CHECK] Scanner 모델 로드 완료 (best.pt)")
except Exception as e:
    print(f"[ERROR] Scanner 모델 로드 실패: {e}")

# ── CCTV 모델 로드 (항상 best2.pt에서 독립 인스턴스로 로드) ──────────────────
try:
    yolo_model_cctv_native = YOLO(CCTV_MODEL_PATH)
    print("[CHECK] CCTV 전용 모델 로드 완료 (best2.pt - 독립 인스턴스)")
except Exception as e:
    print(f"[WARNING] CCTV 모델(best2.pt) 로드 실패, best.pt 독립 인스턴스로 폴백: {e}")
    try:
        yolo_model_cctv_native = YOLO(PT_MODEL_PATH)
        print("[CHECK] CCTV 폴백 모델 로드 완료 (best.pt 독립 인스턴스)")
    except Exception as e2:
        print(f"[ERROR] CCTV 폴백 모델도 로드 실패: {e2}")

# 하위 호환 별칭 (기존 코드 참조 보호)
yolo_model = yolo_model_scanner
yolo_model_cctv = yolo_model_cctv_native

yolo_on_cuda = False
print("[CHECK] CPU 모드로 추론 (CUDA 비활성화 - 세그폴트 방지)")
dino_detector = None


# [ALARM] 글로벌 YOLO 스레드 락
yolo_inference_lock = threading.Lock()
# [ALARM] 글로벌 DB 락 (SQLite 멀티스레딩 충돌 방지)
db_lock = threading.Lock()

# [CHECK] 메인 스레드에서 Scanner 모델만 워밍업
def warmup_scanner():
    if yolo_model is not None:
        try:
            print("[DEBUG] Scanner 모델 메인 스레드 워밍업 시작...")
            _dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            with torch.inference_mode():
                _ = yolo_model(_dummy, verbose=False, device='cpu')
            print("[CHECK] Scanner 모델 워밍업 완료")
        except Exception as e:
            print(f"[WARNING] Scanner 워밍업 실패: {e}")
warmup_scanner()


class VisionDoctorDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.is_destroyed = False

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.scaling_factor = max(0.8, min(sw / 1920, sh / 1080))
        ctk.set_widget_scaling(self.scaling_factor)
        ctk.set_window_scaling(self.scaling_factor)

        ww, wh = int(sw * 0.85), int(sh * 0.85)
        self.title("VISION DOCTOR - Enterprise Balanced Edition")
        self.geometry(f"{ww}x{wh}+{int((sw-ww)/2)}+{int((sh-wh)/2)}")
        self.configure(fg_color=COLOR_BG)

        self.is_analyzing = False
        self.current_img_path = None
        self.stats_history = []
        self.behavior_counts = {"Standing": 0, "Lying": 0, "Eating": 0, "Sleeping": 0}

        self.settings_conf_threshold = 0.40
        self.settings_alert_interval = 300
        self.settings_auto_save = True
        self.settings_mode = "Balanced"
        self.settings_admin_name = "이승철 (Admin)"

        self.responsive_map = {}
        self.resize_timer = None
        self.cctv_active = False
        self.cap = None  # 카메라는 cctv_capture_worker 스레드가 관리

        self.cctv_latest_frame = None
        self.cctv_frame_lock = threading.Lock()
        self._cctv_tk_img_ref = None

        self.cctv_detection_cache = {'boxes': [], 'pollutant_detected': False, 'contamination_ratio': 0.0, 'lens_covered': False}
        self.cctv_detection_lock = threading.Lock()

        # [CHECK] [FIX 1] 스캐너 타이머 ID 추적 — clear_view 시 취소용
        self._scanner_timer = None

        # [CHECK] [FIX 2] CCTV 워커 스레드 종료 이벤트 + 참조 목록
        self._cctv_stop_event = threading.Event()
        self._cctv_stop_event.set()   # 초기: CCTV 비활성 상태
        self._cctv_threads = []        # 실행 중인 CCTV 워커 스레드 목록

        # 원격 제어 및 알람 변수
        self.telegram_last_update_id = 0
        self.telegram_polling_active = True
        self.telegram_enabled = True          # [NEW] 텔레그램 알람 ON/OFF 토글
        self.cctv_paused = False
        self.manual_alarm_trigger = 0
        self.total_dx = 0.0
        self.total_dy = 0.0
        self.last_move_dir = "UNKNOWN"

        self.pollutant_logged = False
        self.alert_logged = False
        self.last_event_t = 0.0

        self.pollutant_detect_start = 0.0
        self.alert_detect_start = 0.0
        self.turned_detect_start = 0.0
        self.ALARM_DEBOUNCE_SECS = 4.0

        # [NEW] 와이퍼 복구 실패 알람 변수
        self.wiper_recovery_pending = False   # 복구 확인 대기 중
        self.recovery_failed = False          # 복구 실패 상태
        self.recovery_fail_time = 0.0         # 복구 실패 발생 시각

        # [NEW] 지속 실패 DB 기록 추적 변수
        self.persistent_issue_start = 0.0     # 지속 이벤트 시작 시각
        self.persistent_issue_type = ""       # 이벤트 종류
        self.persistent_logged = False        # DB 기록 완료 여부
        self.covered_wiper_logged = False     # 가림 감지 시 와이퍼 이미 구동했는지 여부

        # 서보 하드웨어 및 복구 상태 변수
        self.hardware_connected = False
        self.servo_updown = None
        self.servo_leftright = None
        self.servo_wiper = None
        self.servo_pca = None
        self.servo_lock = threading.Lock()

        self.servo_center_lr = 180.0
        self.servo_center_ud = 180.0
        self.servo_current_lr = 180.0
        self.servo_current_ud = 180.0

        self.fov_recovery_enabled = True
        self.fov_ref_saved = False
        self.fov_stable_count = 0
        self.fov_px_per_degree = 0.09375
        self.fov_deadzone = 3.0
        self.fov_correction_active = False

        self.fov_state = "IDLE"
        self.fov_start_lr = 180.0
        self.fov_start_ud = 180.0
        self.last_shock_time = 0.0
        self.cooldown_start = 0.0

        # NIM API 호출 쿨다운 (CCTV 활성 시 30초 1회 제한)
        self.last_nim_call_t = 0.0
        self.nim_min_interval = 30.0

        self.init_servo_hardware()

        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            def _delayed_telegram_init():
                try:
                    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
                    res = requests.get(url, params={"offset": -1, "limit": 1}, timeout=5).json()
                    if res.get("ok") and res.get("result"):
                        self.telegram_last_update_id = res["result"][-1]["update_id"] + 1
                        print(f"[CHECK] 텔레그램 이전 메시지 스킵 (ID: {self.telegram_last_update_id})")
                except:
                    pass
                self.send_telegram_alert("[CHECK] Vision Doctor 시스템이 가동되었습니다.\n원격 제어 대기 중입니다.", include_keyboard=True)
                threading.Thread(target=self.telegram_polling_worker, daemon=True).start()
            self.after(3000, _delayed_telegram_init)

        for folder in ["01_Model", "02_Cattle_Dataset", "05_Detections", "06_Evidences"]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.pack(fill="both", expand=True)

        print("[DEBUG] DB 초기화 시작", flush=True)
        self.init_db()
        print("[DEBUG] DB 초기화 완료", flush=True)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        print("[DEBUG] 대시보드 렌더링 시작", flush=True)
        self.show_dashboard()
        print("[DEBUG] 대시보드 렌더링 완료", flush=True)

    def on_closing(self):
        self.is_destroyed = True
        self.stop_cctv()
        try:
            if self.servo_pca:
                self.servo_pca.deinit()
                print("[SETTINGS] 서보 PCA9685 해제 완료")
        except:
            pass
        try:
            self.destroy()
        except:
            pass
        os._exit(0)

    def safe_after(self, ms, func, *args):
        if not self.is_destroyed:
            try:
                return self.after(ms, func, *args)
            except:
                pass
        return None

    def init_servo_hardware(self):
        if not HARDWARE_AVAILABLE:
            print("[WARNING] 서보 하드웨어 라이브러리 없음 → 소프트웨어 전용 모드")
            return
        try:
            i2c = I2C(7)
            self.servo_pca = PCA9685(i2c)
            self.servo_pca.frequency = 50
            self.servo_updown = servo_lib.Servo(self.servo_pca.channels[0])
            self.servo_leftright = servo_lib.Servo(self.servo_pca.channels[5])
            self.servo_wiper = servo_lib.Servo(self.servo_pca.channels[12])
            self.servo_updown.angle = self.servo_center_ud
            self.servo_leftright.angle = self.servo_center_lr
            self.hardware_connected = True
            print("[CHECK] 서보 하드웨어 초기화 완료 (I2C 7, PCA9685)")
            print(f"   채널0(상하)={self.servo_center_ud}°, 채널5(좌우)={self.servo_center_lr}°, 채널12(와이퍼)")
        except Exception as e:
            print(f"[WARNING] 서보 하드웨어 연결 실패 (소프트웨어 전용): {e}")
            self.hardware_connected = False

    def servo_move(self, lr_angle=None, ud_angle=None):
        if not self.hardware_connected:
            return
        with self.servo_lock:
            try:
                if lr_angle is not None:
                    lr_angle = max(0, min(180, lr_angle))
                    self.servo_leftright.angle = lr_angle
                    self.servo_current_lr = lr_angle
                if ud_angle is not None:
                    ud_angle = max(0, min(180, ud_angle))
                    self.servo_updown.angle = ud_angle
                    self.servo_current_ud = ud_angle
            except Exception as e:
                print(f"[WARNING] 서보 이동 실패: {e}")

    # =========================================================================
    # FOV 상태머신 (IDLE→SHOCK→RECOVERING→COOLDOWN)
    # =========================================================================
    # (fov_state_machine_tick는 cctv_render_loop로 병합되어 제거됨)

    def fov_reset_origin(self):
        self.total_dx = 0.0
        self.total_dy = 0.0
        self.fov_state = "IDLE"
        self.fov_prev_gray = None
        self.fov_ref_saved = False
        self.fov_stable_count = 0
        self.fov_correction_active = False
        self.fov_current_lr = self.servo_center_lr
        self.fov_current_ud = self.servo_center_ud
        if self.hardware_connected:
            self.servo_move(lr_angle=self.servo_center_lr, ud_angle=self.servo_center_ud)
        self.safe_after(0, lambda: self.add_cctv_log("[CHECK] [화각복구] 기준 좌표 재설정 완료 (현재 위치 = 원점)"))

    def init_db(self):
        """DB 스키마 초기화 — 메인 스레드 1회실행, 완료 후 즉시 close (self.db_conn 공유 커넥션 폐기)."""
        DB_PATH = "vision_doctor.db"
        with db_lock:
            try:
                conn = sqlite3.connect(DB_PATH)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS inference_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT, count INTEGER, confidence REAL,
                        behavior TEXT, clarity REAL, admin TEXT, report TEXT, latency REAL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS security_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT, weapon_type TEXT, confidence REAL,
                        image_path TEXT, alert_sent INTEGER
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS lens_failures (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        event_type TEXT,
                        contamination_ratio REAL,
                        duration_seconds REAL,
                        wiper_attempted INTEGER,
                        recovery_result TEXT,
                        admin TEXT
                    )
                """)
                cursor = conn.execute("PRAGMA table_info(inference_logs)")
                existing_cols = [col[1] for col in cursor.fetchall()]
                if "report" not in existing_cols:
                    conn.execute("ALTER TABLE inference_logs ADD COLUMN report TEXT")
                if "latency" not in existing_cols:
                    conn.execute("ALTER TABLE inference_logs ADD COLUMN latency REAL")
                conn.commit()
            except Exception as e:
                print(f"[ERROR] init_db 오류: {e}")
            finally:
                try:
                    conn.close()
                except:
                    pass

    def log_lens_failure(self, event_type, contamination_ratio, duration_seconds, wiper_attempted, recovery_result):
        """[NEW] 렌즈 지속 실패 이벤트 DB 저장"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with db_lock:
            conn = None
            try:
                conn = sqlite3.connect("vision_doctor.db")
                conn.execute(
                    "INSERT INTO lens_failures "
                    "(timestamp, event_type, contamination_ratio, duration_seconds, wiper_attempted, recovery_result, admin) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ts, event_type, contamination_ratio, duration_seconds,
                     1 if wiper_attempted else 0, recovery_result, self.settings_admin_name))
                conn.commit()
                print(f"[DB] lens_failures 저장: {event_type} | {duration_seconds:.1f}s | {recovery_result}")
            except Exception as e:
                print(f"[ERROR] log_lens_failure DB 오류: {e}")
            finally:
                try:
                    if conn:
                        conn.close()
                except:
                    pass

    def log_inference(self, count, confidence, behavior, clarity, report, latency):
        """DB INSERT — 별도 sqlite3.connect()으로 스레드 독립 커넥션 사용."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with db_lock:
            conn = None
            try:
                conn = sqlite3.connect("vision_doctor.db")
                conn.execute(
                    "INSERT INTO inference_logs (timestamp, count, confidence, behavior, clarity, admin, report, latency) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (ts, count, confidence, behavior, clarity, self.settings_admin_name, report, latency))
                conn.commit()
            except Exception as e:
                print(f"[ERROR] DB 저장 오류 (log_inference): {e}")
            finally:
                try:
                    if conn:
                        conn.close()
                except:
                    pass

    def log_security_event(self, weapon_type, confidence, image_path, alert_sent):
        """DB INSERT — 별도 sqlite3.connect()으로 스레드 독립 커넥션 사용."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with db_lock:
            conn = None
            try:
                conn = sqlite3.connect("vision_doctor.db")
                conn.execute(
                    "INSERT INTO security_logs (timestamp, weapon_type, confidence, image_path, alert_sent) VALUES (?, ?, ?, ?, ?)",
                    (ts, weapon_type, confidence, image_path, alert_sent))
                conn.commit()
            except Exception as e:
                print(f"[ERROR] DB 저장 오류 (log_security_event): {e}")
            finally:
                try:
                    if conn:
                        conn.close()
                except:
                    pass

    def save_evidence(self, frame, prefix):
        ts_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name = f"{prefix}_evt_{ts_name}.jpg"
        img_path = os.path.join("06_Evidences", img_name)
        cv2.imwrite(img_path, frame)
        return img_path

    def send_telegram_alert(self, message, image_path=None, include_keyboard=False):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return
        if not getattr(self, 'telegram_enabled', True):
            print(f"[TELEGRAM OFF] 메시지 차단됨: {message[:40]}...")
            return
        def task():
            try:
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
                if include_keyboard:
                    payload["reply_markup"] = json.dumps({
                        "keyboard": [[{"text": "[CAMERA] 현장 캡처"}, {"text": "[DATA] 상태 요약"}],
                                     [{"text": "[ALARM] 수동 사이렌"}, {"text": "[PAUSE] 감시 정지"}, {"text": "[PLAY] 감시 재개"}],
                                     [{"text": "[SCREEN] PC CCTV 켜기"}, {"text": "[SCREEN] PC CCTV 끄기"}]],
                        "resize_keyboard": True, "persistent": True
                    })
                requests.post(url, data=payload, timeout=5)
                if image_path and os.path.exists(image_path):
                    with open(image_path, 'rb') as photo:
                        requests.post(
                            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                            data={"chat_id": TELEGRAM_CHAT_ID, "caption": "[ALARM] 현장 증거 이미지"},
                            files={"photo": photo}, timeout=10)
            except:
                pass
        threading.Thread(target=task, daemon=True).start()

    def telegram_polling_worker(self):
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
        while self.telegram_polling_active and not self.is_destroyed:
            try:
                res = requests.get(url, params={"offset": self.telegram_last_update_id, "timeout": 20}, timeout=25).json()
                if res.get("ok"):
                    for item in res.get("result", []):
                        self.telegram_last_update_id = item["update_id"] + 1
                        text = item.get("message", {}).get("text", "")
                        if text:
                            self.handle_telegram_command(text)
            except:
                time.sleep(3)

    def handle_telegram_command(self, cmd):
        cmd = cmd.strip()
        if cmd in ["/capture", "[CAMERA] 현장 캡처"]:
            if self.cctv_active and getattr(self, 'cctv_latest_frame', None) is not None:
                img_path = self.save_evidence(self.cctv_latest_frame, "manual")
                self.send_telegram_alert("[CAMERA] 원격 수동 캡처 이미지입니다.", img_path)
        elif cmd in ["/alarm", "[ALARM] 수동 사이렌"]:
            self.manual_alarm_trigger = 50
            self.send_telegram_alert("[ALARM] 현장 PC에 수동 경고/사이렌을 발동했습니다!")
            try:
                threading.Thread(target=lambda: winsound.Beep(2000, 1500), daemon=True).start()
            except:
                pass
        elif cmd in ["/stop", "[PAUSE] 감시 정지"]:
            self.cctv_paused = True
            self.send_telegram_alert("[PAUSE] 감시가 일시정지 되었습니다.")
        elif cmd in ["/start", "[PLAY] 감시 재개"]:
            self.cctv_paused = False
            self.send_telegram_alert("[PLAY] 감시가 재개되었습니다.")
        elif cmd in ["/opencctv", "[SCREEN] PC CCTV 켜기"]:
            self.safe_after(0, self.show_cctv)
            self.send_telegram_alert("[SCREEN] PC 화면을 CCTV 모드로 전환했습니다.")
        elif cmd in ["/home", "[HOME] PC 대시보드", "[SCREEN] PC CCTV 끄기"]:
            self.safe_after(0, self.show_dashboard)
            self.send_telegram_alert("[SCREEN] PC CCTV 모니터링을 종료했습니다.")

    def telegram_livestream_worker(self):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return
        stop_ev = self._cctv_stop_event
        while not stop_ev.is_set() and self.cctv_active and getattr(self, 'cctv_latest_frame', None) is None:
            time.sleep(0.5)
        if not self.cctv_active or stop_ev.is_set():
            return
        try:
            with self.cctv_frame_lock:
                frame_copy = self.cctv_latest_frame.copy()
            ret, buffer = cv2.imencode('.jpg', frame_copy)
            if not ret:
                return
            res = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": "[ALARM] CCTV 라이브 스트리밍 시작 (3초 갱신)"},
                files={"photo": ("frame.jpg", buffer.tobytes(), "image/jpeg")}, timeout=10).json()
            if not res.get("ok"):
                return
            message_id = res["result"]["message_id"]
        except:
            return

        url_edit = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/editMessageMedia"
        while not stop_ev.is_set() and self.cctv_active:
            time.sleep(3.0)
            if getattr(self, 'cctv_paused', False) or getattr(self, 'cctv_latest_frame', None) is None:
                continue
            try:
                with self.cctv_frame_lock:
                    frame_copy = self.cctv_latest_frame.copy()
                ret, buffer = cv2.imencode('.jpg', frame_copy)
                if ret:
                    requests.post(url_edit,
                        data={"chat_id": TELEGRAM_CHAT_ID, "message_id": message_id,
                              "media": json.dumps({"type": "photo", "media": "attach://photo"})},
                        files={"photo": ("frame.jpg", buffer.tobytes(), "image/jpeg")}, timeout=10)
            except:
                pass
        try:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/editMessageCaption",
                data={"chat_id": TELEGRAM_CHAT_ID, "message_id": message_id,
                      "caption": "[STOP] CCTV 라이브 스트리밍 종료."}, timeout=5)
        except:
            pass

    def get_font(self, size, weight="normal"):
        return ("Inter", int(size), weight)

    def scale(self, value):
        return int(value)

    def clear_view(self):
        # [CHECK] [FIX 1] 스캐너 타이머 취소 — CCTV 진입 후 타이머 발동 방지
        if self._scanner_timer is not None:
            try:
                self.after_cancel(self._scanner_timer)
            except:
                pass
            self._scanner_timer = None

        self.stop_cctv()
        self.responsive_map = {}
        for widget in self.container.winfo_children():
            try:
                widget.destroy()
            except:
                pass

    def stop_cctv(self):
        """
        [CHECK] [FIX 2] 스레드 안전 종료:
        - stop_event 세팅으로 모든 워커에게 종료 신호
        - join()으로 실제 종료 확인 (최대 3초)
        - cap은 cctv_capture_worker 스레드 내부에서 release됨
        """
        if not self.cctv_active and self._cctv_stop_event.is_set():
            return

        self.cctv_active = False
        self._cctv_stop_event.set()  # 모든 워커에게 종료 신호

        # 워커 스레드가 실제로 끝날 때까지 대기 (메인 스레드는 최대 3초 대기)
        for t in self._cctv_threads:
            t.join(timeout=3.0)
            if t.is_alive():
                print(f"[WARNING] CCTV 워커 스레드 join 타임아웃: {t.name}")
        self._cctv_threads = []
        # cap은 cctv_capture_worker 내부에서 release됨 — 여기서 release 금지
        print("[PROCESS] CCTV 워커 종료 완료")

    def setup_responsive_image(self, label, ctk_img, mode="fill"):
        self.responsive_map[label] = {"img": ctk_img, "mode": mode}
        label.bind("<Configure>", lambda e=None: self.debounce_resize())

    def debounce_resize(self):
        if self.resize_timer:
            self.after_cancel(self.resize_timer)
        self.resize_timer = self.safe_after(40, self.execute_dynamic_resize)

    def execute_dynamic_resize(self):
        for label, data in self.responsive_map.items():
            try:
                w, h = label.winfo_width(), label.winfo_height()
                if w > 10 and h > 10:
                    data["img"].configure(size=(w + 1, h + 1) if data["mode"] == "fill" else (w, h))
            except:
                continue

    # ------------------------------------------
    # 2. 메인 대시보드
    # ------------------------------------------
    def show_dashboard(self):
        self.clear_view()
        self.dashboard_view = ctk.CTkFrame(self.container, fg_color="transparent")
        self.dashboard_view.pack(fill="both", expand=True)

        header_bar = ctk.CTkFrame(self.dashboard_view, fg_color="transparent", height=self.scale(100))
        header_bar.pack(fill="x", padx=self.scale(45), pady=(self.scale(35), 0))

        title_frame = ctk.CTkFrame(header_bar, fg_color="transparent")
        title_frame.pack(side="left")
        ctk.CTkLabel(title_frame, text="Vision Doctor", font=self.get_font(38, "bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w")
        ctk.CTkLabel(title_frame, text="AI-Powered Precision Livestock Diagnostic System", font=self.get_font(16), text_color=COLOR_TEXT_SUB).pack(anchor="w")

        btn_area = ctk.CTkFrame(header_bar, fg_color="transparent")
        btn_area.pack(side="right")
        for ic_text in ["[SETTING]", "[ADMIN]"]:
            lbl = ctk.CTkLabel(btn_area, text=ic_text, font=self.get_font(26), padx=self.scale(12), cursor="hand2", text_color=COLOR_TEXT_MAIN)
            lbl.pack(side="left")
            if ic_text == "[SETTING]":
                lbl.bind("<Button-1>", lambda e=None: self.show_settings())
            elif ic_text == "[ADMIN]":
                lbl.bind("<Button-1>", lambda e=None: self.show_statistics())

        cards_grid = ctk.CTkFrame(self.dashboard_view, fg_color="transparent")
        cards_grid.pack(fill="both", expand=True, padx=45, pady=45)
        cards_grid.grid_columnconfigure(0, weight=1, uniform="dashboard_cols")
        cards_grid.grid_columnconfigure(1, weight=1, uniform="dashboard_cols")
        cards_grid.grid_rowconfigure(0, weight=1)

        self.scanner_card = ctk.CTkFrame(cards_grid, fg_color=COLOR_CARD, corner_radius=32, border_width=1, border_color="#334155")
        self.scanner_card.grid(row=0, column=0, padx=(0, 25), pady=0, sticky="nsew")

        content_inner = ctk.CTkFrame(self.scanner_card, fg_color="transparent")
        content_inner.pack(expand=True)

        try:
            icon_img = Image.open("02_Cattle_Dataset/farm_scanner_icon.png")
            ic_size = self.scale(200)
            self.icon_photo = ctk.CTkImage(light_image=icon_img, dark_image=icon_img, size=(ic_size, ic_size))
            ctk.CTkLabel(content_inner, image=self.icon_photo, text="").pack(pady=(0, self.scale(30)))
        except:
            ctk.CTkLabel(content_inner, text="[SEARCH]", font=self.get_font(100)).pack(pady=(0, self.scale(30)))

        ctk.CTkLabel(content_inner, text="Farm Scanner", font=self.get_font(56, "bold"), text_color=COLOR_TEXT_MAIN).pack(pady=(self.scale(10), self.scale(5)))
        ctk.CTkLabel(content_inner, text="AI-Powered Livestock Recognition\n& Behavioral Analytics", font=self.get_font(17), text_color=COLOR_TEXT_SUB, justify="center").pack(pady=(0, self.scale(65)))

        self.btn_main = ctk.CTkButton(content_inner, text="MISSION OVERVIEW", command=self.show_project_overview,
            fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, font=self.get_font(21, "bold"),
            height=self.scale(65), width=self.scale(320), corner_radius=self.scale(18))
        self.btn_main.pack()

        self.feed_area = ctk.CTkFrame(cards_grid, fg_color="transparent")
        self.feed_area.grid(row=0, column=1, padx=(25, 0), pady=0, sticky="nsew")
        self.feed_area.grid_rowconfigure(0, weight=1, uniform="feed_rows")
        self.feed_area.grid_rowconfigure(1, weight=1, uniform="feed_rows")
        self.feed_area.grid_rowconfigure(2, weight=1, uniform="feed_rows")
        self.feed_area.grid_columnconfigure(0, weight=1)

        self.create_feed_card(self.feed_area, "REAL-TIME FARM SCANNER", "02_Cattle_Dataset/farm_feed_1.png", 0, command=self.show_scanner)
        self.create_feed_card(self.feed_area, "LIVE CCTV MONITORING", "02_Cattle_Dataset/farm_feed_2.png", 1, command=self.show_cctv)
        self.create_feed_card(self.feed_area, "SECURITY EVIDENCE GALLERY", "02_Cattle_Dataset/farm_feed_2.png", 2, command=self.show_security_gallery)

    def create_feed_card(self, parent, title, img_path, row, command=None):
        card = ctk.CTkFrame(parent, fg_color=COLOR_CARD, corner_radius=28, border_width=1, border_color="#334155", cursor="hand2")
        card.grid(row=row, column=0, pady=(0, 25) if row == 0 else (0, 0), sticky="nsew")
        if command:
            card.bind("<Button-1>", lambda e=None: command())

        badge = ctk.CTkFrame(card, fg_color="#0f172a", corner_radius=12)
        badge.place(x=20, y=20)
        ctk.CTkLabel(badge, text=title, font=("Inter", 13, "bold"), text_color=COLOR_TEXT_MAIN, padx=15, pady=6).pack()

        try:
            fimg = Image.open(img_path)
            res_img = ctk.CTkImage(light_image=fimg, dark_image=fimg, size=(100, 100))
            lbl = ctk.CTkLabel(card, image=res_img, text="")
            lbl.pack(expand=True, fill="both", padx=0, pady=0)
            if command:
                lbl.bind("<Button-1>", lambda e=None: command())
            self.setup_responsive_image(lbl, res_img, mode="fill")
            badge.lift()
        except:
            pass

    def show_project_overview(self):
        self.clear_view()
        self.ov_animation_active = True
        v = ctk.CTkFrame(self.container, fg_color=COLOR_BG)
        v.pack(fill="both", expand=True)

        h = ctk.CTkFrame(v, fg_color="transparent")
        h.pack(fill="x", padx=self.scale(45), pady=self.scale(25))
        ctk.CTkButton(h, text="[BACK] BACK TO DASHBOARD",
            command=lambda: [setattr(self, 'ov_animation_active', False), self.show_dashboard()],
            fg_color="#1e293b", hover_color="#334155", width=self.scale(220), height=self.scale(48), corner_radius=self.scale(12)).pack(side="left")

        self.scroll_area = ctk.CTkScrollableFrame(v, fg_color="transparent", label_text="")
        self.scroll_area.pack(expand=True, fill="both", padx=self.scale(60), pady=self.scale(10))

        btn_box = ctk.CTkFrame(v, fg_color="transparent")
        btn_box.pack(fill="x", pady=(self.scale(10), self.scale(50)))
        ctk.CTkButton(btn_box, text="ENTER SCANNER MODULE", font=self.get_font(22, "bold"),
            fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER, height=self.scale(65), width=self.scale(350),
            corner_radius=self.scale(20),
            command=lambda: [setattr(self, 'ov_animation_active', False), self.show_scanner()]).pack()

        self.full_content = [
            ("VISION DOCTOR: THE FUTURE OF LIVESTOCK", "hero"),
            ("At the intersection of Intelligence and animal welfare, Vision Doctor sets the global standard for cattle monitoring.", "body"),
            ("Powering our core engine is YOLOv8, delivering ultra-fast detection for every head of cattle with unparalleled precision.", "body"),
            ("Our mission is Sustainability. By analyzing behavioral consistency and lens clarity, we empower farmers with data-driven insights.", "body"),
            ("Join the next generation of precision livestock technology.", "body"),
            ("--------------------------------------------------", "body"),
            ("READY TO BEGIN THE ANALYTICS JOURNEY?", "highlight")
        ]
        self.animate_overview_text(0)
        self.auto_scroll_overview()

    def animate_overview_text(self, idx):
        if not hasattr(self, 'scroll_area') or not self.scroll_area.winfo_exists() or \
                not getattr(self, 'ov_animation_active', False) or idx >= len(self.full_content):
            return
        text, style = self.full_content[idx]
        font, color = (self.get_font(48, "bold"), "#FFFFFF") if style == "hero" else \
                      (self.get_font(26, "bold"), COLOR_ACCENT) if style == "highlight" else \
                      (self.get_font(22), "#cbd5e1")
        lbl = ctk.CTkLabel(self.scroll_area, text=text, font=font, text_color=color,
                           wraplength=self.scale(1100), justify="left")
        lbl.pack(anchor="w", pady=self.scale(20), padx=self.scale(40))
        self.safe_after(2000 if style == "hero" else 800, lambda: self.animate_overview_text(idx + 1))

    def auto_scroll_overview(self):
        if not hasattr(self, 'scroll_area') or not self.scroll_area.winfo_exists() or \
                not getattr(self, 'ov_animation_active', False):
            return
        try:
            self.scroll_area._parent_canvas.yview_scroll(1, "units")
        except:
            pass
        self.safe_after(2000, self.auto_scroll_overview)

    def show_security_gallery(self):
        self.clear_view()
        main = ctk.CTkFrame(self.container, fg_color=COLOR_BG)
        main.pack(fill="both", expand=True)
        h = ctk.CTkFrame(main, fg_color="transparent")
        h.pack(fill="x", padx=self.scale(45), pady=(self.scale(25), self.scale(15)))
        btn_frame = ctk.CTkFrame(h, fg_color="transparent")
        btn_frame.pack(side="left")

        ctk.CTkButton(btn_frame, text="[BACK] BACK TO DASHBOARD", command=self.show_dashboard,
            fg_color="#1e293b", hover_color="#334155", width=self.scale(200), height=self.scale(48), corner_radius=self.scale(12)).pack(side="left", padx=(0, self.scale(10)))
        ctk.CTkButton(btn_frame, text="[CCTV] CCTV 켜기", command=self.show_cctv,
            fg_color=COLOR_SUCCESS, hover_color="#059669", width=self.scale(150), height=self.scale(48), corner_radius=self.scale(12), font=self.get_font(14, "bold")).pack(side="left")
        ctk.CTkLabel(h, text="Security Evidence Gallery", font=self.get_font(28, "bold"), text_color=COLOR_DANGER).pack(side="right")

        self.gallery_filter_var = ctk.StringVar(value="전체 (All)")
        filters = ["전체 (All)", "복구 실패", "물체 오염", "렌즈 오염/가림", "카메라 움직임"]
        seg_btn = ctk.CTkSegmentedButton(main, values=filters, variable=self.gallery_filter_var,
            command=self.load_gallery_data, font=self.get_font(16),
            selected_color=COLOR_ACCENT, selected_hover_color=COLOR_ACCENT_HOVER)
        seg_btn.pack(pady=(0, self.scale(10)), padx=self.scale(40), fill="x")

        self.gallery_frame = ctk.CTkScrollableFrame(main, fg_color="transparent")
        self.gallery_frame.pack(fill="both", expand=True, padx=self.scale(40), pady=(0, self.scale(20)))
        self.load_gallery_data("전체 (All)")

    def load_gallery_data(self, filter_val):
        for widget in self.gallery_frame.winfo_children():
            widget.destroy()
        query = "SELECT timestamp, weapon_type, image_path FROM security_logs"
        if filter_val == "렌즈 오염/가림":
            query += " WHERE weapon_type IN ('렌즈 가림', '렌즈 오염/가림')"
        elif filter_val != "전체 (All)":
            query += f" WHERE weapon_type = '{filter_val}'"
        query += " ORDER BY id DESC"

        with db_lock:
            conn = None
            try:
                conn = sqlite3.connect("vision_doctor.db")
                all_data = conn.execute(query).fetchall()
            except:
                all_data = []
            finally:
                try:
                    if conn:
                        conn.close()
                except:
                    pass

        if not all_data:
            ctk.CTkLabel(self.gallery_frame, text=f"[{filter_val}] 항목에 대한 기록이 없습니다.",
                font=self.get_font(20), text_color=COLOR_TEXT_SUB).pack(pady=self.scale(100))
            return

        columns = 3
        for i, (ts, w_type, img_path) in enumerate(all_data):
            row, col = i // columns, i % columns
            
            # [NEW] 복구 실패 항목 붉은색 강렬 강조
            is_fail = (w_type == "복구 실패")
            color_border = COLOR_DANGER if is_fail else (COLOR_WARNING if "오염" in w_type or "가림" in w_type else COLOR_ACCENT)
            bg_color = "#450a0a" if is_fail else COLOR_CARD
            
            card = ctk.CTkFrame(self.gallery_frame, fg_color=bg_color, corner_radius=self.scale(15),
                border_width=3 if is_fail else 1, border_color=color_border)
            card.grid(row=row, column=col, padx=self.scale(15), pady=self.scale(15), sticky="nsew")

            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    img.thumbnail((self.scale(320), self.scale(240)))
                    lbl = ctk.CTkLabel(card, image=ctk.CTkImage(light_image=img, dark_image=img, size=img.size), text="")
                    lbl.pack(padx=self.scale(10), pady=(self.scale(10), 0))
                except:
                    ctk.CTkLabel(card, text="Image Load Error", text_color=COLOR_DANGER,
                        width=self.scale(320), height=self.scale(240)).pack(pady=self.scale(10))
            else:
                ctk.CTkLabel(card, text="Image Missing", text_color=COLOR_TEXT_SUB,
                    width=self.scale(320), height=self.scale(240)).pack(pady=self.scale(10))

            dt_parts = ts.split(" ")
            
            if is_fail:
                ctk.CTkLabel(card, text="[CRITICAL] RECOVERY FAILED", font=self.get_font(18, "bold"), text_color="#fca5a5").pack(pady=(self.scale(10), 0))
            else:
                ctk.CTkLabel(card, text=f"Type: {w_type}", font=self.get_font(16, "bold"), text_color=color_border).pack(pady=(self.scale(10), 0))
                
            ctk.CTkLabel(card, text=f"Date: {dt_parts[0] if len(dt_parts) > 0 else ts}", font=self.get_font(14), text_color=COLOR_TEXT_MAIN if not is_fail else "#fecaca").pack()
            ctk.CTkLabel(card, text=f"Time: {dt_parts[1] if len(dt_parts) > 1 else ''}", font=self.get_font(14), text_color=COLOR_TEXT_SUB if not is_fail else "#fca5a5").pack(pady=(0, self.scale(10)))

    # ------------------------------------------
    # 3. Farm Scanner
    # ------------------------------------------
    def show_scanner(self):
        self.clear_view()
        sv = ctk.CTkFrame(self.container, fg_color=COLOR_BG)
        sv.pack(fill="both", expand=True)
        t = ctk.CTkFrame(sv, fg_color="transparent", height=self.scale(100))
        t.pack(fill="x", padx=self.scale(35), pady=(self.scale(35), 0))

        title_box = ctk.CTkFrame(t, fg_color="transparent")
        title_box.pack(side="left")
        ctk.CTkLabel(title_box, text="Vision Doctor", font=self.get_font(30, "bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w")
        ctk.CTkLabel(title_box, text="REAL-TIME SCANNER MODULE", font=self.get_font(14), text_color=COLOR_ACCENT).pack(anchor="w")

        menu_area = ctk.CTkFrame(t, fg_color="transparent")
        menu_area.pack(side="right")
        ctk.CTkButton(menu_area, text="[RANDOM] RANDOM SAMPLE", font=self.get_font(14, "bold"), fg_color="#10b981",
            width=self.scale(160), height=self.scale(42), corner_radius=self.scale(10),
            command=self.run_random_inference).pack(side="left", padx=self.scale(8))
        ctk.CTkButton(menu_area, text="HOME", font=self.get_font(14), fg_color="#334155",
            width=self.scale(90), height=self.scale(42), corner_radius=self.scale(10),
            command=self.show_dashboard).pack(side="left", padx=self.scale(8))

        for ic in ["[SETTING]", "[ADMIN]"]:
            lbl = ctk.CTkLabel(menu_area, text=ic, font=self.get_font(24), padx=self.scale(10), cursor="hand2", text_color=COLOR_TEXT_MAIN)
            lbl.pack(side="left")
            if ic == "[SETTING]":
                lbl.bind("<Button-1>", lambda e=None: self.show_settings())
            elif ic == "[ADMIN]":
                lbl.bind("<Button-1>", lambda e=None: self.show_statistics())

        self.alert_banner = ctk.CTkFrame(sv, fg_color="#dc2626", corner_radius=self.scale(14), height=self.scale(60))
        self.alert_banner_label = ctk.CTkLabel(self.alert_banner, text="", font=self.get_font(16, "bold"), text_color="#FFFFFF")
        self.alert_banner_label.pack(expand=True, fill="both", padx=self.scale(20))
        self.alert_banner_visible = False
        self._alert_blink_active = False

        grid = ctk.CTkFrame(sv, fg_color="transparent")
        grid.pack(fill="both", expand=True, padx=self.scale(35), pady=self.scale(30))
        grid.grid_columnconfigure(0, weight=3)
        grid.grid_columnconfigure(1, weight=1)
        grid.grid_rowconfigure(0, weight=1)

        self.v_card = ctk.CTkFrame(grid, fg_color=COLOR_CARD, corner_radius=self.scale(28), border_width=1, border_color="#334155")
        self.v_card.grid(row=0, column=0, sticky="nsew", padx=(0, self.scale(30)))
        ctk.CTkLabel(self.v_card, text="V I S I O N   M O N I T O R", font=self.get_font(14, "bold"), text_color=COLOR_ACCENT).pack(anchor="nw", padx=self.scale(30), pady=self.scale(25))

        self.img_lbl = ctk.CTkLabel(self.v_card, text="[INIT] INITIALIZING AI ENGINE...", font=self.get_font(24), text_color="#64748b")
        self.img_lbl.pack(expand=True, fill="both", padx=self.scale(30), pady=(0, self.scale(30)))

        self.r_panel = ctk.CTkFrame(grid, fg_color="transparent", width=self.scale(440))
        self.r_panel.grid(row=0, column=1, sticky="nsew")
        self.r_panel.grid_propagate(False)

        st = ctk.CTkFrame(self.r_panel, fg_color=COLOR_SIDEBAR, corner_radius=self.scale(28), border_width=1, border_color=COLOR_ACCENT)
        st.pack(fill="x", pady=(0, self.scale(25)))
        ctk.CTkLabel(st, text="BEHAVIOR SUMMARY", font=self.get_font(13, "bold"), text_color=COLOR_TEXT_SUB).pack(anchor="nw", padx=self.scale(30), pady=(self.scale(25), self.scale(5)))
        self.st_val = ctk.CTkLabel(st, text="[INITIALIZING]", font=self.get_font(36, "bold"), text_color=COLOR_SUCCESS)
        self.st_val.pack(anchor="nw", padx=self.scale(30), pady=(0, self.scale(5)))

        bhv_row = ctk.CTkFrame(st, fg_color="transparent")
        bhv_row.pack(fill="x", padx=self.scale(25), pady=(0, self.scale(10)))
        self.lbl_standing = ctk.CTkLabel(bhv_row, text="[STAND] Standing: 0", font=self.get_font(12), text_color=COLOR_ACCENT)
        self.lbl_standing.pack(side="left", padx=(0, self.scale(5)))
        self.lbl_sleeping = ctk.CTkLabel(bhv_row, text="[SLEEP] Sleeping: 0", font=self.get_font(12), text_color=COLOR_SUCCESS)
        self.lbl_sleeping.pack(side="left", padx=(0, self.scale(5)))
        self.lbl_eating = ctk.CTkLabel(bhv_row, text="[EAT] Eating: 0", font=self.get_font(12), text_color=COLOR_WARNING)
        self.lbl_eating.pack(side="left")

        m_list = ctk.CTkFrame(st, fg_color="transparent")
        m_list.pack(fill="x", padx=self.scale(30), pady=(self.scale(10), self.scale(30)))
        self.m_count = self.add_m_item(m_list, "LIVESTOCK COUNT", "0")
        self.m_conf = self.add_m_item(m_list, "INFERENCE CONFIDENCE", "0.0%")

        ctk.CTkLabel(m_list, text="LENS CLARITY", font=self.get_font(14), text_color=COLOR_TEXT_SUB).pack(anchor="w", pady=(self.scale(10), self.scale(5)))
        self.m_clarity = ctk.CTkProgressBar(m_list, height=self.scale(12), fg_color="#1e293b", progress_color=COLOR_SUCCESS)
        self.m_clarity.pack(fill="x")
        self.m_clarity.set(1.0)

        rt = ctk.CTkFrame(self.r_panel, fg_color=COLOR_SIDEBAR, corner_radius=self.scale(28), border_width=1, border_color="#334155")
        rt.pack(fill="both", expand=True)
        ctk.CTkLabel(rt, text="AI DIAGNOSTIC REPORT", font=self.get_font(18, "bold"), text_color=COLOR_ACCENT).pack(anchor="nw", padx=self.scale(30), pady=self.scale(25))
        self.rpt_box = ctk.CTkTextbox(rt, font=self.get_font(16), fg_color="#0b1322", text_color="#cbd5e1",
            corner_radius=self.scale(18), spacing3=self.scale(10), wrap="word")
        self.rpt_box.pack(fill="both", expand=True, padx=self.scale(25), pady=(0, self.scale(30)))

        print("[DEBUG] 대시보드 타이머 스케줄링 전", flush=True)
        if not self.current_img_path:
            # [CHECK] [FIX 1] 타이머 ID 저장 — clear_view()에서 취소 가능하도록
            self._scanner_timer = self.safe_after(500, self.run_random_inference)
        else:
            self._scanner_timer = self.safe_after(self.settings_alert_interval * 1000, self.run_random_inference)
        print("[DEBUG] 대시보드 타이머 스케줄링 완료", flush=True)

    # ------------------------------------------
    # 4. CCTV 모니터링
    # ------------------------------------------
    def show_cctv(self):
        print("[DEBUG] show_cctv 버튼 클릭됨!", flush=True)
        # [CHECK] [FIX 2] 이미 CCTV 활성 상태면: 먼저 stop_event 세팅 후 딜레이로 재진입
        # stop_cctv()를 여기서 직접 호출하면 join()이 메인 스레드를 블로킹하므로
        # after()로 딜레이 후 _show_cctv_inner 호출
        if getattr(self, 'cctv_active', False):
            self.cctv_active = False
            self._cctv_stop_event.set()
            self.safe_after(400, self._show_cctv_inner)
            return
        self._show_cctv_inner()

    def _show_cctv_inner(self):
        """show_cctv 실제 구현 — 항상 메인 스레드에서 호출됨"""
        # [CHECK] [FIX 2] 이전 세션 스레드가 완전히 종료될 때까지 대기
        for t in self._cctv_threads:
            t.join(timeout=3.0)
            if t.is_alive():
                print(f"[WARNING] 이전 CCTV 워커 아직 실행 중: {t.name}")
        self._cctv_threads = []

        # [CHECK] [FIX 1] clear_view에서 스캐너 타이머도 취소됨
        self.clear_view()

        # [CHECK] [FIX 2] 새 세션용 stop_event 생성 (이전 세션과 분리)
        self._cctv_stop_event = threading.Event()
        self.cctv_active = True

        with self.cctv_frame_lock:
            self.cctv_latest_frame = None
        with self.cctv_detection_lock:
            self.cctv_detection_cache = {'boxes': [], 'pollutant_detected': False, 'contamination_ratio': 0.0, 'lens_covered': False}
        self._cctv_tk_img_ref = None

        # FOV 상태 초기화
        self.fov_ref_saved = False
        self.fov_stable_count = 0
        self.fov_correction_active = False
        self.total_dx = 0.0
        self.total_dy = 0.0
        self.fov_state = "IDLE"
        self.fov_start_lr = self.servo_center_lr
        self.fov_start_ud = self.servo_center_ud
        self.fov_current_lr = self.servo_center_lr
        self.fov_current_ud = self.servo_center_ud
        self.fov_prev_gray = None
        self.fov_last_move_time = time.time()
        self.fov_cooldown_start = 0.0
        self.servo_target_lr = self.servo_center_lr
        self.servo_target_ud = self.servo_center_ud

        # UI 구성
        cv = ctk.CTkFrame(self.container, fg_color=COLOR_BG)
        cv.pack(fill="both", expand=True)
        h = ctk.CTkFrame(cv, fg_color="transparent")
        h.pack(fill="x", padx=self.scale(35), pady=(self.scale(15), 0))
        t_box = ctk.CTkFrame(h, fg_color="transparent")
        t_box.pack(side="left")
        ctk.CTkLabel(t_box, text="Live CCTV Monitoring", font=self.get_font(24, "bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w")
        ctk.CTkLabel(t_box, text="ACTIVE SYSTEM: SCANNING MULTIPLE CHANNELS", font=self.get_font(12), text_color=COLOR_ACCENT).pack(anchor="w")
        ctk.CTkButton(h, text="[BACK] BACK TO DASHBOARD", font=self.get_font(12, "bold"), fg_color="#334155",
            width=self.scale(160), height=self.scale(38), corner_radius=self.scale(10),
            command=self.show_dashboard).pack(side="right")

        grid = ctk.CTkFrame(cv, fg_color="transparent")
        grid.pack(fill="both", expand=True, padx=self.scale(35), pady=self.scale(15))
        grid.grid_columnconfigure(0, weight=4)
        grid.grid_columnconfigure(1, weight=1, minsize=self.scale(350))
        grid.grid_rowconfigure(0, weight=3)
        grid.grid_rowconfigure(1, weight=1)

        self.cctv_monitor = ctk.CTkFrame(grid, fg_color="#000000", corner_radius=self.scale(20), border_width=1, border_color="#334155")
        self.cctv_monitor.grid(row=0, column=0, sticky="nsew", padx=(0, self.scale(20)), pady=(0, self.scale(20)))
        self.cctv_lbl = ctk.CTkLabel(self.cctv_monitor, text="[CAM] CONNECTING TO WEBCAM...", font=self.get_font(18), text_color=COLOR_TEXT_SUB)
        self.cctv_lbl.pack(expand=True, fill="both")

        self.cctv_sidebar = ctk.CTkFrame(grid, fg_color="transparent")
        self.cctv_sidebar.grid(row=0, column=1, sticky="nsew", pady=(0, self.scale(15)))

        # ── 사이드바 헤더 ────────────────────────────────────────────────
        hdr_card = ctk.CTkFrame(self.cctv_sidebar, fg_color="#0f1f38",
                                corner_radius=self.scale(14), border_width=1, border_color="#1e4080")
        hdr_card.pack(fill="x", pady=(0, self.scale(6)))
        ctk.CTkLabel(hdr_card, text="LIVE SYSTEM STATUS",
                     font=self.get_font(12, "bold"), text_color="#60a5fa").pack(anchor="w", padx=14, pady=(10, 0))
        hw_tag = "HW CONNECTED" if self.hardware_connected else "SW SIMULATION"
        hw_col = COLOR_SUCCESS if self.hardware_connected else "#f59e0b"
        ctk.CTkLabel(hdr_card, text=hw_tag, font=self.get_font(10), text_color=hw_col).pack(anchor="w", padx=14, pady=(0, 10))

        # ── 상태 정보 카드 ───────────────────────────────────────────────
        def make_status_card(parent, title, value, title_color, value_color):
            c = ctk.CTkFrame(parent, fg_color=COLOR_CARD, corner_radius=self.scale(12),
                             border_width=1, border_color="#1e293b")
            c.pack(fill="x", pady=self.scale(2))
            row = ctk.CTkFrame(c, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=6)
            ctk.CTkLabel(row, text=title, font=self.get_font(10, "bold"), text_color=title_color).pack(side="left")
            ctk.CTkLabel(row, text=value, font=self.get_font(10, "bold"), text_color=value_color).pack(side="right")
            return c

        make_status_card(self.cctv_sidebar, "AI Engine", "best.pt  ACTIVE", "#94a3b8", COLOR_SUCCESS)
        make_status_card(self.cctv_sidebar, "Inference Mode", "CPU | conf>=0.25", "#94a3b8", "#60a5fa")
        make_status_card(self.cctv_sidebar, "Wiper Recovery", "12s CHECK CYCLE", "#94a3b8", "#a78bfa")
        make_status_card(self.cctv_sidebar, "Persist Log", "DB: lens_failures", "#94a3b8", "#f59e0b")

        # ── 렌즈 오염 상태 카드 (핵심) ──────────────────────────────────
        self.contamination_card = ctk.CTkFrame(
            self.cctv_sidebar, fg_color=COLOR_CARD,
            corner_radius=self.scale(14), border_width=2, border_color=COLOR_SUCCESS)
        self.contamination_card.pack(fill="x", pady=(self.scale(8), self.scale(2)))

        cont_hdr = ctk.CTkFrame(self.contamination_card, fg_color="transparent")
        cont_hdr.pack(fill="x", padx=12, pady=(8, 0))
        ctk.CTkLabel(cont_hdr, text="LENS STATUS",
                     font=self.get_font(11, "bold"), text_color=COLOR_TEXT_SUB).pack(side="left")
        self.cont_ratio_label = ctk.CTkLabel(
            cont_hdr, text="0.0% [OK]", font=self.get_font(12, "bold"), text_color=COLOR_SUCCESS)
        self.cont_ratio_label.pack(side="right")

        self.cont_bar = ctk.CTkProgressBar(
            self.contamination_card, height=self.scale(7),
            fg_color="#1e293b", progress_color=COLOR_SUCCESS)
        self.cont_bar.pack(fill="x", padx=10, pady=(4, 2))
        self.cont_bar.set(0.0)

        ctk.CTkLabel(self.contamination_card,
                     text="[OK] Wiper: Standby",
                     font=self.get_font(9), text_color="#64748b").pack(anchor="w", padx=12, pady=(0, 8))



        ctk.CTkLabel(self.cctv_sidebar, text="HARDWARE OVERRIDE", font=self.get_font(13, "bold"), text_color=COLOR_TEXT_SUB).pack(anchor="w", pady=(self.scale(10), self.scale(5)))
        controls_frame = ctk.CTkFrame(self.cctv_sidebar, fg_color="transparent")
        controls_frame.pack(fill="x", pady=self.scale(5))
        controls_frame.grid_columnconfigure((0, 1, 2), weight=1)

        def make_ccard(text, icon, color, cmd, row, col):
            card = ctk.CTkFrame(controls_frame, fg_color="#1e293b", corner_radius=self.scale(12), border_width=2, border_color=color, cursor="hand2")
            card.grid(row=row, column=col, padx=self.scale(4), pady=self.scale(4), sticky="nsew")
            i_lbl = ctk.CTkLabel(card, text=icon, font=self.get_font(20))
            i_lbl.pack(pady=(self.scale(8), 0))
            t_lbl = ctk.CTkLabel(card, text=text, font=self.get_font(12, "bold"), text_color="#FFFFFF")
            t_lbl.pack(pady=(0, self.scale(8)))
            card.bind("<Button-1>", cmd)
            i_lbl.bind("<Button-1>", cmd)
            t_lbl.bind("<Button-1>", cmd)
            return card, i_lbl, t_lbl

        make_ccard("Lens Wipe", "[WIPE]", COLOR_ACCENT, lambda e=None: self.trigger_lens_wipe(), 0, 0)
        make_ccard("Capture", "[CAP]", COLOR_TEXT_MAIN, lambda e=None: self.handle_telegram_command("/capture"), 0, 1)
        make_ccard("Siren", "[ALARM]", COLOR_DANGER, lambda e=None: self.handle_telegram_command("/alarm"), 0, 2)

        ai_card, ai_icon, ai_text = make_ccard("Pause AI", "[PAUSE]", COLOR_WARNING, lambda e=None: None, 1, 0)
        make_ccard("Set Origin", "[SET]", COLOR_SUCCESS, lambda e=None: self.fov_reset_origin(), 1, 1)
        fov_card, fov_icon, fov_text = make_ccard("FOV Auto", "[AUTO]", "#8b5cf6", lambda e=None: None, 1, 2)

        def toggle_fov(e=None):
            self.fov_recovery_enabled = not self.fov_recovery_enabled
            if self.fov_recovery_enabled:
                fov_icon.configure(text="[AUTO]"); fov_text.configure(text="FOV Auto"); fov_card.configure(border_color="#8b5cf6")
                self.add_cctv_log("[SETTINGS] [화각복구] 자동 복구 활성화")
            else:
                fov_icon.configure(text="[OFF]"); fov_text.configure(text="FOV Off"); fov_card.configure(border_color="#64748b")
                self.add_cctv_log("[STOP] [화각복구] 자동 복구 비활성화")
        fov_card.bind("<Button-1>", toggle_fov)
        fov_icon.bind("<Button-1>", toggle_fov)
        fov_text.bind("<Button-1>", toggle_fov)

        def toggle_ai(e=None):
            self.cctv_paused = not getattr(self, 'cctv_paused', False)
            if self.cctv_paused:
                ai_icon.configure(text="[RUN]"); ai_text.configure(text="Resume AI"); ai_card.configure(border_color=COLOR_SUCCESS)
            else:
                ai_icon.configure(text="[PAUSE]"); ai_text.configure(text="Pause AI"); ai_card.configure(border_color=COLOR_WARNING)
        ai_card.bind("<Button-1>", toggle_ai)
        ai_icon.bind("<Button-1>", toggle_ai)
        ai_text.bind("<Button-1>", toggle_ai)

        ctk.CTkButton(self.cctv_sidebar, text="[HOME] 메인 대시보드", font=self.get_font(14, "bold"),
            fg_color="#334155", height=self.scale(40), corner_radius=self.scale(8),
            command=self.show_dashboard).pack(fill="x", pady=self.scale(8))

        log_frame = ctk.CTkFrame(grid, fg_color=COLOR_CARD, corner_radius=self.scale(20), border_width=1, border_color="#334155")
        log_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        ctk.CTkLabel(log_frame, text="SYSTEM EVENT LOGS", font=self.get_font(14, "bold"), text_color=COLOR_ACCENT).pack(anchor="w", padx=25, pady=(15, 5))
        self.cctv_log = ctk.CTkTextbox(log_frame, font=("Courier New", 14), fg_color="#0b1322",
            text_color="#10b981", corner_radius=15, spacing3=5)
        self.cctv_log.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self.add_cctv_log("SYSTEM START: Initializing secure feed link...")
        hw_status = "HW 연결됨" if self.hardware_connected else "SW 전용"
        self.add_cctv_log(f"[MAINT] [화각복구] 자동 복구: {'ON' if self.fov_recovery_enabled else 'OFF'} | 서보: {hw_status}")
        self.add_cctv_log("[CAM] 카메라 초기화 중... (cctv_capture_worker 스레드에서 오픈)")

        # [OK] [FIX 3+4] 카메라 오픈은 cctv_capture_worker 스레드 내부에서 수행
        # 메인 스레드에서 열고 백그라운드에서 read()하는 V4L2 스레드 불안전 패턴 제거
        t1 = threading.Thread(target=self.cctv_capture_worker, daemon=True, name="cctv_capture")
        t2 = threading.Thread(target=self.cctv_inference_worker, daemon=True, name="cctv_inference")
        t3 = threading.Thread(target=self.telegram_livestream_worker, daemon=True, name="cctv_telegram")
        self._cctv_threads = [t1, t2]  # telegram은 join 대상 제외 (네트워크 지연 있음)
        t1.start()
        t2.start()
        t3.start()

        self.safe_after(30, self.cctv_render_loop)

    def trigger_lens_wipe(self, auto=False):
        if not auto:
            self.add_cctv_log("[PROCESS] MANUAL OVERRIDE: Lens Wiper Activated")
        else:
            self.add_cctv_log("[PROCESS] AUTO OVERRIDE: 렌즈 오염 감지로 와이퍼 구동")

        self.add_cctv_log("[PROCESS] Executing cleaning cycle sequence...")

        # [FIX 3] 와이퍼 구동 중 렌즈 오염 탐지 로직 2초간 일시 정지 (Debounce)
        self.wiper_active_until = time.time() + 2.0

        # [NEW] 자동 구동 시 12초 후 복구 확인 예약
        if auto:
            self.wiper_recovery_pending = True
            self.recovery_failed = False
            self.safe_after(12000, self.check_wiper_recovery)

        if self.hardware_connected and self.servo_wiper:
            def wipe_cycle():
                try:
                    with self.servo_lock:
                        self.servo_wiper.angle = 0
                    time.sleep(0.5)
                    with self.servo_lock:
                        self.servo_wiper.angle = 180
                    time.sleep(0.5)
                    with self.servo_lock:
                        self.servo_wiper.angle = 0
                    time.sleep(0.5)
                    with self.servo_lock:
                        self.servo_wiper.angle = 90
                    self.safe_after(0, lambda: self.add_cctv_log("[OK] 렌즈 와이퍼 클리닝 완료"))
                except Exception as e:
                    self.safe_after(0, lambda: self.add_cctv_log(f"[WARNING] 와이퍼 오류: {e}"))
            threading.Thread(target=wipe_cycle, daemon=True).start()
        else:
            self.add_cctv_log("[WARNING] 서보 하드웨어 미연결 (시뮬레이션 모드)")

    def check_wiper_recovery(self):
        """[NEW] 와이퍼 구동 후 오염 복구 여부 확인 — 실패 시 텔레그램 + UI 경고"""
        if not self.cctv_active or self.is_destroyed:
            self.wiper_recovery_pending = False
            return
        if not getattr(self, 'wiper_recovery_pending', False):
            return

        self.wiper_recovery_pending = False

        with self.cctv_detection_lock:
            cache = self.cctv_detection_cache.copy()

        still_polluted = cache.get('pollutant_detected', False)
        still_covered = cache.get('lens_covered', False)

        if still_polluted or still_covered:
            # ── 복구 실패 처리 ────────────────────────────────────────────
            self.recovery_failed = True
            self.recovery_fail_time = time.time()
            fail_reason = "물체 오염" if still_polluted else "렌즈 가림/오염"
            ratio = cache.get('contamination_ratio', 0.0)

            self.add_cctv_log(
                f"[ALARM] *** 복구 실패 *** 와이퍼 작동 후에도 {fail_reason} 지속! ({ratio:.1f}%)")

            # 텔레그램 경고 전송
            self.send_telegram_alert(
                f"[ALERT] [복구 실패] 와이퍼 작동 후에도 렌즈 {fail_reason}이 지속됩니다!\n"
                f"오염 비율: {ratio:.1f}%\n"
                f"즉시 현장을 확인하고 수동으로 렌즈를 닦아주세요."
            )

            # [NEW] 복구 실패 시 증거 사진 캡처 및 갤러리 로깅
            with self.cctv_frame_lock:
                fail_frame = self.cctv_latest_frame.copy() if hasattr(self, 'cctv_latest_frame') and self.cctv_latest_frame is not None else None
            
            if fail_frame is not None:
                try:
                    img_path = self.save_evidence(fail_frame, "recovery_failed")
                    # weapon_type을 "복구 실패"로 저장하여 갤러리에서 강조되도록 함
                    self.log_security_event("복구 실패", ratio, img_path, 1)
                except Exception as e:
                    print(f"[ERROR] 복구 실패 이미지 저장 오류: {e}")


            # 2차 와이퍼 재시도
            self.add_cctv_log("[PROCESS] 복구 재시도 중 (2차 와이퍼 구동)...")
            if self.hardware_connected and self.servo_wiper:
                def retry_wipe():
                    try:
                        with self.servo_lock:
                            self.servo_wiper.angle = 0
                        time.sleep(0.4)
                        with self.servo_lock:
                            self.servo_wiper.angle = 180
                        time.sleep(0.4)
                        with self.servo_lock:
                            self.servo_wiper.angle = 90
                    except Exception as e:
                        print(f"[WARNING] 2차 와이퍼 오류: {e}")
                threading.Thread(target=retry_wipe, daemon=True).start()
        else:
            # ── 복구 성공 처리 ────────────────────────────────────────────
            self.recovery_failed = False
            self.add_cctv_log("[OK] 복구 성공: 와이퍼 작동 후 렌즈 오염이 해소되었습니다.")
            self.send_telegram_alert("[OK] [복구 성공] 와이퍼 작동 후 렌즈 상태가 정상으로 복구되었습니다.")


    def add_cctv_log(self, msg):
        if not hasattr(self, 'cctv_log') or not self.cctv_log.winfo_exists():
            return
        ts = datetime.now().strftime("%H:%M:%S")
        self.cctv_log.insert(tk.END, f"[{ts}] {msg}\n")
        self.cctv_log.see(tk.END)

    def sync_cctv_ui(self, frame, contamination_ratio=0.0, pollutant_detected=False, is_covered=False):
        if not self.cctv_active or not hasattr(self, 'cctv_lbl') or not self.cctv_lbl.winfo_exists():
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            w = max(100, self.cctv_lbl.winfo_width())
            h = max(100, self.cctv_lbl.winfo_height())
            self._cctv_tk_img_ref = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(w, h))
            self.cctv_lbl.configure(image=self._cctv_tk_img_ref, text="")

            if hasattr(self, 'cont_ratio_label') and self.cont_ratio_label.winfo_exists():
                bar_val = min(contamination_ratio / 100.0, 1.0)
                # [NEW] 복구 실패 상태 최우선 표시
                if getattr(self, 'recovery_failed', False):
                    self.contamination_card.configure(border_color=COLOR_DANGER)
                    self.cont_ratio_label.configure(
                        text=f"{contamination_ratio:.1f}% [복구 실패!]", text_color=COLOR_DANGER)
                    self.cont_bar.configure(progress_color=COLOR_DANGER)
                    
                    if not getattr(self, 'recovery_popup_shown', False):
                        self.recovery_popup_shown = True
                        self.safe_after(0, self.show_recovery_warning_popup)
                else:
                    self.recovery_popup_shown = False
                    if pollutant_detected and contamination_ratio > 0:
                        level = "심각" if contamination_ratio > 30 else "경고" if contamination_ratio > 10 else "주의"
                        self.contamination_card.configure(border_color="#FFD700")
                        self.cont_ratio_label.configure(text=f"{contamination_ratio:.1f}% [WARN][{level}]", text_color="#FFD700")
                        self.cont_bar.configure(progress_color="#FFD700")
                    elif is_covered:
                        level = "심각" if contamination_ratio > 75 else "경고" if contamination_ratio > 48 else "주의"
                        self.contamination_card.configure(border_color="#FFD700")
                        self.cont_ratio_label.configure(text=f"{contamination_ratio:.1f}% [WARN][{level}]", text_color="#FFD700")
                        self.cont_bar.configure(progress_color="#FFD700")
                    else:
                        self.contamination_card.configure(border_color=COLOR_SUCCESS)
                        self.cont_ratio_label.configure(text="0.0% [OK]", text_color=COLOR_SUCCESS)
                        self.cont_bar.configure(progress_color=COLOR_SUCCESS)
                self.cont_bar.set(bar_val)
        except:
            pass

    def show_recovery_warning_popup(self):
        """[NEW] 복구 실패 시 팝업 경고 창 생성"""
        if getattr(self, 'recovery_popup', None) and self.recovery_popup.winfo_exists():
            return
        self.recovery_popup = ctk.CTkToplevel(self)
        self.recovery_popup.title("LENS RECOVERY FAILED")
        self.recovery_popup.geometry("450x260")
        self.recovery_popup.attributes("-topmost", True)
        self.recovery_popup.configure(fg_color="#450a0a")
        
        self.recovery_popup.update_idletasks()
        w = self.recovery_popup.winfo_width()
        h = self.recovery_popup.winfo_height()
        x = (self.recovery_popup.winfo_screenwidth() // 2) - (w // 2)
        y = (self.recovery_popup.winfo_screenheight() // 2) - (h // 2)
        self.recovery_popup.geometry(f"+{x}+{y}")

        ctk.CTkLabel(self.recovery_popup, text="[CRITICAL WARNING]", font=self.get_font(20, "bold"), text_color="#f87171").pack(pady=(20, 10))
        ctk.CTkLabel(self.recovery_popup, text="렌즈 복구가 실패했습니다.\n오염 또는 가림이 계속되고 있습니다.\n\n즉시 현장을 방문하여\n수동으로 렌즈를 점검하고 닦아주세요.", font=self.get_font(14), text_color="#fca5a5", justify="center").pack(pady=10)
        ctk.CTkButton(self.recovery_popup, text="확인 (닫기)", font=self.get_font(14, "bold"), fg_color="#b91c1c", hover_color="#991b1b", command=self.recovery_popup.destroy).pack(pady=15)


    # =========================================================================
    # [OK] [FIX 3] cctv_capture_worker: 카메라 오픈/읽기/닫기를 모두 이 스레드에서
    # V4L2는 스레드 비안전 — 오픈한 스레드에서만 read()해야 세그폴트 없음
    # =========================================================================
    def cctv_capture_worker(self):
        print("[DEBUG] cctv_capture_worker 시작 (카메라 오픈 포함)", flush=True)
        stop_ev = self._cctv_stop_event  # 현재 세션의 stop_event 로컬 참조

        # [OK] 카메라를 이 스레드에서 직접 열기
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
            backend = cap.getBackendName()
            self.cap = cap
            print(f"[OK] [cctv_capture_worker] 카메라 오픈 성공 (Backend: {backend})")
            self.safe_after(0, lambda: self.add_cctv_log(f"[OK] 카메라 연결 성공 (Backend: {backend})"))
        else:
            print("[FAIL] [cctv_capture_worker] 웹캠 연결 실패")
            self.safe_after(0, lambda: self.add_cctv_log("[FAIL] Webcam Not Detected"))
            return

        # [OK] 읽기 루프: stop_ev 또는 cctv_active가 False가 될 때까지
        while not stop_ev.is_set() and self.cctv_active and not self.is_destroyed:
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    with self.cctv_frame_lock:
                        self.cctv_latest_frame = frame.copy()
                else:
                    time.sleep(0.005)
            else:
                time.sleep(0.01)

        # [OK] 루프 종료 후 이 스레드에서 직접 release
        try:
            cap.release()
            print("[CAM] [cctv_capture_worker] 카메라 release 완료")
        except Exception as e:
            print(f"[WARNING] [cctv_capture_worker] release 오류: {e}")
        self.cap = None
        print("[DEBUG] cctv_capture_worker 종료", flush=True)

    # =========================================================================
    # [OK] [FIX 4] cctv_inference_worker: best.pt 단독 사용 (프린트 출력물 탐지 대비, conf=0.25)
    # =========================================================================
    def cctv_inference_worker(self):
        print("[DEBUG] cctv_inference_worker 시작 (best.pt 단독 모드)")
        stop_ev = self._cctv_stop_event  # 현재 세션의 stop_event 로컬 참조
        dev = 0 if yolo_on_cuda else 'cpu'

        # [OK] best.pt 워밍업 (메인 스레드에서 이미 완료되었으므로 생략 가능하나 안전하게 재확인)
        if yolo_model is not None:
            try:
                print("[DEBUG] CCTV worker best.pt 워밍업 확인...")
                _dummy = np.zeros((416, 416, 3), dtype=np.uint8)
                with torch.inference_mode():
                    with yolo_inference_lock:
                        _ = yolo_model(_dummy, verbose=False, device=dev)
                print("[OK] CCTV best.pt 워밍업 완료")
            except Exception as e:
                print(f"[WARNING] CCTV best.pt 워밍업 실패: {e}")

        last_log_t = time.time()
        DARKNESS_THRESHOLD = 40
        BLUR_THRESHOLD = 15
        frame_count = 0

        while not stop_ev.is_set() and self.cctv_active and not self.is_destroyed:
            # AI 추론 초당 ~2회로 제한 (세그폴트·OOM 방지)
            time.sleep(0.5)

            if stop_ev.is_set() or not self.cctv_active:
                break

            with self.cctv_frame_lock:
                if self.cctv_latest_frame is None:
                    continue
                frame_copy = self.cctv_latest_frame.copy()

            if getattr(self, 'cctv_paused', False) or time.time() < getattr(self, 'wiper_active_until', 0.0):
                continue

            if frame_count == 0:
                print("[DEBUG] 첫 프레임 추론 루프 진입")
            frame_count += 1

            # ── 렌즈 상태 분석 (블러/암도 기반) ────────────────────────
            gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
            dark_pixel_ratio = np.sum(gray < 25) / (gray.shape[0] * gray.shape[1])
            h_g, w_g = gray.shape
            bh, bw = h_g // 5, w_g // 5
            flat_blocks = 0
            flat_block_coords = []

            for r in range(5):
                for c in range(5):
                    block = gray[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
                    block_var = cv2.Laplacian(block, cv2.CV_64F).var()
                    if block_var < 35.0:
                        flat_blocks += 1
                        fx1, fy1 = c*bw, r*bh
                        flat_block_coords.append((fx1, fy1, min((c+1)*bw, w_g-1), min((r+1)*bh, h_g-1)))

            is_covered = (
                (np.mean(gray) < DARKNESS_THRESHOLD) or
                (cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD) or
                (dark_pixel_ratio > 0.3) or
                (flat_blocks >= 12)
            )

            # ── best.pt 단독 추론 (conf=0.25, imgsz=416) ────────────────
            # 프린트 출력물 탐지를 위해 임계값을 낮게 설정
            new_boxes = []
            pollutant_det = False
            contamination_ratio = 0.0
            pollutant_mask = np.zeros((h_g, w_g), dtype=np.uint8)

            # stop_event 재확인: 추론 진입 직전 종료 신호 체크
            if stop_ev.is_set() or not self.cctv_active:
                break

            if yolo_model is not None:
                try:
                    with torch.inference_mode():
                        with yolo_inference_lock:
                            res1 = yolo_model(frame_copy, conf=0.25, imgsz=416,
                                              verbose=False, device=dev)[0]
                    for box in res1.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        conf_v = box.conf[0].item()
                        label = yolo_model.names[cls]

                        if label in ('pollutant', 'contamination'):
                            # 표 6-4 조건부 충족: 설정 화면의 AI 신뢰도 임계값과 연동
                            if conf_v < getattr(self, 'settings_conf_threshold', 0.40):
                                continue
                            box_region = gray[max(0, y1):min(h_g, y2), max(0, x1):min(w_g, x2)]
                            box_blur = cv2.Laplacian(box_region, cv2.CV_64F).var() if box_region.size > 0 else 999.0
                            box_area_ratio = ((x2 - x1) * (y2 - y1)) / (h_g * w_g)
                            
                            # 5장 기능 구현(5.2.2): 블러 수준(Laplacian Variance < 80) 및 점유 면적(> 5%) 복합 조건
                            if box_blur < 80.0 and box_area_ratio > 0.05:
                                pollutant_det = True
                                pollutant_mask[y1:y2, x1:x2] = 1
                                new_boxes.append((x1, y1, x2, y2, label, conf_v))
                        else:
                            # 소 등 일반 객체: 0.25 이상이면 추가
                            new_boxes.append((x1, y1, x2, y2, label, conf_v))
                except Exception:
                    pass

            if pollutant_det:
                contamination_ratio = (np.sum(pollutant_mask) / (h_g * w_g)) * 100
            if is_covered:
                contamination_ratio = (len(flat_block_coords) / 25.0) * 100.0

            with self.cctv_detection_lock:
                self.cctv_detection_cache = {
                    'boxes': new_boxes,
                    'pollutant_detected': pollutant_det,
                    'lens_covered': is_covered,
                    'contamination_ratio': contamination_ratio,
                    'flat_block_coords': flat_block_coords,
                    'is_turned': self.cctv_detection_cache.get('is_turned', False),
                    'last_move_dir': self.cctv_detection_cache.get('last_move_dir', 'UNKNOWN'),
                }

            if time.time() - last_log_t > 4.0:
                if pollutant_det:
                    msg = f"!!! ALERT: Pollutant detected on lens [{len([b for b in new_boxes if b[4] in ('pollutant','contamination')])} box] !!!"
                elif is_covered:
                    msg = "!!! WARNING: Lens obstructed / contaminated !!!"
                elif len(new_boxes) > 0:
                    msg = f"AI DETECTION: {len(new_boxes)} targets identified (best.pt, conf>=0.25)."
                else:
                    msg = "SYSTEM: Scanning environment... (best.pt)"
                self.safe_after(0, lambda m=msg: self.add_cctv_log(m))
                last_log_t = time.time()

        print("[DEBUG] cctv_inference_worker 종료")


    # =========================================================================
    # CCTV 렌더 루프 (메인 스레드 — 30ms 주기)
    # =========================================================================
    def cctv_render_loop(self):
        if self.is_destroyed or not self.cctv_active:
            return
        if not hasattr(self, 'cctv_lbl') or not self.cctv_lbl.winfo_exists():
            return

        with self.cctv_frame_lock:
            if self.cctv_latest_frame is None:
                self.safe_after(30, self.cctv_render_loop)
                return
            render_frame = self.cctv_latest_frame.copy()

        if getattr(self, 'cctv_paused', False):
            cv2.putText(render_frame, "AI ANALYSIS PAUSED", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
            self.sync_cctv_ui(render_frame)
            self.safe_after(30, self.cctv_render_loop)
            return

        # 광학흐름 계산 및 FOV 상태머신
        gray = cv2.cvtColor(render_frame, cv2.COLOR_BGR2GRAY)
        fov_st = getattr(self, 'fov_state', 'IDLE')

        # RECOVERING / COOLDOWN 상태: 렌더링만 수행 후 조기 리턴 (모터 구동/반동 중 연산 차단)
        if fov_st in ('RECOVERING', 'COOLDOWN'):
            self.fov_prev_gray = gray
            label_color = (0, 0, 255) if fov_st == 'RECOVERING' else (255, 100, 0)
            
            if fov_st == 'COOLDOWN':
                passed_time = time.time() - getattr(self, 'fov_cooldown_start', time.time())
                cv2.putText(render_frame, f"STATE: COOLDOWN ({2.0 - passed_time:.1f}s)", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
                if passed_time > 2.0:
                    self.fov_state = "IDLE"
                    self.fov_prev_gray = None
                    self.safe_after(0, lambda: self.add_cctv_log("[CHECK] [FOV 상태머신] 안정화 완료! 다음 독립 충격 대기 중"))
            else:
                cv2.putText(render_frame, f"STATE: {fov_st}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
                
            with self.cctv_detection_lock:
                cache = self.cctv_detection_cache.copy()
            self._render_detections(render_frame, cache)
            self.sync_cctv_ui(render_frame,
                              cache.get('contamination_ratio', 0.0),
                              cache.get('pollutant_detected', False),
                              cache.get('lens_covered', False))
            self.safe_after(30, self.cctv_render_loop)
            return

        # IDLE / SHOCK: 광학흐름 분석 (이전 프레임 있을 때만)
        cv2.putText(render_frame, f"STATE: {fov_st}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        fov_prev = getattr(self, 'fov_prev_gray', None)
        
        # [FIX 3] 와이퍼 가동 중에는 광학흐름 연산도 일시정지 (흔들림으로 오인 방지)
        if fov_prev is not None and self.fov_recovery_enabled and time.time() >= getattr(self, 'wiper_active_until', 0.0):
            try:
                p0 = cv2.goodFeaturesToTrack(fov_prev, maxCorners=50, qualityLevel=0.03, minDistance=10)
                if p0 is not None:
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(fov_prev, gray, p0, None)
                    if p1 is not None and st is not None:
                        good_new = p1[st == 1]
                        good_old = p0[st == 1]
                        if len(good_new) > 8:
                            M, _ = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
                            if M is not None:
                                dx, dy = M[0, 2], M[1, 2]
                                
                                # 민감도 0.8로 하향 조정하여 미세한 흔들림도 빠르게 감지
                                if abs(dx) > 0.8 or abs(dy) > 0.8:
                                    if self.fov_state == "IDLE":
                                        self.fov_state = "SHOCK"
                                        self.fov_start_lr = getattr(self, 'fov_current_lr', self.servo_center_lr)
                                        self.fov_start_ud = getattr(self, 'fov_current_ud', self.servo_center_ud)
                                        self.fov_correction_active = True
                                        self.safe_after(0, lambda: self.add_cctv_log(
                                            f"[ALARM] [FOV 상태머신] 충격 감지! 충격 전 각도 저장 (LR:{self.fov_start_lr:.1f}°, UD:{self.fov_start_ud:.1f}°)"))
                                        
                                    self.total_dx = getattr(self, 'total_dx', 0.0) + dx
                                    self.total_dy = getattr(self, 'total_dy', 0.0) + dy
                                    self.last_shock_time = time.time()
                                    self.last_move_dir = (
                                        "LEFT" if dx > 0 else
                                        "RIGHT" if abs(dx) > abs(dy) else
                                        "UP" if dy > 0 else "DOWN"
                                    )
            except Exception:
                pass

        self.fov_prev_gray = gray
        
        # 흔들림 멈추고 2초 후 원점 복구 트리거
        if self.fov_state == "SHOCK":
            quiet_duration = time.time() - getattr(self, 'last_shock_time', time.time())
            if quiet_duration > 2.0:
                self.fov_state = "RECOVERING"
                self.safe_after(0, lambda: self.add_cctv_log(
                    f"[TARGET] 충격 종료 확인! 기억해둔 원점으로 복구를 시작합니다. (누적 오차 X:{self.total_dx:.1f}, Y:{self.total_dy:.1f})"))
                self.safe_after(50, self.smooth_recovery_step)

        with self.cctv_detection_lock:
            cache = self.cctv_detection_cache.copy()
        current_time = time.time()

        for (x1, y1, x2, y2, label, conf) in cache.get('boxes', []):
            color = (0, 255, 0) if label == 'cow' else (0, 255, 255)
            cv2.rectangle(render_frame, (x1, y1), (x2, y2), color,
                          3 if label in ('pollutant', 'contamination') else 2)
            cv2.putText(render_frame, f"{label.upper()} {conf:.1%}", (x1, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0) if label in ('pollutant', 'contamination') else (255, 255, 255), 2)

        c_ratio = cache.get('contamination_ratio', 0.0)
        DEBOUNCE = getattr(self, 'ALARM_DEBOUNCE_SECS', 4.0)

        # ── AI 오염 감지 (pollutant) ─────────────────────────────────────
        PERSISTENT_SECS = 45.0  # 45초 지속 시 DB 기록
        is_polluted = cache.get('pollutant_detected', False)
        is_covered  = cache.get('lens_covered', False)
        any_issue   = is_polluted or is_covered

        if is_polluted:
            if self.pollutant_detect_start == 0.0:
                self.pollutant_detect_start = current_time
                self.safe_after(0, lambda m="[WARN] 렌즈 오염 감지 중... 지속 확인 중": self.add_cctv_log(m))
            elapsed = current_time - self.pollutant_detect_start
            self.last_event_t = current_time
            h_f, w_f = render_frame.shape[:2]

            if elapsed >= DEBOUNCE and not getattr(self, 'pollutant_logged', False):
                img_path = self.save_evidence(render_frame, "pollutant")
                self.log_security_event("물체 오염", c_ratio, img_path, 1)
                self.send_telegram_alert(
                    f"[WARN] 렌즈 오염 감지! 오염 비율: {c_ratio:.1f}%\n즉시 확인바랍니다.", img_path)
                self.pollutant_logged = True
                threading.Thread(target=lambda: self.trigger_lens_wipe(auto=True), daemon=True).start()
        else:
            if self.pollutant_detect_start > 0.0:
                self.pollutant_detect_start = 0.0

        # ── 렌즈 가림/블러 감지 (lens_covered) ──────────────────────────
        if is_covered:
            if self.alert_detect_start == 0.0:
                self.alert_detect_start = current_time
                self.covered_wiper_logged = False
                self.safe_after(0, lambda m="[WARN] 렌즈 가림/블러 감지 중... 지속 확인 중": self.add_cctv_log(m))
            elapsed = current_time - self.alert_detect_start
            self.last_event_t = current_time
            h_f, w_f = render_frame.shape[:2]

            if elapsed >= DEBOUNCE and not getattr(self, 'alert_logged', False):
                img_path = self.save_evidence(render_frame, "covered")
                self.log_security_event("렌즈 오염/가림", c_ratio, img_path, 1)
                self.send_telegram_alert(
                    f"[WARN] 렌즈 가림/오염 감지! 가림 비율: {c_ratio:.1f}%", img_path)
                self.alert_logged = True
                if not getattr(self, 'covered_wiper_logged', False):
                    self.covered_wiper_logged = True
                    threading.Thread(target=lambda: self.trigger_lens_wipe(auto=True), daemon=True).start()
                    self.safe_after(0, lambda: self.add_cctv_log("[PROCESS] 렌즈 가림 감지 — 와이퍼 자동 구동"))
        else:
            if self.alert_detect_start > 0.0:
                self.alert_detect_start = 0.0
                self.covered_wiper_logged = False

        # ── 지속 실패 DB 기록 (45초 이상 지속 시) ──────────────────────
        if any_issue:
            if self.persistent_issue_start == 0.0:
                self.persistent_issue_start = current_time
                self.persistent_logged = False
                self.persistent_issue_type = "물체 오염" if is_polluted else "렌즈 가림/블러"
            else:
                persist_elapsed = current_time - self.persistent_issue_start
                if persist_elapsed >= PERSISTENT_SECS and not self.persistent_logged:
                    self.persistent_logged = True
                    wiper_tried = getattr(self, 'pollutant_logged', False) or \
                                  getattr(self, 'covered_wiper_logged', False)
                    recovery = "복구 실패 (지속 중)" if getattr(self, 'recovery_failed', False) else "복구 시도 중"
                    # 백그라운드에서 DB 저장 (UI 블로킹 방지)
                    _etype = self.persistent_issue_type
                    _ratio = c_ratio
                    _dur   = persist_elapsed
                    threading.Thread(
                        target=lambda: self.log_lens_failure(
                            _etype, _ratio, _dur, wiper_tried, recovery),
                        daemon=True).start()
                    self.send_telegram_alert(
                        f"[ALERT] 렌즈 이상 장시간 지속!\n"
                        f"유형: {_etype}\n"
                        f"지속 시간: {_dur:.0f}초\n"
                        f"오염 비율: {_ratio:.1f}%\n"
                        f"와이퍼 시도: {'예' if wiper_tried else '아니오'}\n"
                        f"상태: {recovery}\n"
                        f"즉시 현장 점검이 필요합니다."
                    )
                    self.safe_after(0, lambda: self.add_cctv_log(
                        f"[DB] 지속 실패 기록 완료 — {_etype} {_dur:.0f}s | DB 저장됨"))
        else:
            # 이슈 해소 — 지속 실패 타이머 리셋
            if self.persistent_issue_start > 0.0:
                persist_elapsed = current_time - self.persistent_issue_start
                if self.persistent_logged:
                    _etype = self.persistent_issue_type
                    _ratio = c_ratio
                    _dur   = persist_elapsed
                    threading.Thread(
                        target=lambda: self.log_lens_failure(
                            _etype, _ratio, _dur, True, "복구 성공 (자동 해제)"),
                        daemon=True).start()
                self.persistent_issue_start = 0.0
                self.persistent_issue_type = ""
                self.persistent_logged = False

        # ── 카메라 움직임 감지 ───────────────────────────────────────────
        if cache.get('is_turned', False):
            if self.turned_detect_start == 0.0:
                self.turned_detect_start = current_time
                self.safe_after(0, lambda m="[WARN] 카메라 이동 감지 중...": self.add_cctv_log(m))
            elapsed = current_time - self.turned_detect_start
            self.last_event_t = current_time
            if elapsed >= DEBOUNCE and not getattr(self, 'alert_logged', False):
                img_path = self.save_evidence(render_frame, "camera_move")
                self.log_security_event("카메라 움직임", 0.0, img_path, 1)
                self.send_telegram_alert("[WARN] 카메라가 물리적으로 이동되었습니다!", img_path)
                self.alert_logged = True
        else:
            if self.turned_detect_start > 0.0:
                self.turned_detect_start = 0.0

        # ── 이벤트 초기화 (5초 정상 유지 시) ───────────────────────────
        if getattr(self, 'pollutant_logged', False) or getattr(self, 'alert_logged', False):
            if current_time - self.last_event_t > 5.0:
                self.pollutant_logged = self.alert_logged = False
                self.covered_wiper_logged = False
                self.pollutant_detect_start = 0.0
                self.alert_detect_start = 0.0
                self.turned_detect_start = 0.0
                self.safe_after(0, lambda m="[OK] 위험 요소 해제 — 렌즈 상태 정상": self.add_cctv_log(m))

        # ==================================================================
        # [OVERLAY] 단일 우선순위 배너 — 최대 하나만 표시
        # 우선순위: recovery_failed > pollutant > covered > camera
        # ==================================================================
        h_f, w_f = render_frame.shape[:2]
        BAN_H = int(h_f * 0.13)   # 배너 높이 (프레임 13%)
        BAR_H = 10                 # 하단 진행 바 높이

        # ── 상태 결정 ────────────────────────────────────────────────────
        is_recovery_fail  = getattr(self, 'recovery_failed', False)
        is_pol_active     = is_polluted and getattr(self, 'pollutant_logged', False)
        is_cov_active     = is_covered  and getattr(self, 'alert_logged', False)
        is_cam_active     = cache.get('is_turned', False) and getattr(self, 'alert_logged', False)

        # ── 단일 배너 그리기 ─────────────────────────────────────────────
        if is_recovery_fail:
            # 복구 실패 — 가장 높은 우선순위
            ov = render_frame.copy()
            cv2.rectangle(ov, (0, 0), (w_f, BAN_H), (0, 0, 160), -1)
            cv2.addWeighted(ov, 0.80, render_frame, 0.20, 0, render_frame)
            cv2.putText(render_frame, "WIPER RECOVERY FAILED",
                        (int(w_f * 0.04), int(BAN_H * 0.45)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.95, (80, 80, 255), 2, cv2.LINE_AA)
            cv2.putText(render_frame, f"Lens: {c_ratio:.1f}%  |  Manual cleaning required!",
                        (int(w_f * 0.04), int(BAN_H * 0.82)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1, cv2.LINE_AA)
            # 오염 해소 시 자동 해제
            if not is_polluted and not is_covered:
                fail_elapsed = current_time - getattr(self, 'recovery_fail_time', current_time)
                if fail_elapsed > 5.0:
                    self.recovery_failed = False
                    self.safe_after(0, lambda: self.add_cctv_log("[OK] 복구 실패 상태 해제 — 렌즈 정상 복귀"))

        elif is_pol_active:
            # 오염 물질 감지 확정
            ov = render_frame.copy()
            cv2.rectangle(ov, (0, 0), (w_f, BAN_H), (30, 30, 180), -1)
            cv2.addWeighted(ov, 0.75, render_frame, 0.25, 0, render_frame)
            cv2.putText(render_frame,
                        f"LENS FOULING ALERT: {c_ratio:.1f}%",
                        (int(w_f * 0.04), int(BAN_H * 0.55)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.95, (100, 100, 255), 2, cv2.LINE_AA)

        elif is_cov_active:
            # 렌즈 가림 감지 확정
            ov = render_frame.copy()
            cv2.rectangle(ov, (0, 0), (w_f, BAN_H), (30, 30, 180), -1)
            cv2.addWeighted(ov, 0.75, render_frame, 0.25, 0, render_frame)
            cv2.putText(render_frame,
                        f"LENS COVERED ALERT: {c_ratio:.1f}%",
                        (int(w_f * 0.04), int(BAN_H * 0.55)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.95, (100, 100, 255), 2, cv2.LINE_AA)

        elif is_cam_active:
            # 카메라 이동 확정
            ov = render_frame.copy()
            cv2.rectangle(ov, (0, 0), (w_f, BAN_H), (30, 30, 180), -1)
            cv2.addWeighted(ov, 0.75, render_frame, 0.25, 0, render_frame)
            cv2.putText(render_frame,
                        f"CAMERA MOVED: {cache.get('last_move_dir', 'UNKNOWN')}",
                        (int(w_f * 0.04), int(BAN_H * 0.55)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.95, (100, 100, 255), 2, cv2.LINE_AA)

        # ── 진행 바 및 상태 텍스트 (배너 미표시 시 or 디바운스 중) ──────
        # 오염/가림 감지 디바운스 중 (아직 logged 되기 전)
        if not is_pol_active and not is_cov_active and not is_cam_active and not is_recovery_fail:
            if is_polluted and self.pollutant_detect_start > 0.0:
                elapsed = current_time - self.pollutant_detect_start
                prog = min(elapsed / DEBOUNCE, 1.0)
                cv2.rectangle(render_frame, (0, h_f - BAR_H), (int(w_f * prog), h_f), (60, 60, 220), -1)
                cv2.putText(render_frame,
                            f"FOULING DETECTING: {c_ratio:.1f}%  [{elapsed:.1f}s / {DEBOUNCE:.0f}s]",
                            (18, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2)
            elif is_covered and self.alert_detect_start > 0.0:
                elapsed = current_time - self.alert_detect_start
                prog = min(elapsed / DEBOUNCE, 1.0)
                cv2.rectangle(render_frame, (0, h_f - BAR_H), (int(w_f * prog), h_f), (60, 60, 220), -1)
                cv2.putText(render_frame,
                            f"LENS COVERED: {c_ratio:.1f}%  [{elapsed:.1f}s / {DEBOUNCE:.0f}s]",
                            (18, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2)
        elif is_recovery_fail:
            # 복구 실패 중에도 하단 바 표시 (지속 시간)
            persist_dur = current_time - getattr(self, 'persistent_issue_start', current_time)
            if persist_dur > 0:
                cv2.rectangle(render_frame, (0, h_f - BAR_H), (w_f, h_f), (60, 60, 200), -1)



        # ── FOV 이동 화살표 ──────────────────────────────────────────────
        h_f, w_f = render_frame.shape[:2]
        dx_draw = getattr(self, 'total_dx', 0)
        dy_draw = getattr(self, 'total_dy', 0)
        move_px = (dx_draw**2 + dy_draw**2) ** 0.5

        if self.fov_correction_active:
            arrow_color = (0, 200, 255)
            cv2.putText(render_frame, f"FOV RECOVERY: {move_px:.1f}px", (20, h_f - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        elif self.fov_recovery_enabled and self.fov_ref_saved:
            arrow_color = (0, 255, 0)
        else:
            arrow_color = (255, 0, 255)

        max_arrow_len = 100
        dx_constrained = max(-max_arrow_len, min(max_arrow_len, dx_draw))
        dy_constrained = max(-max_arrow_len, min(max_arrow_len, dy_draw))
        cv2.arrowedLine(render_frame, (w_f//2, h_f//2),
                        (int(w_f//2 + dx_constrained), int(h_f//2 + dy_constrained)),
                        arrow_color, 3, tipLength=0.2)

        self.sync_cctv_ui(render_frame, c_ratio,
                          cache.get('pollutant_detected', False),
                          cache.get('lens_covered', False))
        self.safe_after(30, self.cctv_render_loop)

    def _render_detections(self, frame, cache):
        for (x1, y1, x2, y2, label, conf) in cache.get('boxes', []):
            color = (0, 255, 0) if label == 'cow' else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                          3 if label in ('pollutant', 'contamination') else 2)
            cv2.putText(frame, f"{label.upper()} {conf:.1%}", (x1, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0) if label in ('pollutant', 'contamination') else (255, 255, 255), 2)

    def smooth_recovery_step(self):
        if getattr(self, 'fov_state', 'IDLE') != 'RECOVERING':
            return
        if self.is_destroyed or not self.cctv_active:
            return

        fov_px_per_degree = 0.2
        start_lr = getattr(self, 'fov_start_lr', self.servo_center_lr)
        start_ud = getattr(self, 'fov_start_ud', self.servo_center_ud)
        target_lr = start_lr - (self.total_dx * fov_px_per_degree)
        target_ud = start_ud - (self.total_dy * fov_px_per_degree)
        target_lr = max(0, min(180, target_lr))
        target_ud = max(0, min(180, target_ud))

        step_size = 1.5
        cur_lr = getattr(self, 'fov_current_lr', self.servo_center_lr)
        cur_ud = getattr(self, 'fov_current_ud', self.servo_center_ud)
        done_lr = False
        done_ud = False

        if abs(cur_lr - target_lr) > step_size:
            cur_lr += step_size if cur_lr < target_lr else -step_size
        else:
            cur_lr = target_lr
            done_lr = True

        if abs(cur_ud - target_ud) > step_size:
            cur_ud += step_size if cur_ud < target_ud else -step_size
        else:
            cur_ud = target_ud
            done_ud = True

        self.fov_current_lr = cur_lr
        self.fov_current_ud = cur_ud

        if self.fov_recovery_enabled and self.hardware_connected:
            try:
                # 하드웨어 피드백 강화 - 별도 스레드에서 즉시 모터 이동 (우선순위 확보)
                threading.Thread(target=self.servo_move, kwargs={'lr_angle': cur_lr, 'ud_angle': cur_ud}, daemon=True).start()
                self.safe_after(0, lambda: self.add_cctv_log(
                    f"[PROCESS] 원점 복구 중.. 좌우: {cur_lr:.1f}°, 상하: {cur_ud:.1f}°"))
            except Exception as e:
                print(f"[WARNING] smooth_recovery_step 모터 인가 에러: {e}")

        if done_lr and done_ud:
            print("[OK] [성공] 기억해둔 최초 원점으로 정확히 복구 완료되었습니다!")
            self.total_dx = 0.0
            self.total_dy = 0.0
            self.fov_prev_gray = None
            self.fov_correction_active = False
            self.fov_state = "COOLDOWN"
            self.fov_cooldown_start = time.time()
            self.safe_after(0, lambda: self.add_cctv_log("[RECOVER] 모터 정지 반동 흡수 중 (COOLDOWN)..."))
        else:
            self.safe_after(40, self.smooth_recovery_step)

    def add_m_item(self, master, lbl, val):
        f = ctk.CTkFrame(master, fg_color="transparent")
        f.pack(fill="x", pady=self.scale(6))
        ctk.CTkLabel(f, text=lbl, font=self.get_font(14), text_color=COLOR_TEXT_SUB).pack(side="left")
        v = ctk.CTkLabel(f, text=val, font=self.get_font(16, "bold"), text_color=COLOR_TEXT_MAIN)
        v.pack(side="right")
        return v

    # ------------------------------------------
    # 5. Scanner 추론 (백그라운드 스레드)
    # ------------------------------------------
    def run_random_inference(self):
        self._scanner_timer = None  # 타이머 소비됨
        idir = r"02_Cattle_Dataset"
        imgs = [f for f in os.listdir(idir) if f.endswith(('.jpg', '.png'))]
        if not imgs:
            return
        self.current_img_path = os.path.join(idir, random.choice(imgs))
        if getattr(self, 'is_analyzing', False):
            return
        self.is_analyzing = True
        if hasattr(self, 'img_lbl') and self.img_lbl.winfo_exists():
            self.img_lbl.configure(text="[SYNC] SYNCING SCANNER WITH CLOUD AI...")
        if hasattr(self, 'rpt_box') and self.rpt_box.winfo_exists():
            self.rpt_box.delete("1.0", tk.END)
            self.rpt_box.insert(tk.END, "[WAIT] AI Diagnostic Engine is generating an official report...\n")

        print("[DEBUG] run_random_inference → 백그라운드 스레드 시작")
        # [OK] [FIX 5] worker를 백그라운드 스레드에서 실행
        # 기존: self.after(50, lambda: self.worker(...)) → 메인 스레드에서 YOLO 실행 → 세그폴트
        # 수정: threading.Thread → 백그라운드에서 YOLO 실행, UI 업데이트만 safe_after로 메인 스레드에
        threading.Thread(target=self.worker, args=(self.current_img_path,), daemon=True).start()

    def worker(self, path):
        """
        [OK] [FIX 5] 이 함수는 이제 백그라운드 스레드에서 실행됨.
        YOLO 추론과 res.plot()이 메인 스레드를 블로킹하지 않음.
        UI 업데이트는 모두 safe_after(0, ...) 를 통해 메인 스레드에서 수행.
        """
        print(f"[DEBUG] worker 시작 (백그라운드): {path}")
        start_t = time.time()
        try:
            imgsz = 640 if self.settings_mode == "High Precision" else 320
            print("[DEBUG] cv2.imread 호출 전")
            raw = cv2.imread(path)
            print(f"[DEBUG] cv2.imread 완료, raw shape: {raw.shape if raw is not None else 'None'}")
            if raw is None:
                raise FileNotFoundError(f"Load failed: {path}")

            # [FIX 1] 프린트 사진 인식 개선: CLAHE를 사용해 대비(Contrast) 전처리
            lab = cv2.cvtColor(raw, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            raw = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            img_h, img_w = raw.shape[:2]

            with torch.inference_mode():
                with yolo_inference_lock:
                    print(f"[DEBUG] YOLO 추론 시작 (imgsz={imgsz})")
                    res = yolo_model(raw, conf=self.settings_conf_threshold, imgsz=imgsz,
                                    verbose=False, device='cpu')[0]
                    print("[DEBUG] YOLO 추론 완료")

            all_boxes = res.boxes
            conf = all_boxes.conf.mean().item() if len(all_boxes) > 0 else 0.0
            if torch.is_tensor(conf):
                conf = conf.item()
            conf = float(conf)

            if self.settings_auto_save:
                try:
                    save_dir = "05_Detections"
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(save_dir, f"scan_{ts}_{self.settings_admin_name.replace(' ', '_')}.jpg")
                    print("[DEBUG] res.save 호출 전")
                    res.save(save_path)
                    print("[DEBUG] res.save 호출 완료")
                except:
                    pass

            current_counts = {"Standing": 0, "Sleeping": 0, "Eating": 0}
            cow_boxes = [b for b in all_boxes if yolo_model.names[int(b.cls[0])] == 'cow']
            count = len(cow_boxes)

            for box in cow_boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 1.0
                if aspect_ratio >= 1.2:
                    current_counts["Sleeping"] += 1
                else:
                    current_counts["Standing"] += 1

            for k, v in current_counts.items():
                self.behavior_counts[k] = self.behavior_counts.get(k, 0) + v

            if count > 0:
                active_bhvs = {k: v for k, v in current_counts.items() if v > 0}
                if active_bhvs:
                    max_val = max(active_bhvs.values())
                    top_bhvs = [k for k, v in active_bhvs.items() if v == max_val]
                    top_bhv = f"[{' & '.join(top_bhvs).upper()} : {max_val}]"
                else:
                    top_bhv = "[NONE]"
            else:
                top_bhv = "[COW NOT FOUND]"

            pols = [b for b in all_boxes if yolo_model.names[int(b.cls[0])] in ('pollutant', 'contamination') and float(b.conf[0]) >= 0.90]
            has_pollutant = len(pols) > 0
            pollutant_conf = max([float(b.conf[0]) for b in pols]) if has_pollutant else 0.0

            p_area = float(sum([(b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]) for b in pols])) if has_pollutant else 0.0
            clarity = max(0.0, min(1.0, float(1.0 - ((p_area / (img_h * img_w)) * 5))))

            self.stats_history.append({'conf': conf, 'count': count})
            # [ALARM] res.plot()은 폰트/그래픽 드라이버 접근 → 반드시 메인 스레드에서만 호출
            # 백그라운드 스레드에서는 res 객체만 메인 스레드로 전달
            print("[DEBUG] sync_ui 메인 스레드 위임 전")
            self.safe_after(0, lambda r=res, cc=current_counts, ct=count, cf=conf, cl=clarity, hp=has_pollutant, pc=pollutant_conf:
                self.sync_ui(r, cc, ct, cf, cl, hp, pc))

            def fetch_llm_report():
                print("[DEBUG] fetch_llm_report 호출 전")
                now_t = time.time()
                if getattr(self, 'cctv_active', False):
                    elapsed_nim = now_t - getattr(self, 'last_nim_call_t', 0.0)
                    if elapsed_nim < getattr(self, 'nim_min_interval', 30.0):
                        rpt = (f" VISION DOCTOR AI 진단 리포트\n"
                               f"▪ 관제 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                               f"▪ 객체 분석: {count}마리 감지 (CCTV 활성 중 - API 쿨다운 {int(getattr(self,'nim_min_interval',30)-elapsed_nim)}초 대기)\n"
                               f"▪ 환경 제어: {'오염 감지됨' if has_pollutant else '렌즈 정상'}\n"
                               f"▪ 종합 소견: CCTV 라이브 모니터링 중 NIM API 호출이 일시 제한됩니다.")
                    else:
                        self.last_nim_call_t = now_t
                        rpt = self.gen_korean_rpt(count, top_bhv, clarity, current_counts, has_pollutant, pollutant_conf)
                else:
                    rpt = self.gen_korean_rpt(count, top_bhv, clarity, current_counts, has_pollutant, pollutant_conf)
                latency = (time.time() - start_t) * 1000
                print("[DEBUG] log_inference DB 저장 전")
                self.log_inference(count, conf, top_bhv, clarity, rpt, latency)
                print("[DEBUG] log_inference DB 저장 완료")
                self.safe_after(0, lambda: self.type_rpt(rpt, 0))

            print("[DEBUG] fetch_llm_report 스레드 시작 전")
            threading.Thread(target=fetch_llm_report, daemon=True).start()

        except Exception as e:
            print(f"[FAIL] Worker Logic Error: {e}")
            err_msg = f"[WARN] 분석 중 오류 발생\n\n{type(e).__name__}: {str(e)[:120]}"
            self.safe_after(0, lambda: getattr(self, 'img_lbl').configure(
                text=err_msg, image="", text_color=COLOR_DANGER) if hasattr(self, 'img_lbl') else None)
            self.is_analyzing = False

    def _show_alert_banner(self, conf_value):
        if not hasattr(self, 'alert_banner') or not self.alert_banner.winfo_exists():
            return
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            self.alert_banner_label.configure(text=f"[ ! ] 오염물질 감지! | 신뢰도: {conf_value:.1%} | 시각: {ts} | 서보 와이퍼 가동을 권장합니다 [ ! ]")
            if not getattr(self, 'alert_banner_visible', False):
                try:
                    parent_children = self.alert_banner.master.winfo_children()
                    if len(parent_children) >= 3:
                        self.alert_banner.pack(fill="x", padx=self.scale(35), pady=(self.scale(5), 0), before=parent_children[2])
                    else:
                        self.alert_banner.pack(fill="x", padx=self.scale(35), pady=(self.scale(5), 0))
                except:
                    self.alert_banner.pack(fill="x", padx=self.scale(35), pady=(self.scale(5), 0))
                self.alert_banner_visible = True
            if not getattr(self, '_alert_blink_active', False):
                self._alert_blink_active = True
                self._blink_alert_banner(True)
        except:
            pass

    def _hide_alert_banner(self):
        if not hasattr(self, 'alert_banner'):
            return
        self._alert_blink_active = False
        if getattr(self, 'alert_banner_visible', False):
            self.alert_banner.pack_forget()
            self.alert_banner_visible = False

    def _blink_alert_banner(self, is_bright):
        if not getattr(self, '_alert_blink_active', False) or not hasattr(self, 'alert_banner') or not self.alert_banner.winfo_exists():
            return
        self.alert_banner.configure(fg_color="#dc2626" if is_bright else "#7f1d1d")
        self.safe_after(500, lambda: self._blink_alert_banner(not is_bright))

    def sync_ui(self, res, counts: dict, count: int, conf: float, clarity: float, has_pollutant: bool = False, pollutant_conf: float = 0.0):
        """메인 스레드 전용. res.plot()을 여기서 실행하여 폰트/그래픽 드라이버 충돌 방지."""
        if self.is_destroyed or not hasattr(self, 'img_lbl') or not self.img_lbl.winfo_exists():
            return
        try:
            # [ALARM] res.plot()은 메인 스레드에서만 호출 (백그라운드 스레드 호출 시 즉사)
            img = res.plot()
            w, h = max(100, self.img_lbl.winfo_width()), max(100, self.img_lbl.winfo_height())
            self.img_lbl.configure(
                image=ctk.CTkImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), size=(w, h)), text="")

            if has_pollutant and pollutant_conf >= 0.90:
                self._show_alert_banner(pollutant_conf)
                if hasattr(self, 'v_card'):
                    self.v_card.configure(border_color=COLOR_DANGER, border_width=2)
            else:
                self._hide_alert_banner()
                if hasattr(self, 'v_card'):
                    self.v_card.configure(border_color="#334155", border_width=1)

            if count > 0:
                active_bhvs = {k: v for k, v in counts.items() if v > 0}
                if active_bhvs:
                    max_val = max(active_bhvs.values())
                    top_bhvs = [k for k, v in active_bhvs.items() if v == max_val]
                    bhv_display = f"[{' & '.join(top_bhvs).upper()} : {max_val}]"
                    top = top_bhvs[0]
                else:
                    bhv_display, top = "[NONE]", "Standing"
            else:
                bhv_display, top = "[COW NOT FOUND]", "Standing"

            if hasattr(self, 'st_val'):
                t_len = len(bhv_display)
                self.st_val.configure(
                    text=bhv_display,
                    text_color={"Standing": COLOR_ACCENT, "Sleeping": COLOR_SUCCESS, "Eating": COLOR_WARNING}.get(top, COLOR_TEXT_MAIN),
                    font=self.get_font(max(18, int(40 * (10 / t_len))) if t_len > 10 else 40, "bold"))

            if hasattr(self, 'lbl_standing'):
                st_t = f"[STAND] Standing: {counts.get('Standing', 0)}"
                sl_t = f"[SLEEP] Sleeping: {counts.get('Sleeping', 0)}"
                ea_t = f"[EAT] Eating: {counts.get('Eating', 0)}"
                tc = len(st_t) + len(sl_t) + len(ea_t)
                sz = max(8, int(12 * (36 / tc))) if tc > 36 else 12
                self.lbl_standing.configure(text=st_t, font=self.get_font(sz))
                self.lbl_sleeping.configure(text=sl_t, font=self.get_font(sz))
                self.lbl_eating.configure(text=ea_t, font=self.get_font(sz))

            if hasattr(self, 'm_count'):
                self.m_count.configure(text=str(count))
            if hasattr(self, 'm_conf'):
                self.m_conf.configure(text=f"{conf:.1%}")
            if hasattr(self, 'm_clarity'):
                self.m_clarity.set(clarity)
        except:
            pass

    def gen_korean_rpt(self, c, b, l, counts, has_pollutant, poll_conf):
        now_str = datetime.now().strftime("%Y년 %m월 %d일 %p %I시 %M분")
        poll_status = (f"렌즈 오염(신뢰도 {poll_conf:.2f}) 감지됨. 엣지 컴퓨팅 기반 서보 와이퍼 액추에이터 제어 모듈 가동 필요."
                       if has_pollutant else "오염 물질 미감지. 렌즈 상태 정상.")
        p = (f"당신은 엣지 컴퓨팅 기반 스마트 축산 AI 'Vision Doctor'의 수석 진단 엔진입니다.\n"
             f"제공된 데이터를 바탕으로 시스템 관리자를 위한 가장 전문적인 관제 리포트를 작성하십시오.\n\n"
             f"[입력 데이터]\n"
             f"- 일시: {now_str}\n"
             f"- 개체수: 총 {c}마리 (서있음 {counts.get('Standing',0)}, 누워있음 {counts.get('Sleeping',0)}, 먹이먹음 {counts.get('Eating',0)})\n"
             f"- 환경 상태: {poll_status}\n"
             f"- 시야 확보율: {l:.1%}\n\n"
             f"반드시 아래의 양식에 정확하게 맞추어 출력하십시오. (다른 인사말이나 추가 설명 절대 금지):\n\n"
             f" VISION DOCTOR AI 진단 리포트\n"
             f"▪ 관제 일시: {now_str} (Jetson Orin Nano Edge)\n"
             f"▪ 객체 분석: [입력 데이터를 바탕으로 소들의 현재 행동 상태 및 특이사항을 전문가적 어조로 1~2줄 이내로 요약]\n"
             f"▪ 환경 제어: [오염이 감지된 경우 'CCTV 렌즈에서 오염이 감지되었습니다. 시스템이 와이퍼를 가동하였습니다.'와 유사하게 작성. "
             f"오염이 없으면 '현재 렌즈 상태는 청결하며 최적의 모니터링 환경을 유지하고 있습니다.'라고 작성]\n"
             f"▪ 종합 소견: [가축의 행동 상태와 하드웨어 환경을 종합한 시스템적 관리자 조언 1문장]")

        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        payload = json.dumps({
            "model": "meta/llama-3.1-8b-instruct",
            "messages": [{"role": "user", "content": p}],
            "max_tokens": 350, "temperature": 0.4, "top_p": 1.0
        })

        for attempt in range(3):
            try:
                cmd = ["curl", "-s", "-X", "POST", url,
                       "-H", f"Authorization: Bearer {NVIDIA_API_KEY}",
                       "-H", "Content-Type: application/json",
                       "-d", payload]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60.0)
                if result.returncode != 0:
                    raise ValueError("Curl 에러")
                report_text = json.loads(result.stdout).get('choices', [{}])[0].get('message', {}).get('content', '')
                if report_text and len(report_text.strip()) > 5:
                    return report_text.strip()
            except:
                time.sleep(5 * (attempt + 1))
        return (" VISION DOCTOR AI 진단 리포트\n"
                "▪ 시스템 오류: AI 서버 연결 지연 (Timeout)\n"
                "▪ 종합 소견: NVIDIA NIM 네트워크 상태를 확인해주십시오.")

    def type_rpt(self, txt, i):
        if not hasattr(self, 'rpt_box') or not self.rpt_box.winfo_exists():
            self.is_analyzing = False
            return
        if i == 0:
            self.rpt_box.delete("1.0", tk.END)
        if i < len(txt):
            try:
                self.rpt_box.insert(tk.END, txt[i])
                self.safe_after(10, lambda: self.type_rpt(txt, i + 1))
            except:
                self.is_analyzing = False
        else:
            self.is_analyzing = False

    def show_settings(self):
        self.clear_view()
        sv = ctk.CTkFrame(self.container, fg_color=COLOR_BG)
        sv.pack(fill="both", expand=True)
        h = ctk.CTkFrame(sv, fg_color="transparent")
        h.pack(fill="x", padx=self.scale(45), pady=(self.scale(40), self.scale(20)))
        ctk.CTkLabel(h, text="Vision Doctor", font=self.get_font(24, "bold"), text_color=COLOR_TEXT_MAIN).pack(side="left")

        body = ctk.CTkFrame(sv, fg_color="transparent")
        body.pack(expand=True, fill="both", padx=self.scale(60), pady=self.scale(20))
        body.grid_columnconfigure((0, 1), weight=1, uniform="settings_cols")
        ctk.CTkLabel(body, text="System Configuration", font=self.get_font(38, "bold"), text_color=COLOR_TEXT_MAIN).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, self.scale(40)))

        left = ctk.CTkFrame(body, fg_color="transparent")
        left.grid(row=1, column=0, sticky="nsew", padx=(0, self.scale(40)))
        ctk.CTkLabel(left, text="AI Confidence Threshold", font=self.get_font(16), text_color=COLOR_TEXT_SUB).pack(anchor="w", pady=(0, self.scale(5)))
        conf_val = ctk.CTkLabel(left, text=f"{self.settings_conf_threshold:.2f}", font=self.get_font(16, "bold"), text_color=COLOR_ACCENT)
        conf_val.pack(anchor="w")
        
        # 표 6-4 품질 테스트 조치결과: 슬라이더 범위 0.40 ~ 0.90으로 제한
        s1 = ctk.CTkSlider(left, from_=0.4, to=0.9, number_of_steps=50, fg_color="#1e293b", progress_color=COLOR_ACCENT,
            command=lambda v: conf_val.configure(text=f"{v:.2f}"))
        s1.pack(fill="x", pady=(self.scale(10), self.scale(30)))
        s1.set(self.settings_conf_threshold)

        ctk.CTkLabel(left, text="Feature Automation", font=self.get_font(16), text_color=COLOR_TEXT_SUB).pack(anchor="w", pady=(self.scale(10), self.scale(15)))
        sw1 = ctk.CTkSwitch(left, text="Auto-save cattle detection imgs", font=self.get_font(15), progress_color=COLOR_SUCCESS)
        sw1.pack(anchor="w", pady=self.scale(8))
        sw1.select() if self.settings_auto_save else sw1.deselect()
        sw2 = ctk.CTkSwitch(left, text="Auto-save behavior analytics", font=self.get_font(15), progress_color=COLOR_SUCCESS)
        sw2.pack(anchor="w", pady=self.scale(8))
        sw2.select()
        sw3 = ctk.CTkSwitch(left, text="Real-time cloud sync", font=self.get_font(15), progress_color=COLOR_SUCCESS)
        sw3.pack(anchor="w", pady=self.scale(8))
        sw3.select()

        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=1, column=1, sticky="nsew")
        ctk.CTkLabel(right, text="AI Alert Interval (sec)", font=self.get_font(16), text_color=COLOR_TEXT_SUB).pack(anchor="w", pady=(0, self.scale(5)))
        int_val = ctk.CTkLabel(right, text=f"{self.settings_alert_interval}", font=self.get_font(16, "bold"), text_color=COLOR_WARNING)
        int_val.pack(anchor="w")
        s2 = ctk.CTkSlider(right, from_=10, to=600, number_of_steps=59, fg_color="#1e293b", progress_color=COLOR_WARNING,
            command=lambda v: int_val.configure(text=f"{int(v)}"))
        s2.pack(fill="x", pady=(self.scale(10), self.scale(30)))
        s2.set(self.settings_alert_interval)

        ctk.CTkLabel(right, text="Analysis Mode", font=self.get_font(16), text_color=COLOR_TEXT_SUB).pack(anchor="w", pady=(self.scale(10), self.scale(5)))
        m_opt = ctk.CTkOptionMenu(right, values=["Balanced", "High Precision", "Low Latency"],
            font=self.get_font(14), fg_color="#1e293b", button_color=COLOR_ACCENT)
        m_opt.pack(fill="x", pady=(self.scale(10), self.scale(30)))
        m_opt.set(self.settings_mode)

        ctk.CTkLabel(right, text="Administrator Name", font=self.get_font(16), text_color=COLOR_TEXT_SUB).pack(anchor="w", pady=(self.scale(10), self.scale(5)))
        admin_ent = ctk.CTkEntry(right, height=self.scale(45), font=self.get_font(16), fg_color="#1e293b", border_width=0)
        admin_ent.pack(fill="x", pady=(self.scale(10), 0))
        admin_ent.insert(0, self.settings_admin_name)

        # [NEW] 텔레그램 알람 ON/OFF 섹션
        ctk.CTkLabel(right, text="Telegram Notification", font=self.get_font(16), text_color=COLOR_TEXT_SUB).pack(anchor="w", pady=(self.scale(20), self.scale(8)))
        tg_frame = ctk.CTkFrame(right, fg_color="#1e293b", corner_radius=self.scale(12))
        tg_frame.pack(fill="x", pady=(0, self.scale(10)))
        tg_icon = ctk.CTkLabel(tg_frame, text="[TG]", font=self.get_font(14), text_color=COLOR_ACCENT)
        tg_icon.pack(side="left", padx=(self.scale(12), self.scale(6)), pady=self.scale(10))
        sw_tg = ctk.CTkSwitch(tg_frame, text="텔레그램 알람 활성화", font=self.get_font(15),
            progress_color=COLOR_ACCENT, button_color="#2563eb")
        sw_tg.pack(side="left", pady=self.scale(10))
        sw_tg.select() if self.telegram_enabled else sw_tg.deselect()
        tg_status = ctk.CTkLabel(tg_frame, text="ON" if self.telegram_enabled else "OFF",
            font=self.get_font(13, "bold"),
            text_color=COLOR_ACCENT if self.telegram_enabled else COLOR_TEXT_SUB)
        tg_status.pack(side="right", padx=self.scale(12))
        def on_tg_toggle():
            is_on = bool(sw_tg.get())
            tg_status.configure(text="ON" if is_on else "OFF",
                text_color=COLOR_ACCENT if is_on else COLOR_TEXT_SUB)
        sw_tg.configure(command=on_tg_toggle)

        f = ctk.CTkFrame(sv, fg_color="transparent")
        f.pack(fill="x", pady=self.scale(60))
        ctk.CTkButton(f, text="SAVE & APPLY SETTINGS", font=self.get_font(18, "bold"),
            fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HOVER,
            width=self.scale(300), height=self.scale(55), corner_radius=self.scale(15),
            command=lambda: self.apply_settings(
                s1.get(), s2.get(), sw1.get(), m_opt.get(), admin_ent.get(), bool(sw_tg.get())
            )).pack()

    def apply_settings(self, conf, interval, auto_save, mode, admin, telegram_enabled=True):
        self.settings_conf_threshold = conf
        self.settings_alert_interval = int(interval)
        self.settings_auto_save = bool(auto_save)
        self.settings_mode = mode
        self.settings_admin_name = admin
        prev_tg = self.telegram_enabled
        self.telegram_enabled = bool(telegram_enabled)
        if prev_tg != self.telegram_enabled:
            status = "활성화" if self.telegram_enabled else "비활성화"
            print(f"[SETTINGS] 텔레그램 알람 {status}")
            if self.telegram_enabled:
                # 재활성화 시 확인 메시지 전송
                self.send_telegram_alert(f"[CHECK] 텔레그램 알람이 재활성화되었습니다. (관리자: {admin})")
        self.show_dashboard()

    def show_statistics(self, tab="Dashboard"):
        self.clear_view()
        main = ctk.CTkFrame(self.container, fg_color=COLOR_BG)
        main.pack(fill="both", expand=True)
        side = ctk.CTkFrame(main, fg_color=COLOR_SIDEBAR, width=self.scale(280), corner_radius=0)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        ctk.CTkLabel(side, text="[STAT] Analytics", font=self.get_font(24, "bold"), text_color=COLOR_ACCENT).pack(pady=(self.scale(40), self.scale(30)), padx=self.scale(30), anchor="w")

        tabs = [("Dashboard", "show_statistics"), ("Reports", "show_statistics"), ("Performance", "show_statistics"), ("Settings", "show_settings")]
        for name, cmd_name in tabs:
            cmd = (lambda tab_name=name: self.show_statistics(tab_name)) if name != "Settings" else self.show_settings
            btn = ctk.CTkButton(side, text=f"   {name}", font=self.get_font(16), fg_color="transparent",
                text_color=COLOR_TEXT_SUB, anchor="w", hover_color="#1e293b", height=self.scale(50), command=cmd)
            btn.pack(fill="x", padx=self.scale(15), pady=self.scale(2))
            if name == tab:
                btn.configure(fg_color="#3b82f6", text_color="#FFFFFF", font=self.get_font(16, "bold"))

        ctk.CTkButton(side, text="[BACK] BACK HOME", font=self.get_font(14, "bold"), fg_color="#334155",
            command=self.show_dashboard, height=self.scale(45)).pack(side="bottom", fill="x", padx=self.scale(30), pady=self.scale(40))

        cnt_frame = ctk.CTkFrame(main, fg_color="transparent")
        cnt_frame.pack(side="right", expand=True, fill="both")
        cnt = ctk.CTkScrollableFrame(cnt_frame, fg_color="transparent", corner_radius=0)
        cnt.pack(expand=True, fill="both", padx=self.scale(40), pady=self.scale(40))

        h = ctk.CTkFrame(cnt, fg_color="transparent")
        h.pack(fill="x", pady=(0, self.scale(30)))
        ctk.CTkLabel(h, text=f"Statistical {tab}", font=self.get_font(32, "bold"), text_color=COLOR_TEXT_MAIN).pack(side="left")

        # DB 조회 — 스레드 독립 커넥션으로 세그폴트 방지
        with db_lock:
            conn = None
            try:
                conn = sqlite3.connect("vision_doctor.db")
                all_data = conn.execute(
                    "SELECT confidence, behavior, report, latency, timestamp, count, id FROM inference_logs ORDER BY id DESC LIMIT 100"
                ).fetchall()
            except:
                all_data = []
            finally:
                try:
                    if conn:
                        conn.close()
                except:
                    pass

        if tab == "Dashboard":
            import re
            from collections import Counter
            meta_bar = ctk.CTkFrame(cnt, fg_color="transparent")
            meta_bar.pack(fill="x", pady=(0, self.scale(10)))
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ctk.CTkLabel(meta_bar, text=f"[TIME] Last Updated: {now_str}", font=self.get_font(12), text_color=COLOR_TEXT_SUB).pack(side="left")
            ctk.CTkButton(meta_bar, text="[REFRESH] REFRESH", font=self.get_font(12, "bold"), fg_color="#1e293b", hover_color="#334155",
                width=self.scale(110), height=self.scale(32), corner_radius=self.scale(8),
                command=lambda: self.show_statistics("Dashboard")).pack(side="right")

            confidences = [d[0] for d in all_data] if all_data else [0]
            total_scans = len(all_data)
            avg_conf = (sum(confidences) / len(confidences)) if all_data else 0.0
            total_cattle = sum((d[5] or 0) for d in all_data) if all_data else 0
            latencies = [d[3] for d in all_data if d[3] is not None]
            avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0

            today_str = datetime.now().strftime("%Y-%m-%d")
            today_scans = len([d for d in all_data if d[4] and str(d[4]).startswith(today_str)])
            today_cattle = sum((d[5] or 0) for d in all_data if d[4] and str(d[4]).startswith(today_str))

            high_conf = len([c for c in confidences if c >= 0.7])
            reliability_pct = (high_conf / len(confidences) * 100) if all_data else 0

            kpi_row = ctk.CTkFrame(cnt, fg_color="transparent")
            kpi_row.pack(fill="x", pady=(self.scale(5), self.scale(15)))
            kpi_row.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="kpi")

            def make_kpi(parent, col, icon, label, value, sub, accent_color):
                card = ctk.CTkFrame(parent, fg_color=COLOR_CARD, corner_radius=self.scale(16),
                    border_width=1, border_color="#334155", height=self.scale(135))
                card.grid(row=0, column=col, padx=self.scale(7), sticky="nsew")
                card.pack_propagate(False)
                top = ctk.CTkFrame(card, fg_color="transparent")
                top.pack(fill="x", padx=self.scale(20), pady=(self.scale(18), 0))
                ctk.CTkLabel(top, text=icon, font=self.get_font(22), text_color=accent_color).pack(side="left")
                ctk.CTkLabel(top, text=label, font=self.get_font(11, "bold"), text_color=COLOR_TEXT_SUB).pack(side="left", padx=(self.scale(10), 0))
                ctk.CTkLabel(card, text=value, font=self.get_font(30, "bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w", padx=self.scale(20), pady=(self.scale(8), 0))
                ctk.CTkLabel(card, text=sub, font=self.get_font(11), text_color=accent_color).pack(anchor="w", padx=self.scale(20), pady=(self.scale(2), self.scale(15)))

            lat_color = COLOR_SUCCESS if avg_latency < 150 else (COLOR_WARNING if avg_latency < 250 else COLOR_DANGER)
            conf_color = COLOR_SUCCESS if avg_conf >= 0.8 else (COLOR_WARNING if avg_conf >= 0.6 else COLOR_DANGER)

            make_kpi(kpi_row, 0, "[STAT]", "TOTAL SCANS", f"{total_scans:,}", f"+{today_scans} today", COLOR_ACCENT)
            make_kpi(kpi_row, 1, "[SET]", "AVG CONFIDENCE", f"{avg_conf:.1%}", f"Reliability {reliability_pct:.0f}%", conf_color)
            make_kpi(kpi_row, 2, "[COW]", "CATTLE DETECTED", f"{total_cattle:,}", f"+{today_cattle} today", COLOR_WARNING)
            make_kpi(kpi_row, 3, "[LATENCY]", "AVG LATENCY", f"{avg_latency:.0f} ms", "AI inference speed", lat_color)

            behavior_counts = Counter()
            for d in all_data:
                bstr = str(d[1] or "")
                matches = re.findall(r'(STANDING|SLEEPING|EATING|LYING)\s*:\s*(\d+)', bstr.upper())
                if matches:
                    for nm, num in matches:
                        behavior_counts[nm.title()] += int(num)
                else:
                    for key in ["Standing", "Sleeping", "Eating", "Lying"]:
                        if key.upper() in bstr.upper():
                            behavior_counts[key] += (d[5] or 1)
            for k in ["Standing", "Sleeping", "Eating", "Lying"]:
                behavior_counts.setdefault(k, 0)

            row2 = ctk.CTkFrame(cnt, fg_color="transparent")
            row2.pack(fill="x", pady=self.scale(10))
            row2.grid_columnconfigure((0, 1), weight=1, uniform="row2")

            bd_card = ctk.CTkFrame(row2, fg_color=COLOR_CARD, corner_radius=self.scale(20), border_width=1, border_color="#334155")
            bd_card.grid(row=0, column=0, padx=(0, self.scale(15)), sticky="nsew")
            bd_card.pack_propagate(False)
            ctk.CTkLabel(bd_card, text="[COW] Behavior Distribution", font=self.get_font(16, "bold"), text_color=COLOR_TEXT_SUB).pack(pady=self.scale(20))
            for k, v in behavior_counts.items():
                row = ctk.CTkFrame(bd_card, fg_color="transparent")
                row.pack(fill="x", padx=self.scale(20), pady=self.scale(5))
                ctk.CTkLabel(row, text=k, font=self.get_font(13), width=80, anchor="w").pack(side="left")
                ctk.CTkProgressBar(row, progress_color=COLOR_ACCENT, fg_color="#1e293b").pack(side="left", expand=True, fill="x", padx=10)
                ctk.CTkLabel(row, text=str(v), font=self.get_font(13, "bold")).pack(side="left", padx=10)

            at_card = ctk.CTkFrame(row2, fg_color=COLOR_CARD, corner_radius=self.scale(20), border_width=1, border_color="#334155")
            at_card.grid(row=0, column=1, sticky="nsew")
            at_card.pack_propagate(False)
            ctk.CTkLabel(at_card, text="[TREND] Cattle Trend Summary", font=self.get_font(16, "bold"), text_color=COLOR_TEXT_SUB).pack(pady=self.scale(20))
            ctk.CTkLabel(at_card, text=f"Recent Mean: {sum([d[5] or 0 for d in all_data[:10]])/max(1, len(all_data[:10])):.1f} heads",
                font=self.get_font(24, "bold")).pack(expand=True)

            tbl_card = ctk.CTkFrame(cnt, fg_color=COLOR_CARD, corner_radius=self.scale(20), border_width=1, border_color="#334155")
            tbl_card.pack(fill="x", pady=self.scale(15))
            th = ctk.CTkFrame(tbl_card, fg_color="transparent")
            th.pack(fill="x", padx=self.scale(30), pady=(self.scale(25), self.scale(12)))
            ctk.CTkLabel(th, text="[LOG] Historical Log Entries", font=self.get_font(18, "bold"), text_color=COLOR_ACCENT).pack(side="left")
            ctk.CTkLabel(th, text=f"Showing latest {min(15, len(all_data))} of {len(all_data)}", font=self.get_font(12), text_color=COLOR_TEXT_SUB).pack(side="right")

            hdr = ctk.CTkFrame(tbl_card, fg_color="#0f172a", height=self.scale(38), corner_radius=self.scale(8))
            hdr.pack(fill="x", padx=self.scale(20), pady=(0, self.scale(6)))
            hdr.pack_propagate(False)
            for h_text, h_w in [("ID", 60), ("TIMESTAMP", 200), ("CATTLE", 110), ("CONFIDENCE", 130), ("BEHAVIOR", 200), ("STATUS", 90)]:
                ctk.CTkLabel(hdr, text=h_text, font=self.get_font(11, "bold"), text_color=COLOR_TEXT_SUB,
                    width=self.scale(h_w), anchor="w").pack(side="left", padx=self.scale(10))

            for r in all_data[:15]:
                rf = ctk.CTkFrame(tbl_card, fg_color="transparent", height=self.scale(42))
                rf.pack(fill="x", padx=self.scale(20))
                rf.pack_propagate(False)
                ctk.CTkLabel(rf, text=f"#{r[6]}", font=self.get_font(13, "bold"), text_color=COLOR_SUCCESS, width=self.scale(60), anchor="w").pack(side="left", padx=self.scale(10))
                ctk.CTkLabel(rf, text=r[4] or "—", font=self.get_font(13), text_color="#cbd5e1", width=self.scale(200), anchor="w").pack(side="left", padx=self.scale(10))
                ctk.CTkLabel(rf, text=f"{r[5]} cattle", font=self.get_font(13), text_color=COLOR_TEXT_MAIN, width=self.scale(110), anchor="w").pack(side="left", padx=self.scale(10))

                conf_v = r[0] or 0
                ccol = COLOR_SUCCESS if conf_v >= 0.85 else (COLOR_WARNING if conf_v >= 0.7 else COLOR_DANGER)
                ctk.CTkLabel(rf, text=f"{conf_v:.1%}", font=self.get_font(13, "bold"), text_color=ccol, width=self.scale(130), anchor="w").pack(side="left", padx=self.scale(10))
                ctk.CTkLabel(rf, text=r[1] or "—", font=self.get_font(13), text_color=COLOR_WARNING, width=self.scale(200), anchor="w").pack(side="left", padx=self.scale(10))

                status = "OK" if conf_v >= self.settings_conf_threshold else "ALERT"
                sb = ctk.CTkFrame(rf, fg_color=COLOR_SUCCESS if status == "OK" else COLOR_DANGER,
                    corner_radius=self.scale(6), width=self.scale(70), height=self.scale(24))
                sb.pack(side="left", padx=self.scale(10), pady=self.scale(8))
                sb.pack_propagate(False)
                ctk.CTkLabel(sb, text=status, font=self.get_font(11, "bold"), text_color="#FFFFFF").pack(expand=True)
                ctk.CTkFrame(tbl_card, fg_color="#1e293b", height=1).pack(fill="x", padx=self.scale(20))
            ctk.CTkFrame(tbl_card, fg_color="transparent", height=self.scale(15)).pack()

        elif tab == "Reports":
            ctk.CTkLabel(cnt, text="Historical AI Diagnostics (LLM Archives)", font=self.get_font(16), text_color=COLOR_TEXT_SUB).pack(anchor="w", pady=(0, self.scale(20)))
            for r in all_data:
                card = ctk.CTkFrame(cnt, fg_color=COLOR_CARD, corner_radius=self.scale(15), border_width=1, border_color="#334155")
                card.pack(fill="x", pady=self.scale(10))
                ctk.CTkLabel(card, text=f"[TIME] {r[1]} Reporting", font=self.get_font(14, "bold"), text_color=COLOR_ACCENT).pack(anchor="w", padx=self.scale(25), pady=(self.scale(20), self.scale(5)))
                ctk.CTkLabel(card, text=f"Summary: \n{r[2]}", font=self.get_font(15), text_color="#FFFFFF", wraplength=self.scale(1000), justify="left").pack(anchor="w", padx=self.scale(25), pady=(0, self.scale(20)))

        elif tab == "Performance":
            perf_card = ctk.CTkFrame(cnt, fg_color=COLOR_CARD, corner_radius=self.scale(20), border_width=1, border_color="#334155")
            perf_card.pack(fill="both", expand=True, pady=self.scale(20))
            ctk.CTkLabel(perf_card, text="AI Engine Latency Trend (Inference Speed)", font=self.get_font(18, "bold"), text_color=COLOR_ACCENT).pack(pady=self.scale(20))
            latencies = [d[3] for d in all_data[::-1] if d[3] is not None]

            if not latencies:
                ctk.CTkLabel(perf_card, text="Insufficient performance data. Run more scans.", font=self.get_font(16)).pack(pady=self.scale(40))
            else:
                # [ALARM] plt.subplots() 대신 Figure 객체를 직접 생성 (최상단에 from matplotlib.figure import Figure 필요)
                from matplotlib.figure import Figure

                f3 = Figure(figsize=(8, 4), facecolor=COLOR_CARD)
                ax3 = f3.add_subplot(111) # 서브플롯 직접 추가
                ax3.set_facecolor(COLOR_CARD)
                ax3.plot(latencies, color=COLOR_SUCCESS, linewidth=2, marker='o', markersize=4, markerfacecolor="#FFFFFF")
                ax3.set_ylabel("Latency (ms)", color=COLOR_TEXT_SUB)
                ax3.tick_params(colors=COLOR_TEXT_SUB)
                for s in ax3.spines.values():
                    s.set_visible(False)

                f3.tight_layout()
                FigureCanvasTkAgg(f3, master=perf_card).get_tk_widget().pack(fill="both", expand=True, padx=self.scale(30), pady=self.scale(20))
                ctk.CTkLabel(perf_card, text=f"Average Inference Speed: {sum(latencies)/len(latencies):.2f} ms",
                    font=self.get_font(20, "bold"), text_color=COLOR_TEXT_MAIN).pack(pady=self.scale(20))

if __name__ == "__main__":
    app = VisionDoctorDashboard()
    app.mainloop()