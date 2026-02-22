"""
Requirements:
    pip install deepface opencv-python numpy
"""

import json
import os
import shutil
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace


class _JSONEncoder(json.JSONEncoder):
    """Converts NumPy scalar types (float32, int64, …) to plain Python types."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _dumps(obj, **kw) -> str:
    return json.dumps(obj, cls=_JSONEncoder, **kw)


def _loads(s: str):
    return json.loads(s)


def _sanitise_emotions(emotions: dict) -> dict:
    """Convert emotion scores to plain Python floats (DeepFace returns numpy float32)."""
    return {k: float(v) for k, v in emotions.items()}



PROFILES_DIR        = Path("profiles")
SESSION_LOG         = Path("session_log.json")
RECOGNITION_MODEL   = "Facenet512"   # VGG-Face | Facenet | Facenet512 | ArcFace
RECOGNITION_METRIC  = "cosine"       # cosine | euclidean | euclidean_l2
RECOGNITION_THRESH  = 0.40           # lower = stricter
UNKNOWN_LABEL       = "Unknown"
ANALYZE_EVERY_N     = 10             # process every Nth camera frame


# ══════════════════════════════════════════════════════════════════════
#  Terminal helpers
# ══════════════════════════════════════════════════════════════════════

C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_CYAN   = "\033[96m"
C_GREEN  = "\033[92m"
C_YELLOW = "\033[93m"
C_RED    = "\033[91m"
C_DIM    = "\033[2m"

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def banner():
    print(f"""{C_CYAN}{C_BOLD}
╔══════════════════════════════════════════════╗
║      Face Profile Manager  ·  DeepFace       ║
╚══════════════════════════════════════════════╝{C_RESET}""")

def section(title: str):
    print(f"\n{C_YELLOW}{C_BOLD}── {title} ──{C_RESET}")

def ok(msg: str):
    print(f"{C_GREEN}✔  {msg}{C_RESET}")

def err(msg: str):
    print(f"{C_RED}✘  {msg}{C_RESET}")

def info(msg: str):
    print(f"{C_DIM}   {msg}{C_RESET}")

def prompt(msg: str) -> str:
    return input(f"{C_CYAN}▶  {msg}{C_RESET}").strip()

def confirm(msg: str) -> bool:
    ans = prompt(f"{msg} [y/N] ").lower()
    return ans in ("y", "yes")

def pause():
    input(f"\n{C_DIM}Enter para continuar…{C_RESET}")

def choose_menu(title: str, options: list[str]) -> int:
    """Print a numbered menu, return 0-based index of chosen option."""
    section(title)
    for i, opt in enumerate(options, 1):
        print(f"  {C_BOLD}{i}{C_RESET}. {opt}")
    while True:
        raw = prompt(f"Opcion entre (1–{len(options)}): ")
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        err(f"Por favor ingresa un numero entre 1 y {len(options)}.")


class ProfileStore:
    def __init__(self, profiles_dir: Path = PROFILES_DIR):
        self.dir = profiles_dir
        self.dir.mkdir(exist_ok=True)

    def add_profile(self, name: str, face_image: np.ndarray) -> str:
        profile_id   = str(uuid.uuid4())[:8]
        profile_path = self.dir / profile_id
        profile_path.mkdir()
        meta = {"id": profile_id, "name": name, "created_at": datetime.now().isoformat()}
        (profile_path / "meta.json").write_text(_dumps(meta, indent=2))
        cv2.imwrite(str(profile_path / "face.jpg"), face_image)
        return profile_id

    def add_face_sample(self, profile_id: str, face_image: np.ndarray) -> bool:
        profile_path = self.dir / profile_id
        if not profile_path.exists():
            return False
        idx = len(list(profile_path.glob("face*.jpg")))
        cv2.imwrite(str(profile_path / f"face_{idx}.jpg"), face_image)
        return True

    def list_profiles(self) -> list[dict]:
        profiles = []
        for p in self.dir.iterdir():
            meta_file = p / "meta.json"
            if meta_file.exists():
                profiles.append(_loads(meta_file.read_text()))
        return sorted(profiles, key=lambda x: x["created_at"])

    def get_profile(self, profile_id: str) -> dict | None:
        meta_file = self.dir / profile_id / "meta.json"
        return _loads(meta_file.read_text()) if meta_file.exists() else None

    def delete_profile(self, profile_id: str) -> bool:
        profile_path = self.dir / profile_id
        if profile_path.exists():
            shutil.rmtree(profile_path)
            return True
        return False

    def get_face_paths(self, profile_id: str) -> list[str]:
        return [str(p) for p in sorted((self.dir / profile_id).glob("face*.jpg"))]


# ══════════════════════════════════════════════════════════════════════
#  Session Manager
# ══════════════════════════════════════════════════════════════════════

class SessionManager:
    def __init__(self, log_path: Path = SESSION_LOG):
        self.log_path   = log_path
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._readings: dict[str, list[dict]] = {}

    def record(self, profile_id: str, emotions: dict, dominant: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "emotions":  _sanitise_emotions(emotions),   # ensure plain float, not numpy float32
            "dominant":  str(dominant),
        }
        self._readings.setdefault(profile_id, []).append(entry)

    def reading_count(self, profile_id: str) -> int:
        return len(self._readings.get(profile_id, []))

    def summary(self, profile_id: str) -> dict | None:
        readings = self._readings.get(profile_id)
        if not readings:
            return None
        emotion_keys = list(readings[0]["emotions"].keys())
        averages = {
            e: round(sum(r["emotions"].get(e, 0) for r in readings) / len(readings), 1)
            for e in emotion_keys
        }
        counts: dict[str, int] = {}
        for r in readings:
            counts[r["dominant"]] = counts.get(r["dominant"], 0) + 1
        return {
            "profile_id":           profile_id,
            "total_readings":       len(readings),
            "average_emotions":     averages,
            "most_common_dominant": max(counts, key=counts.get),
            "dominant_counts":      counts,
        }

    def all_summaries(self) -> list[dict]:
        return [s for pid in self._readings if (s := self.summary(pid))]

    def save(self):
        data = _loads(self.log_path.read_text()) if self.log_path.exists() else {}
        data[self.session_id] = {
            "session_id": self.session_id,
            "profiles": {
                pid: {
                    "readings": self._readings[pid],
                    "summary":  self.summary(pid),
                }
                for pid in self._readings
            },
        }
        self.log_path.write_text(_dumps(data, indent=2))


# ══════════════════════════════════════════════════════════════════════
#  Face Engine
# ══════════════════════════════════════════════════════════════════════

class FaceEngine:
    def __init__(self, store: ProfileStore):
        self.store = store

    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        try:
            faces = DeepFace.extract_faces(
                img_path=frame, detector_backend="opencv", enforce_detection=False
            )
            return [f for f in faces if f.get("confidence", 0) > 0.5]
        except Exception:
            return []

    def analyze_emotion(self, face_img: np.ndarray) -> dict | None:
        try:
            result = DeepFace.analyze(
                img_path=face_img, actions=["emotion"],
                enforce_detection=False, silent=True
            )
            r = result[0] if isinstance(result, list) else result
            return {"emotions": r["emotion"], "dominant": r["dominant_emotion"]}
        except Exception:
            return None

    def identify(self, face_img: np.ndarray) -> tuple[str, str]:
        profiles = self.store.list_profiles()
        if not profiles:
            return UNKNOWN_LABEL, UNKNOWN_LABEL
        best_dist, best_profile = float("inf"), None
        for profile in profiles:
            for face_path in self.store.get_face_paths(profile["id"]):
                try:
                    result = DeepFace.verify(
                        img1_path=face_img, img2_path=face_path,
                        model_name=RECOGNITION_MODEL,
                        distance_metric=RECOGNITION_METRIC,
                        enforce_detection=False, silent=True,
                    )
                    if result["distance"] < best_dist:
                        best_dist    = result["distance"]
                        best_profile = profile
                except Exception:
                    continue
        if best_profile and best_dist < RECOGNITION_THRESH:
            return best_profile["id"], best_profile["name"]
        return UNKNOWN_LABEL, UNKNOWN_LABEL


# ══════════════════════════════════════════════════════════════════════
#  Camera Thread
# ══════════════════════════════════════════════════════════════════════

class CameraSession:
    """
    Runs the OpenCV camera loop in a daemon thread so the terminal
    stays fully interactive while the preview window is open.
    """

    def __init__(self, engine: FaceEngine, session: SessionManager,
                 store: ProfileStore, camera_index: int = 0):
        self.engine        = engine
        self.session       = session
        self.store         = store
        self.camera_index  = camera_index
        self._stop_event   = threading.Event()
        self._reg_name: str | None = None   # set to capture next face
        self._reg_result: str | None = None
        self._thread: threading.Thread | None = None
        self._last_results: list[dict] = []

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        # destroyAllWindows must be called from the main thread on most platforms
        cv2.destroyAllWindows()
        cv2.waitKey(1)   # flush the event queue so windows actually close

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def request_register(self, name: str):
        """Signal the loop to capture the next face as this name."""
        self._reg_name = name

    # ── internal ──────────────────────────────────────────────────────

    def _loop(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            err(f"error al abrir la camara {self.camera_index}")
            return

        frame_count = 0
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # ── capture for registration ──────────────────────────
                if self._reg_name:
                    name = self._reg_name
                    self._reg_name = None
                    faces = self.engine.detect_faces(frame)
                    if faces:
                        fa   = faces[0]["facial_area"]
                        crop = frame[fa["y"]:fa["y"]+fa["h"], fa["x"]:fa["x"]+fa["w"]]
                        if name.startswith("__sample__"):
                            pid = name.replace("__sample__", "")
                            self.store.add_face_sample(pid, crop)
                            print(f"\n{C_GREEN}✔  Sample added to profile {pid}{C_RESET}")
                        else:
                            pid = self.store.add_profile(name, crop)
                            print(f"\n{C_GREEN}✔  Registered '{name}'  →  id {pid}{C_RESET}")
                    else:
                        print(f"\n{C_RED}✘  No face detected — try again{C_RESET}")

                # ── analysis every N frames ───────────────────────────
                if frame_count % ANALYZE_EVERY_N == 0:
                    self._last_results = self._process(frame)

                # ── draw overlay ──────────────────────────────────────
                display = frame.copy()
                for res in self._last_results:
                    self._draw(display, res)
                cv2.imshow("Face Profile Manager  [presiona Q para salir de esta ventana]", display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self._stop_event.set()
        finally:
            cap.release()
            # Do NOT call destroyAllWindows here — it must run on the main thread.
            # CameraSession.stop() handles it after join().

    def _process(self, frame: np.ndarray) -> list[dict]:
        results = []
        for fd in self.engine.detect_faces(frame):
            fa   = fd["facial_area"]
            crop = frame[fa["y"]:fa["y"]+fa["h"], fa["x"]:fa["x"]+fa["w"]]
            if crop.size == 0:
                continue
            pid, name    = self.engine.identify(crop)
            emotion_data = self.engine.analyze_emotion(crop)
            if emotion_data and pid != UNKNOWN_LABEL:
                self.session.record(pid, emotion_data["emotions"], emotion_data["dominant"])
            results.append({
                "box":      (fa["x"], fa["y"], fa["w"], fa["h"]),
                "name":     name,
                "pid":      pid,
                "emotion":  emotion_data["dominant"] if emotion_data else "—",
                "emotions": emotion_data["emotions"]  if emotion_data else {},
            })
        return results

    def _draw(self, frame: np.ndarray, res: dict):
        x, y, w, h = res["box"]
        color = (0, 200, 0) if res["name"] != UNKNOWN_LABEL else (0, 80, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{res['name']} | {res['emotion']}",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        bx = x + w + 5
        for i, (emo, score) in enumerate(res["emotions"].items()):
            by  = y + i * 18
            bl  = int(score * 0.8)
            cv2.rectangle(frame, (bx, by), (bx+bl, by+14), (200, 200, 50), -1)
            cv2.putText(frame, f"{emo[:3]} {score:.0f}%", (bx+bl+3, by+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)


# ══════════════════════════════════════════════════════════════════════
#  Interactive Terminal UI
# ══════════════════════════════════════════════════════════════════════

class App:
    def __init__(self):
        self.store   = ProfileStore()
        self.session = SessionManager()
        self.engine  = FaceEngine(self.store)
        self.camera: CameraSession | None = None

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self):
        clear()
        banner()
        info(f"Session  : {self.session.session_id}")
        info(f"Profiles : {len(self.store.list_profiles())} saved")

        while True:
            cam_label = f"{C_GREEN}RUNNING{C_RESET}" if self._cam_on() else f"{C_RED}OFF{C_RESET}"
            choice = choose_menu(
                f"Main Menu  [camera: {cam_label}]",
                [
                    "iniciar / apagar camara",
                    "Registrar nuevo perfil",
                    "Agregar muestra a perfil existente",
                    "Lista de perfiles existentes",
                    "Borrar perfil",
                    "Resumen de sesión actual",
                    "Historial de sesiones anteriores",
                    "Exit",
                ],
            )

            actions = [
                self._menu_camera,
                self._menu_register,
                self._menu_add_sample,
                self._menu_list_profiles,
                self._menu_delete_profile,
                self._menu_session_summary,
                self._menu_history,
                self._exit,
            ]
            actions[choice]()

    # ── Camera ────────────────────────────────────────────────────────

    def _cam_on(self) -> bool:
        return self.camera is not None and self.camera.is_running()

    def _menu_camera(self):
        if self._cam_on():
            if confirm("Cámara está encendida. ¿Detenerla?"):
                self.camera.stop()
                self.camera = None
                ok("camara apagada.")
        else:
            raw = prompt("Camera index [0]: ")
            idx = int(raw) if raw.isdigit() else 0
            self.camera = CameraSession(self.engine, self.session, self.store, idx)
            self.camera.start()
            ok("Camara iniciada.")
            info("Puedes seguir usando este menu mientras la camara está activa.")
            info("Presiona Q en la ventana de la camara para detenerla, o usa esta opción de nuevo.")
        pause()

    # ── Register new profile ──────────────────────────────────────────

    def _menu_register(self):
        section("Nuevo perfil")
        name = prompt("Nombre del perfil: ")
        if not name:
            err("Nombre no puede estar vacio.")
            pause()
            return

        choice = choose_menu("Registration source", [
            "Capture from live camera",
            "Load from image file",
        ])

        if choice == 0:
            if not self._cam_on():
                err("Camera is not running. Start it first (option 1).")
                pause()
                return
            print(f"\n{C_YELLOW}Position the face in the camera, then press Enter…{C_RESET}")
            input()
            self.camera.request_register(name)
            info("Capturing in background — watch the camera window.")

        else:
            path = prompt("Path to image file: ")
            img  = cv2.imread(path)
            if img is None:
                err(f"Could not read: {path}")
                pause()
                return
            pid = self.store.add_profile(name, img)
            ok(f"Registered '{name}'  →  id {pid}")

        pause()

    # ── Add face sample ───────────────────────────────────────────────

    def _menu_add_sample(self):
        section("Add Face Sample")
        profiles = self.store.list_profiles()
        if not profiles:
            err("Aun no hay perfiles guardados. Registra un perfil nuevo primero (opcion 2).")
            pause()
            return

        profile = self._pick_profile(profiles)
        if not profile:
            return

        choice = choose_menu("Sample source", [
            "Capture from live camera",
            "Load from image file",
        ])

        if choice == 0:
            if not self._cam_on():
                err("Camera is not running. Start it first (option 1).")
                pause()
                return
            print(f"\n{C_YELLOW}Position the face in the camera, then press Enter…{C_RESET}")
            input()
            self.camera.request_register(f"__sample__{profile['id']}")
            info("Capturing — the face will be added as an additional sample.")

        else:
            path = prompt("Path to image file: ")
            img  = cv2.imread(path)
            if img is None:
                err(f"Could not read: {path}")
                pause()
                return
            if self.store.add_face_sample(profile["id"], img):
                ok(f"Sample added to '{profile['name']}'.")
            else:
                err("Failed to add sample.")

        pause()

    # ── List profiles ─────────────────────────────────────────────────

    def _menu_list_profiles(self):
        section("Perfiles guardados")
        profiles = self.store.list_profiles()
        if not profiles:
            info("Aun no hay perfiles guardados.")
        else:
            for p in profiles:
                samples  = len(self.store.get_face_paths(p["id"]))
                readings = self.session.reading_count(p["id"])
                print(
                    f"  {C_BOLD}{p['id']}{C_RESET}  "
                    f"{C_CYAN}{p['name']:<22}{C_RESET}"
                    f"  {samples} sample(s)  |  {readings} reading(s) this session  "
                    f"{C_DIM}created {p['created_at'][:10]}{C_RESET}"
                )
        pause()

    # ── Delete profile ────────────────────────────────────────────────

    def _menu_delete_profile(self):
        section("Borrar perfil")
        profiles = self.store.list_profiles()
        if not profiles:
            err("No hay perfiles para borrar.")
            pause()
            return

        profile = self._pick_profile(profiles)
        if not profile:
            return

        if confirm(f"Delete '{profile['name']}' ({profile['id']})? This cannot be undone."):
            self.store.delete_profile(profile["id"])
            ok(f"Deleted '{profile['name']}'.")
        else:
            info("Cancelled.")
        pause()

    # ── Session summary ───────────────────────────────────────────────

    def _menu_session_summary(self):
        section(f"Session Summary  [{self.session.session_id}]")
        summaries = self.session.all_summaries()
        if not summaries:
            info("Sin datos registrados en esta sesión aún.")
        else:
            for s in summaries:
                profile = self.store.get_profile(s["profile_id"])
                name    = profile["name"] if profile else s["profile_id"]
                print(f"\n  {C_BOLD}{C_CYAN}{name}{C_RESET}  ({s['total_readings']} readings)")
                print(f"    Most common emotion  :  {C_YELLOW}{s['most_common_dominant']}{C_RESET}")
                print(f"    Dominant breakdown   :")
                max_count = max(s["dominant_counts"].values())
                for emo, count in sorted(s["dominant_counts"].items(), key=lambda x: -x[1]):
                    bar = "█" * int(count / max_count * 24)
                    print(f"      {emo:<12} {C_YELLOW}{bar:<24}{C_RESET} {count}")
                print(f"    Average scores       :")
                for emo, avg in sorted(s["average_emotions"].items(), key=lambda x: -x[1]):
                    bar = "░" * int(avg / 4)
                    print(f"      {emo:<12} {C_DIM}{bar:<25}{C_RESET} {avg:.1f}%")
        pause()

    # ── History ───────────────────────────────────────────────────────

    def _menu_history(self):
        section("Historical Sessions")
        if not SESSION_LOG.exists():
            info("Aun no hay sesiones registradas.")
            pause()
            return

        data = _loads(SESSION_LOG.read_text())
        if not data:
            info("Log file is empty.")
            pause()
            return

        for sid, sdata in sorted(data.items(), reverse=True):
            print(f"\n  {C_BOLD}{sid}{C_RESET}")
            profiles_in_session = sdata.get("profiles", {})
            if not profiles_in_session:
                info("    No data recorded.")
                continue
            for pid, pdata in profiles_in_session.items():
                s       = pdata.get("summary", {})
                profile = self.store.get_profile(pid)
                name    = profile["name"] if profile else pid
                print(
                    f"    {C_CYAN}{name:<22}{C_RESET}"
                    f"  {s.get('total_readings', 0)} readings  "
                    f"dominant: {C_YELLOW}{s.get('most_common_dominant', '—')}{C_RESET}"
                )
        pause()

    # ── Exit ──────────────────────────────────────────────────────────

    def _exit(self):
        self._shutdown()

    def _shutdown(self, interrupted: bool = False):
        """Cleanly stop everything, save, and hard-exit."""
        section("Apagando…")
        if self._cam_on():
            info("Deteniendo camara…")
            self.camera.stop()
        # Ensure all OpenCV windows are gone (safe here — we're on the main thread)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        info("Saving session data…")
        try:
            self.session.save()
            ok(f"Session saved  →  {SESSION_LOG}")
        except Exception as exc:
            err(f"Could not save session: {exc}")
        summaries = self.session.all_summaries()
        if summaries:
            print()
            for s in summaries:
                profile = self.store.get_profile(s["profile_id"])
                name    = profile["name"] if profile else s["profile_id"]
                print(f"  {C_CYAN}{name}{C_RESET}  —  "
                      f"{s['total_readings']} readings  |  "
                      f"dominant: {C_YELLOW}{s['most_common_dominant']}{C_RESET}")
        if interrupted:
            print(f"\n{C_DIM}Interrupted — goodbye.{C_RESET}\n")
        else:
            print(f"\n{C_CYAN}Goodbye!{C_RESET}\n")
        # os._exit skips Python's atexit/finaliser machinery and exits immediately,
        # which prevents daemon threads from printing errors after shutdown.
        os._exit(0)

    # ── Helper: profile picker ─────────────────────────────────────────

    def _pick_profile(self, profiles: list[dict]) -> dict | None:
        names = [f"{p['name']}  {C_DIM}({p['id']}){C_RESET}" for p in profiles]
        names.append("← Back")
        idx = choose_menu("Seleccionar perfil", names)
        if idx == len(profiles):
            return None
        return profiles[idx]


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    try:
        app.run()
    except KeyboardInterrupt:
        app._shutdown(interrupted=True)