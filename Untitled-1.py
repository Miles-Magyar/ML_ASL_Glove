import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import ai_edge_litert.interpreter as tflite
import time
import threading
import pyttsx3
from collections import deque

pred_buffer = deque(maxlen=5)

try:
    import winsound
    def play_beep(freq, duration):
        winsound.Beep(int(freq), int(duration))
except ImportError:
    def play_beep(freq, duration):
        print("\a")
        time.sleep(duration / 1000)

lost_frames = 0
MAX_LOST = 8
CUT_TOP = 0.3
CUT_BOTTOM = 0.0
CUT_SIDES = 0.15
MASK_INFLATION = 60
CONFIDENCE_THRESH = 0.5
cwd = os.path.dirname(os.path.abspath(__file__))


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#colors
C = {
    "bg":           "#1E1E2E",   #navy
    "sidebar":      "#161623",   #dark
    "panel":        "#252538",   #dark blue
    "surface":      "#2D2D44",   #card/widget surface
    "border":       "#3A3A55",   #subtle separator
    "accent":       "#007AFF",   #Xcode blue
    "accent2":      "#34C759",   #Xcode green
    "accent3":      "#FF9F0A",   #Xcode orange
    "accent4":      "#FF453A",   #Xcode red
    "text":         "#F2F2F7",   #primary text
    "text2":        "#8E8EA0",   #secondary/muted
    "text3":        "#48485E",   #disabled
    "highlight":    "#0A84FF",   #active selection
    "toolbar":      "#1C1C2D",   #top toolbar
}
#fonts
FONT_MONO  = ("Menlo", 11)
FONT_MONO_LG = ("Menlo", 14, "bold")
FONT_UI    = ("SF Pro Display", 11) if sys.platform == "darwin" else ("Helvetica", 11)
FONT_UI_SM = ("SF Pro Display", 9)  if sys.platform == "darwin" else ("Helvetica", 9)
FONT_BIG   = ("Menlo", 52, "bold")
FONT_MID   = ("Menlo", 28, "bold")
FONT_LABEL = ("SF Pro Display", 10, "bold") if sys.platform == "darwin" else ("Helvetica", 10, "bold")


#audio for blind
class AudioFeedback:
    def __init__(self): #noise constructor
        self.active = True
        self.mode = "blind"
        self.interval = 1.0
        self.last_beep = 0
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
        except:
            self.engine = None
        self.voice_lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self): #loop for beeping
        while self.active:
            if self.mode == "blind":
                now = time.time()
                if now - self.last_beep >= self.interval:
                    play_beep(1200, 60)
                    self.last_beep = time.time()
                time.sleep(0.02)
            else:
                time.sleep(0.5)

    def set_interval(self, val): #interval time
        self.interval = max(0.05, min(2.0, val))

    def speak(self, text): #speaking out loud -- curently not working
        def _speak():
            with self.voice_lock:
                if self.engine:
                    try:
                        self.engine.say(text)
                        self.engine.runAndWait()
                    except:
                        pass
        threading.Thread(target=_speak, daemon=True).start()


audio = AudioFeedback()

#tflite models
REQUIRED_MODELS = {
    "hand_det": "detector.tflite",
    "hand_lm":  "landmarks.tflite",
    "asl_ai":   "asl.tflite",
    "depth":    "midas.tflite",
    "glove_ai": "sign_language_glove.tflite",
    "face_emot": "face_emotion.tflite" 
}

loaded = {}
print("\n── LOADING AI ──────────────────────────")
for key, filename in REQUIRED_MODELS.items():
    path = os.path.join(cwd, filename)
    if not os.path.exists(path):
        print(f"  X    MISSING: {filename}")
        sys.exit()
    interp = tflite.Interpreter(model_path=path)
    interp.allocate_tensors()
    loaded[key] = {
        "model": interp,
        "in":    interp.get_input_details()[0]['index'],
        "outs":  interp.get_output_details(),
        "shape": interp.get_input_details()[0]['shape'] #store shape dynamically
    }
    print(f"  Loaded:  {filename}")
print("────────────────────────────────────────\n")

#labels for sign language models
LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")

#labels for face reading model
EMOTIONS = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


#tkinter widgets and window
root = tk.Tk()
root.title("Assistive AI  —  Multiple-Modes")
root.configure(bg=C["bg"])
root.geometry("1280x760")
root.minsize(1100, 680)


#layout: sidebar 52px - video panel - inspector 280px

#toolbar
toolbar = tk.Frame(root, bg=C["toolbar"], height=44)
toolbar.pack(side="top", fill="x")
toolbar.pack_propagate(False)

tk.Label(toolbar, text="Assistive AI", font=FONT_LABEL,
         fg=C["accent"], bg=C["toolbar"]).pack(side="left", padx=18, pady=12)

lbl_mode_badge = tk.Label(toolbar, text="● BLIND",
                           font=("Menlo", 9, "bold"),
                           fg=C["accent3"], bg=C["toolbar"])
lbl_mode_badge.pack(side="left", padx=6)

lbl_status_bar = tk.Label(toolbar, text="System initialising…",
                           font=FONT_UI_SM, fg=C["text2"], bg=C["toolbar"])
lbl_status_bar.pack(side="right", padx=18)

separator_h = tk.Frame(root, bg=C["border"], height=1)
separator_h.pack(fill="x")

#main body
body = tk.Frame(root, bg=C["bg"])
body.pack(fill="both", expand=True)

#left sidebar
sidebar = tk.Frame(body, bg=C["sidebar"], width=56)
sidebar.pack(side="left", fill="y")
sidebar.pack_propagate(False)

tk.Frame(sidebar, bg=C["border"], height=1).pack(fill="x")

SIDEBAR_BTNS = []
def make_sidebar_btn(emoji, label, cmd, active=False):
    frame = tk.Frame(sidebar, bg=C["sidebar"], cursor="hand2")
    frame.pack(fill="x", pady=1)

    indicator = tk.Frame(frame, bg=C["accent"] if active else C["sidebar"], width=3, height=44)
    indicator.pack(side="left", fill="y")

    inner = tk.Frame(frame, bg=C["sidebar"])
    inner.pack(fill="both", expand=True)

    ico = tk.Label(inner, text=emoji, font=("Arial", 18),
                   bg=C["sidebar"] if not active else C["surface"],
                   fg=C["text"] if active else C["text2"],
                   width=3, pady=10)
    ico.pack()

    def on_enter(e):
        if not ico.is_active:
            ico.config(bg=C["surface"], fg=C["text"])
            inner.config(bg=C["surface"])
    def on_leave(e):
        if not ico.is_active:
            ico.config(bg=C["sidebar"], fg=C["text2"])
            inner.config(bg=C["sidebar"])
    def on_click(e):
        cmd()
        for b in SIDEBAR_BTNS:
            b["indicator"].config(bg=C["sidebar"])
            b["ico"].config(bg=C["sidebar"], fg=C["text2"])
            b["inner"].config(bg=C["sidebar"])
            b["ico"].is_active = False
        indicator.config(bg=C["accent"])
        ico.config(bg=C["surface"], fg=C["text"])
        inner.config(bg=C["surface"])
        ico.is_active = True

    ico.is_active = active
    ico.bind("<Enter>", on_enter)
    ico.bind("<Leave>", on_leave)
    frame.bind("<Button-1>", on_click)
    ico.bind("<Button-1>", on_click)
    inner.bind("<Button-1>", on_click)

    SIDEBAR_BTNS.append({"indicator": indicator, "ico": ico, "inner": inner})
    return frame

#center video panel
video_panel = tk.Frame(body, bg=C["bg"])
video_panel.pack(side="left", fill="both", expand=True)

video_header = tk.Frame(video_panel, bg=C["toolbar"], height=32)
video_header.pack(fill="x")
video_header.pack_propagate(False)
tk.Label(video_header, text="Camera Feed", font=FONT_UI_SM,
         fg=C["text2"], bg=C["toolbar"]).pack(side="left", padx=14, pady=8)

lbl_vid = tk.Label(video_panel, bg="black", relief="flat", bd=0)
lbl_vid.pack(fill="both", expand=True, padx=14, pady=(8, 14))

#right inspector panel
sep_v = tk.Frame(body, bg=C["border"], width=1)
sep_v.pack(side="left", fill="y")

inspector = tk.Frame(body, bg=C["panel"], width=280)
inspector.pack(side="right", fill="y")
inspector.pack_propagate(False)

insp_header = tk.Frame(inspector, bg=C["toolbar"], height=32)
insp_header.pack(fill="x")
insp_header.pack_propagate(False)
tk.Label(insp_header, text="Inspector", font=FONT_UI_SM,
         fg=C["text2"], bg=C["toolbar"]).pack(side="left", padx=14, pady=8)

#blind mode proximity display
blind_card = tk.Frame(inspector, bg=C["surface"], relief="flat")
blind_card.pack(fill="x", padx=10, pady=(12, 0))

tk.Label(blind_card, text="PROXIMITY", font=("Menlo", 8),
         fg=C["text3"], bg=C["surface"]).pack(anchor="w", padx=12, pady=(10, 0))

lbl_prox_value = tk.Label(blind_card, text="—", font=FONT_BIG,
                            fg=C["accent3"], bg=C["surface"])
lbl_prox_value.pack(pady=(4, 0))

lbl_prox_sub = tk.Label(blind_card, text="audio interval  —",
                          font=("Menlo", 10), fg=C["text2"], bg=C["surface"])
lbl_prox_sub.pack(pady=(0, 10))

blind_bar_frame = tk.Frame(inspector, bg=C["panel"])
blind_bar_frame.pack(fill="x", padx=10, pady=(6, 0))
tk.Label(blind_bar_frame, text="PROXIMITY LEVEL", font=("Menlo", 8),
         fg=C["text3"], bg=C["panel"]).pack(anchor="w")
canvas_prox = tk.Canvas(blind_bar_frame, height=6, bg=C["surface"],
                        highlightthickness=0, relief="flat")
canvas_prox.pack(fill="x", pady=(3, 0))
prox_fill = canvas_prox.create_rectangle(0, 0, 0, 6, fill=C["accent3"], width=0)

def update_prox_bar(pct):
    w = canvas_prox.winfo_width()
    canvas_prox.coords(prox_fill, 0, 0, int(w * pct / 100), 6)
    color = C["accent4"] if pct > 70 else C["accent3"] if pct > 35 else C["accent2"]
    canvas_prox.itemconfig(prox_fill, fill=color)

# prediction displays for deaf/glove/emotion modes
pred_card = tk.Frame(inspector, bg=C["surface"], relief="flat")
#not packed initially, shown only in deaf/glove/emotion

tk.Label(pred_card, text="PREDICTION", font=("Menlo", 8),
         fg=C["text3"], bg=C["surface"]).pack(anchor="w", padx=12, pady=(10, 0))

lbl_letter = tk.Label(pred_card, text="—", font=FONT_BIG,
                       fg=C["accent"], bg=C["surface"])
lbl_letter.pack(pady=(4, 0))

lbl_conf = tk.Label(pred_card, text="confidence  0%",
                     font=("Menlo", 10), fg=C["text2"], bg=C["surface"])
lbl_conf.pack(pady=(0, 10))

#confidence
bar_frame = tk.Frame(inspector, bg=C["panel"])
# not packed initially
tk.Label(bar_frame, text="CONFIDENCE", font=("Menlo", 8),
         fg=C["text3"], bg=C["panel"]).pack(anchor="w")
canvas_bar = tk.Canvas(bar_frame, height=6, bg=C["surface"],
                        highlightthickness=0, relief="flat")
canvas_bar.pack(fill="x", pady=(3, 0))
bar_fill = canvas_bar.create_rectangle(0, 0, 0, 6, fill=C["accent"], width=0)

def update_conf_bar(pct):
    w = canvas_bar.winfo_width()
    canvas_bar.coords(bar_fill, 0, 0, int(w * pct / 100), 6)
    color = C["accent2"] if pct > 70 else C["accent"] if pct > 40 else C["accent3"]
    canvas_bar.itemconfig(bar_fill, fill=color)

#divider
def insp_divider():
    tk.Frame(inspector, bg=C["border"], height=1).pack(fill="x", padx=0, pady=8)

insp_divider()

#ai view
brain_section = tk.Frame(inspector, bg=C["panel"])
brain_section.pack(fill="x", padx=10)

lbl_brain_title = tk.Label(brain_section, text="DEPTH MAP",
                             font=("Menlo", 8), fg=C["text3"], bg=C["panel"])
lbl_brain_title.pack(anchor="w", pady=(0, 4))

brain_card = tk.Frame(brain_section, bg=C["surface"])
brain_card.pack(fill="x")
lbl_brain = tk.Label(brain_card, bg=C["surface"], width=150, height=150, relief="flat")
lbl_brain.pack(padx=1, pady=1)

insp_divider()

#sliders
sliders_section = tk.Frame(inspector, bg=C["panel"])
sliders_section.pack(fill="x", padx=10)

def section_label(parent, text):
    tk.Label(parent, text=text, font=("Menlo", 8),
             fg=C["text3"], bg=C["panel"]).pack(anchor="w", pady=(0, 6))

def styled_slider(parent, label, variable, from_, to_, resolution=1):
    row = tk.Frame(parent, bg=C["panel"])
    row.pack(fill="x", pady=2)
    tk.Label(row, text=label, font=("Menlo", 9), fg=C["text2"],
             bg=C["panel"], width=8, anchor="w").pack(side="left")
    style = ttk.Style()
    style.theme_use("default")
    style.configure("Xcode.Horizontal.TScale",
                    background=C["panel"],
                    troughcolor=C["surface"],
                    sliderlength=14,
                    sliderrelief="flat")
    s = ttk.Scale(row, variable=variable, from_=from_, to=to_,
                  orient="horizontal", style="Xcode.Horizontal.TScale")
    s.pack(side="left", fill="x", expand=True, padx=(4, 0))
    val_lbl = tk.Label(row, text="", font=("Menlo", 8),
                       fg=C["accent"], bg=C["panel"], width=5, anchor="e")
    val_lbl.pack(side="right")
    def update_label(*_):
        v = variable.get()
        val_lbl.config(text=f"{v:.1f}" if isinstance(resolution, float) else str(int(v)))
    variable.trace_add("write", update_label)
    update_label()
    return s

#image adjust - deaf
img_frame = tk.Frame(sliders_section, bg=C["panel"])
img_bright = tk.DoubleVar(value=1.0)
img_sat    = tk.DoubleVar(value=1.0)

def build_img_sliders():
    section_label(img_frame, "IMAGE ADJUST")
    styled_slider(img_frame, "Bright", img_bright, 0.0, 2.0, 0.1)
    styled_slider(img_frame, "Sat",    img_sat,    0.0, 2.0, 0.1)

build_img_sliders()

#HSV Calibration (glove mode) - masking
hsv_frame = tk.Frame(sliders_section, bg=C["panel"])
h_min = tk.IntVar(value=37);  s_min = tk.IntVar(value=33);  v_min = tk.IntVar(value=102)
h_max = tk.IntVar(value=82);  s_max = tk.IntVar(value=205); v_max = tk.IntVar(value=255)

def build_hsv_sliders():
    section_label(hsv_frame, "HSV CALIBRATION")
    styled_slider(hsv_frame, "H Min", h_min, 0, 179)
    styled_slider(hsv_frame, "H Max", h_max, 0, 179)
    styled_slider(hsv_frame, "S Min", s_min, 0, 255)
    styled_slider(hsv_frame, "S Max", s_max, 0, 255)
    styled_slider(hsv_frame, "V Min", v_min, 0, 255)
    styled_slider(hsv_frame, "V Max", v_max, 0, 255)

build_hsv_sliders()

insp_divider()

#key hints
hints = tk.Frame(inspector, bg=C["panel"])
hints.pack(fill="x", padx=10)
tk.Label(hints, text="KEYBOARD", font=("Menlo", 8),
         fg=C["text3"], bg=C["panel"]).pack(anchor="w", pady=(0, 4))
for key, desc in [("I", "Invert mask"), ("F", "Flip camera")]:
    row = tk.Frame(hints, bg=C["panel"])
    row.pack(fill="x", pady=1)
    tk.Label(row, text=f" {key} ", font=("Menlo", 9, "bold"),
             fg=C["bg"], bg=C["accent"],
             padx=4, pady=1, relief="flat").pack(side="left")
    tk.Label(row, text=desc, font=("Menlo", 9),
             fg=C["text2"], bg=C["panel"]).pack(side="left", padx=6)


#state
state = {
    "invert":       False,
    "flip":         True,
    "current_mode": "blind",
    "last_mode":    "blind",
}

#mode switching
def _clear_sliders():
    img_frame.pack_forget()
    hsv_frame.pack_forget()

def _show_blind_widgets():
    blind_card.pack(fill="x", padx=10, pady=(12, 0))
    blind_bar_frame.pack(fill="x", padx=10, pady=(6, 0))
    pred_card.pack_forget()
    bar_frame.pack_forget()

def _show_pred_widgets():
    blind_card.pack_forget()
    blind_bar_frame.pack_forget()
    pred_card.pack(fill="x", padx=10, pady=(12, 0))
    bar_frame.pack(fill="x", padx=10, pady=(6, 0))

def switch_to_blind():
    state["current_mode"] = "blind"
    audio.mode = "blind"
    lbl_brain_title.config(text="DEPTH MAP")
    lbl_mode_badge.config(text="● BLIND", fg=C["accent3"])
    _clear_sliders()
    _show_blind_widgets()

def switch_to_deaf():
    state["current_mode"] = "deaf"
    audio.mode = "deaf"
    lbl_letter.config(font=FONT_BIG) #reset font size
    lbl_brain_title.config(text="ASL — BARE HAND")
    lbl_mode_badge.config(text="● DEAF", fg=C["accent2"])
    _clear_sliders()
    _show_pred_widgets()
    img_frame.pack(fill="x")

def switch_to_glove():
    state["current_mode"] = "glove"
    audio.mode = "glove"
    lbl_letter.config(font=FONT_BIG)
    lbl_brain_title.config(text="COLOR GLOVE")
    lbl_mode_badge.config(text="● GLOVE", fg=C["accent"])
    _clear_sliders()
    _show_pred_widgets()
    hsv_frame.pack(fill="x")

def switch_to_emotion():
    state["current_mode"] = "emotion"
    audio.mode = "emotion"
    lbl_letter.config(font=FONT_MID)
    lbl_brain_title.config(text="FACE DETECTION")
    lbl_mode_badge.config(text="● EMOTION", fg=C["accent4"])
    _clear_sliders()
    _show_pred_widgets()


#sidebar buttons
make_sidebar_btn("<o>", "Blind",  switch_to_blind, active=True)
make_sidebar_btn("/>", "Deaf",   switch_to_deaf)
make_sidebar_btn("Glove", "Glove",  switch_to_glove)
make_sidebar_btn("Face", "Emotion", switch_to_emotion)


#key binding
def toggle_invert(e): state["invert"] = not state["invert"]
def toggle_flip(e):   state["flip"]   = not state["flip"]
root.bind('i', toggle_invert)
root.bind('f', toggle_flip)


#camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

start_time = time.time()
audio.speak("System ready. Mode is blind.")
lbl_status_bar.config(text="System ready  ·  Blind mode active")
# blind is the default — show proximity panel, hide prediction panel
pred_card.pack_forget()
bar_frame.pack_forget()


#loop
def loop():
    global lost_frames

    #announce mode change
    if state["current_mode"] != state["last_mode"]:
        msgs = {"blind": "Blind mode", "deaf": "Deaf mode", "glove": "Color glove mode", "emotion": "Emotion mode"}
        audio.speak(f"Switched to {msgs.get(state['current_mode'], '')}")
        lbl_status_bar.config(
            text=f"Mode: {state['current_mode'].capitalize()}  ·  {time.strftime('%H:%M:%S')}")
        state["last_mode"] = state["current_mode"]

    ok, raw_frame = cap.read()
    if not ok:
        root.after(10, loop)
        return

    if state["flip"]:
        raw_frame = cv2.flip(raw_frame, 1)

    h_raw, w_raw, _ = raw_frame.shape
    y1 = int(h_raw * CUT_TOP)
    y2 = int(h_raw * (1.0 - CUT_BOTTOM))
    x1 = int(w_raw * CUT_SIDES)
    x2 = int(w_raw * (1.0 - CUT_SIDES))
    frame = raw_frame[y1:y2, x1:x2].copy()
    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape
    clean_frame = frame.copy()

    box_size = 280
    cx, cy = w // 2, h // 2
    fx1 = max(0, cx - box_size // 2)
    fy1 = max(0, cy - box_size // 2)
    fx2 = min(w, cx + box_size // 2)
    fy2 = min(h, cy + box_size // 2)

    # startup overlay
    elapsed = time.time() - start_time
    if elapsed < 10:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, "System Ready — Blind Mode",
                    (14, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 122, 255), 1, cv2.LINE_AA)

    # blind
    if state["current_mode"] == "blind":
        img_in   = cv2.resize(frame, (256, 256))
        img_rgb  = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        inp      = np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)

        dm = loaded["depth"]
        dm["model"].set_tensor(dm["in"], inp)
        dm["model"].invoke()
        out = dm["model"].get_tensor(dm["outs"][0]['index'])[0]

        d_min, d_max = out.min(), out.max()
        d_range = max(d_max - d_min, 1e-6)
        depth_norm = (out - d_min) / d_range
        depth_u8   = (depth_norm * 255).astype(np.uint8)
        depth_col  = cv2.applyColorMap(depth_u8, cv2.COLORMAP_MAGMA)

        view = cv2.resize(depth_col, (150, 150))
        img_view = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)))
        lbl_brain.config(image=img_view); lbl_brain.img = img_view

        mid, roi = 128, 30
        proximity = float(np.mean(depth_norm[mid - roi:mid + roi, mid - roi:mid + roi]))
        audio.set_interval(1.5 - proximity * 1.45)

        prox_pct = int(proximity * 100)
        lbl_prox_value.config(text=f"{prox_pct}%")
        lbl_prox_sub.config(text=f"audio interval  {audio.interval:.2f}s")
        update_prox_bar(prox_pct)

    #glove
    elif state["current_mode"] == "glove":
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 122, 255), 2)

        roi_color = clean_frame[fy1:fy2, fx1:fx2]
        hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        lower = np.array([h_min.get(), s_min.get(), v_min.get()])
        upper = np.array([h_max.get(), s_max.get(), v_max.get()])
        roi_mask = cv2.inRange(hsv, lower, upper)

        if cv2.countNonZero(roi_mask) > 500:
            lost_frames = 0
            final_in = cv2.resize(roi_mask, (28, 28), interpolation=cv2.INTER_AREA)
            if state["invert"]:
                final_in = cv2.bitwise_not(final_in)

            view = cv2.resize(final_in, (150, 150), interpolation=cv2.INTER_NEAREST)
            img_view = ImageTk.PhotoImage(Image.fromarray(view))
            lbl_brain.config(image=img_view); lbl_brain.img = img_view

            feat = final_in.reshape(1, 28, 28, 1).astype(np.float32) / 255.0
            gm = loaded["glove_ai"]
            gm["model"].set_tensor(gm["in"], feat)
            gm["model"].invoke()
            preds    = gm["model"].get_tensor(gm["outs"][0]['index'])[0]
            idx      = int(np.argmax(preds))
            conf     = float(preds[idx])

            pred_buffer.append(idx)
            final_idx = max(set(pred_buffer), key=pred_buffer.count) if len(pred_buffer) == 5 else idx

            if conf > 0.4:
                lbl_letter.config(text=LABELS[final_idx] if final_idx < len(LABELS) else "?",
                                   fg=C["accent"])
                lbl_conf.config(text=f"confidence  {int(conf * 100)}%")
                update_conf_bar(int(conf * 100))
        else:
            lost_frames += 1
            if lost_frames >= MAX_LOST:
                lbl_letter.config(text="—", fg=C["text3"])
                lbl_conf.config(text="confidence  0%")
                update_conf_bar(0)

    #emotion
    elif state["current_mode"] == "emotion":
        gray = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY)
        
        #detect face
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            lost_frames = 0
            
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            fx, fy, fw, fh = faces[0]

            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 122, 255), 2)
            
            em_mod = loaded["face_emot"]
            in_shape = em_mod["shape"]
            req_h, req_w = in_shape[1], in_shape[2]
            channels = in_shape[3] if len(in_shape) > 3 else 1

            if channels == 1:
                roi_face = gray[fy:fy+fh, fx:fx+fw]
            else:
                roi_face = clean_frame[fy:fy+fh, fx:fx+fw]

            final_in = cv2.resize(roi_face, (req_w, req_h))

            view = cv2.resize(final_in, (150, 150))
            if channels == 1:
                view_rgb = cv2.cvtColor(view, cv2.COLOR_GRAY2RGB)
            else:
                view_rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
                
            img_view = ImageTk.PhotoImage(Image.fromarray(view_rgb))
            lbl_brain.config(image=img_view); lbl_brain.img = img_view

            feat = final_in.reshape(1, req_h, req_w, channels).astype(np.float32) / 255.0
            
            em_mod["model"].set_tensor(em_mod["in"], feat)
            em_mod["model"].invoke()
            preds = em_mod["model"].get_tensor(em_mod["outs"][0]['index'])[0]
            
            idx  = int(np.argmax(preds))
            conf = float(preds[idx])

            pred_buffer.append(idx)
            final_idx = max(set(pred_buffer), key=pred_buffer.count) if len(pred_buffer) == 5 else idx

            lbl_letter.config(text=EMOTIONS[final_idx].upper() if final_idx < len(EMOTIONS) else "?", 
                              fg=C["accent4"])
            lbl_conf.config(text=f"confidence  {int(conf * 100)}%")
            update_conf_bar(int(conf * 100))

        else:
            lost_frames += 1
            if lost_frames >= MAX_LOST:
                lbl_letter.config(text="—", fg=C["text3"])
                lbl_conf.config(text="confidence  0%")
                update_conf_bar(0)

    #deaf
    elif state["current_mode"] == "deaf":
        img_in  = cv2.resize(frame, (256, 256))
        img_rgb = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        inp     = np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)

        det = loaded["hand_det"]
        det["model"].set_tensor(det["in"], inp)
        det["model"].invoke()
        out0   = det["model"].get_tensor(det["outs"][0]['index'])
        out1   = det["model"].get_tensor(det["outs"][1]['index'])
        scores = out0.flatten() if out0.shape[-1] < out1.shape[-1] else out1.flatten()

        if np.max(sigmoid(scores)) > CONFIDENCE_THRESH:
            lost_frames = 0
            lm_mod = loaded["hand_lm"]
            lm_mod["model"].set_tensor(lm_mod["in"], inp)
            lm_mod["model"].invoke()

            lm_raw = None
            for out in lm_mod["outs"]:
                t = lm_mod["model"].get_tensor(out['index']).flatten()
                if len(t) >= 63:
                    lm_raw = t
                    break

            if lm_raw is not None:
                max_val = float(np.max(lm_raw))
                points_list = []
                for i in range(0, 63, 3):
                    vx, vy = lm_raw[i], lm_raw[i + 1]
                    if max_val <= 1.0:
                        px, py = int(vx * w), int(vy * h)
                    else:
                        px, py = int((vx / 256) * w), int((vy / 256) * h)
                    points_list.append([px, py])
                    cv2.circle(frame, (px, py), 2, (0, 122, 255), -1)

                if points_list:
                    mask   = np.zeros((h, w), dtype=np.uint8)
                    hull   = cv2.convexHull(np.array(points_list, dtype=np.int32))
                    cv2.fillConvexPoly(mask, hull, 255)
                    kernel = np.ones((MASK_INFLATION, MASK_INFLATION), np.uint8)
                    mask   = cv2.dilate(mask, kernel, iterations=1)

                    b_val = img_bright.get()
                    s_val = img_sat.get()
                    hsv_h = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                    hsv_h[:, :, 1] = np.clip(hsv_h[:, :, 1] * s_val, 0, 255)
                    hsv_h[:, :, 2] = np.clip(hsv_h[:, :, 2] * b_val, 0, 255)
                    adj_hand  = cv2.cvtColor(hsv_h.astype(np.uint8), cv2.COLOR_HSV2BGR)
                    white_bg  = np.ones_like(clean_frame) * 255
                    mask_3ch  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    final_comp = np.where(mask_3ch > 0, adj_hand, white_bg)

                    hx, hy, hw2, hh2 = cv2.boundingRect(np.array(points_list, dtype=np.int32))
                    dyn = max(hw2, hh2) + 40
                    mcx, mcy = hx + hw2 // 2, hy + hh2 // 2
                    dx1 = max(0, mcx - dyn // 2); dy1 = max(0, mcy - dyn // 2)
                    dx2 = min(w, mcx + dyn // 2); dy2 = min(h, mcy + dyn // 2)
                    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 122, 255), 2)

                    roi = final_comp[dy1:dy2, dx1:dx2]
                    if roi.size > 0:
                        gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        final_in = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
                        if state["invert"]:
                            final_in = cv2.bitwise_not(final_in)

                        view = cv2.resize(final_in, (150, 150), interpolation=cv2.INTER_NEAREST)
                        img_view = ImageTk.PhotoImage(Image.fromarray(view))
                        lbl_brain.config(image=img_view); lbl_brain.img = img_view

                        feat = final_in.reshape(1, 28, 28, 1).astype(np.float32) / 255.0
                        asl  = loaded["asl_ai"]
                        asl["model"].set_tensor(asl["in"], feat)
                        asl["model"].invoke()
                        preds    = asl["model"].get_tensor(asl["outs"][0]['index'])[0]
                        idx      = int(np.argmax(preds))
                        conf     = float(preds[idx])

                        pred_buffer.append(idx)
                        final_idx = max(set(pred_buffer), key=pred_buffer.count) if len(pred_buffer) == 5 else idx

                        if conf > 0.4:
                            lbl_letter.config(
                                text=LABELS[final_idx] if final_idx < len(LABELS) else "?",
                                fg=C["accent2"])
                            lbl_conf.config(text=f"confidence  {int(conf * 100)}%")
                            update_conf_bar(int(conf * 100))
        else:
            lost_frames += 1
            if lost_frames >= MAX_LOST:
                lbl_letter.config(text="—", fg=C["text3"])
                lbl_conf.config(text="confidence  0%")
                update_conf_bar(0)

    #rendering
    img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    lbl_vid.config(image=img_tk)
    lbl_vid.img = img_tk

    root.after(10, loop)


loop()
root.mainloop()