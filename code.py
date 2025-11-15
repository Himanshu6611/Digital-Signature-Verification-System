import tkinter as tk
from tkinter import simpledialog, messagebox
import time
import os
import numpy as np

TEMPLATE_DIR = "sig_templates"
os.makedirs(TEMPLATE_DIR, exist_ok=True)

CANVAS_W, CANVAS_H = 800, 300
RESAMPLE_N = 100
THRESHOLD = 0.75   # similarity threshold

# ----------------- Capture strokes -----------------
class SignatureCanvas(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.strokes = []
        self.current = []

        self.bind("<ButtonPress-1>", self.on_down)
        self.bind("<B1-Motion>", self.on_move)
        self.bind("<ButtonRelease-1>", self.on_up)

    def on_down(self, event):
        self.current = []
        t = int(time.time() * 1000)
        p = getattr(event, "pressure", 0.5) if hasattr(event, "pressure") else 0.5
        self.current.append((event.x, event.y, t, p))

    def on_move(self, event):
        t = int(time.time() * 1000)
        p = getattr(event, "pressure", 0.5) if hasattr(event, "pressure") else 0.5

        last = self.current[-1]
        self.create_line(last[0], last[1], event.x, event.y,
                         width=2, capstyle=tk.ROUND, smooth=True)

        self.current.append((event.x, event.y, t, p))

    def on_up(self, event):
        if len(self.current) > 0:
            self.strokes.append(self.current)
            self.current = []

    def clear(self):
        self.delete("all")
        self.strokes = []
        self.current = []


# ----------------- Feature extraction utils -----------------
def flatten_strokes(strokes):
    pts = []
    for stroke in strokes:
        for p in stroke:
            pts.append((float(p[0]), float(p[1]), float(p[2]), float(p[3])))
    return pts


def resample_points(points, n=RESAMPLE_N):
    if not points:
        return np.zeros((n, 2), dtype=float)

    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)

    diffs = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    dist = np.concatenate([[0.0], np.cumsum(diffs)])

    total = dist[-1]
    if total == 0:
        arr = np.vstack([xs, ys]).T
        return np.tile(arr[0], (n, 1))

    target = np.linspace(0, total, n)
    rx = np.interp(target, dist, xs)
    ry = np.interp(target, dist, ys)

    return np.vstack([rx, ry]).T


def normalize_shape(arr):
    arr = np.array(arr, dtype=float)
    minxy = arr.min(axis=0)
    arr = arr - minxy

    span = arr.max()
    if span > 0:
        arr = arr / span

    return arr.flatten()


def compute_speed_stats(points):
    if len(points) < 2:
        return 0.0, 0.0

    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    ts = np.array([p[2] for p in points], dtype=float)

    dt = np.diff(ts)
    dist = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)

    dt[dt == 0] = 1.0
    speeds = dist / dt

    return float(speeds.mean()), float(speeds.std())


def pressure_stats(points):
    if not points:
        return 0.5, 0.0

    ps = np.array([p[3] for p in points], dtype=float)
    return float(ps.mean()), float(ps.std())


def dtw_distance(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    na, nb = len(a), len(b)
    D = np.full((na + 1, nb + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = np.linalg.norm(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    return float(D[na, nb])


# ----------------- High-level feature extraction -----------------
def extract_features_from_strokes(strokes):
    pts = flatten_strokes(strokes)
    shape_resampled = resample_points(pts, RESAMPLE_N)
    shape_norm = normalize_shape(shape_resampled)

    mean_speed, std_speed = compute_speed_stats(pts)
    mean_p, std_p = pressure_stats(pts)

    return {
        "shape_vec": shape_norm,
        "mean_speed": mean_speed,
        "std_speed": std_speed,
        "mean_pressure": mean_p,
        "std_pressure": std_p
    }


# ----------------- Save/Load templates -----------------
def save_template(user, features):
    path = os.path.join(TEMPLATE_DIR, f"{user}_template.npz")
    np.savez(path, **features)
    return path


def load_template(user):
    path = os.path.join(TEMPLATE_DIR, f"{user}_template.npz")
    if not os.path.exists(path):
        return None

    data = np.load(path)
    return {
        "shape_vec": data["shape_vec"],
        "mean_speed": float(data["mean_speed"]),
        "std_speed": float(data["std_speed"]),
        "mean_pressure": float(data["mean_pressure"]),
        "std_pressure": float(data["std_pressure"])
    }


# ----------------- Compare features -----------------
def compare_features(featA, featB):
    A = featA["shape_vec"].reshape(-1, 2)
    B = featB["shape_vec"].reshape(-1, 2)

    shape_dist = dtw_distance(A, B)

    def norm_diff(a, b):
        return abs(a - b) / (abs(a) + abs(b) + 1e-6)

    spd_diff = norm_diff(featA["mean_speed"], featB["mean_speed"])
    pres_diff = norm_diff(featA["mean_pressure"], featB["mean_pressure"])

    shape_norm = shape_dist / (10 + shape_dist)

    combined = 0.7 * shape_norm + 0.2 * spd_diff + 0.1 * pres_diff
    similarity = 1 - combined
    similarity = max(0, min(1, similarity))

    return {
        "similarity": similarity,
        "shape_dist": shape_dist,
        "spd_diff": spd_diff,
        "pres_diff": pres_diff
    }


# ----------------- GUI actions -----------------
def enroll_action(canvas):
    if not canvas.strokes:
        messagebox.showwarning("Enroll", "Draw signature before enrolling.")
        return

    user = simpledialog.askstring("Enroll", "Enter username to enroll:")
    if not user:
        return

    feats = extract_features_from_strokes(canvas.strokes)
    save_template(user, feats)

    messagebox.showinfo("Enroll", f"Enrolled '{user}'. Template saved.")


def auth_action(canvas):
    if not canvas.strokes:
        messagebox.showwarning("Authenticate", "Draw signature first.")
        return

    user = simpledialog.askstring("Authenticate", "Enter username:")
    if not user:
        return

    template = load_template(user)
    if template is None:
        messagebox.showerror("Authenticate", f"User '{user}' not enrolled.")
        return

    probe = extract_features_from_strokes(canvas.strokes)
    res = compare_features(template, probe)

    sim = res["similarity"]
    accepted = sim >= THRESHOLD

    msg = (
        f"Similarity: {sim:.3f}\n"
        f"Result: {'ACCEPTED' if accepted else 'REJECTED'}\n\n"
        f"Details:\n"
        f"shape_dist = {res['shape_dist']:.3f}\n"
        f"spd_diff   = {res['spd_diff']:.3f}\n"
        f"pres_diff  = {res['pres_diff']:.3f}"
    )

    messagebox.showinfo("Authenticate", msg)


def build_gui():
    root = tk.Tk()
    root.title("Signature Verification (Python Only)")

    frame = tk.Frame(root)
    frame.pack(padx=10, pady=8)

    canvas = SignatureCanvas(frame, width=CANVAS_W, height=CANVAS_H, bg="white")
    canvas.pack()

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=6)

    tk.Button(btn_frame, text="Enroll", width=12,
              command=lambda: enroll_action(canvas)).grid(row=0, column=0, padx=6)

    tk.Button(btn_frame, text="Authenticate", width=12,
              command=lambda: auth_action(canvas)).grid(row=0, column=1, padx=6)

    tk.Button(btn_frame, text="Clear", width=12,
              command=canvas.clear).grid(row=0, column=2, padx=6)

    tk.Button(btn_frame, text="Quit", width=12,
              command=root.destroy).grid(row=0, column=3, padx=6)

    tk.Label(root, text="Draw signature → Enroll → Authenticate").pack(pady=4)

    root.mainloop()


if __name__ == "__main__":
    build_gui()
