"""Visualize per-frame hand velocity, the signal LongVIL uses for keyframe extraction.

Usage:
    python plot_hand_vel.py <path/to/video> [--out_dir DIR] [--sigma 5] [--min_gap 15]

Reproduces the velocity computation in get_frames.py (FrameExtractor):
  * MediaPipe HandLandmarker (VIDEO mode, 2 hands, confidence 0.2)
  * hand position = mean of the 21 landmarks, in pixels
  * speed[i] = ||pos[i] - pos[i+1]||  (px/frame; NaN when the hand is missing)
  * NaNs linearly interpolated, then Gaussian smoothing (sigma=5)
  * valleys of the smoothed curve = local minima, thinned so consecutive
    valleys of the same hand are >= 15 frames apart
  * keyframes = sorted union of both hands' valleys + first + last frame

Writes <stem>_hand_velocity.png and <stem>_hand_velocity.csv to --out_dir.
Standalone: it does not import get_frames.py (which needs the legacy
`mediapipe.solutions` API removed in recent mediapipe releases).
"""
import argparse
import csv
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
HANDS = ["Left", "Right"]  # handedness labels as reported by MediaPipe
COLORS = {"Left": "#2a78d6", "Right": "#eb6834"}  # categorical slots 1-2 (validated)
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def track_hands(video_path, model_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pos = {h: np.full((n, 2), np.nan) for h in HANDS}
    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.2,
        min_hand_presence_confidence=0.2,
        min_tracking_confidence=0.2,
    )
    i = 0
    with vision.HandLandmarker.create_from_options(options) as lm:
        while True:
            ok, frame = cap.read()
            if not ok or i >= n:
                break
            h, w = frame.shape[:2]
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = lm.detect_for_video(img, int(round(i * 1000.0 / fps)))
            for lms, hd in zip(res.hand_landmarks, res.handedness):
                label = hd[0].category_name
                if label in pos:
                    pos[label][i] = (np.mean([p.x for p in lms]) * w, np.mean([p.y for p in lms]) * h)
            i += 1
    cap.release()
    return pos, fps, n


def analyze(pos, n, sigma, min_gap):
    out = {}
    for hand, xy in pos.items():
        speed = np.linalg.norm(np.diff(xy, axis=0), axis=1)  # NaN if either frame missing
        valid = ~np.isnan(speed)
        if not valid.any():
            continue
        x = np.arange(len(speed))
        filled = speed.copy()
        filled[~valid] = np.interp(x[~valid], x[valid], speed[valid])
        smooth = gaussian_filter(filled, sigma=sigma)
        valleys, _ = find_peaks(-smooth)
        selected = []
        for v in valleys:
            if not selected or v - selected[-1] >= min_gap:
                selected.append(int(v))
        out[hand] = {"raw": speed, "smooth": smooth, "valleys": selected}
    keyframes = sorted({0, n - 1, *[v for d in out.values() for v in d["valleys"]]})
    return out, keyframes


def plot(curves, keyframes, fps, n, title, png_path):
    hands = [h for h in HANDS if h in curves]
    fig, axes = plt.subplots(len(hands), 1, figsize=(11, 2.6 * len(hands) + 0.9),
                             sharex=True, squeeze=False, facecolor=SURFACE)
    axes = axes[:, 0]
    ymax = max(np.nanmax(curves[h]["raw"]) for h in hands) * 1.08
    for ax, hand in zip(axes, hands):
        c, d = COLORS[hand], curves[hand]
        x = np.arange(len(d["raw"]))
        ax.set_facecolor(SURFACE)
        miss = np.isnan(d["raw"]).astype(int)
        edges = np.flatnonzero(np.diff(np.r_[0, miss, 0]))
        for s, e in zip(edges[::2], edges[1::2]):  # spans with no detection
            ax.axvspan(s - 0.5, e - 0.5, color=GRID, alpha=0.6, lw=0, zorder=0)
        for k in keyframes:
            ax.axvline(k, color=INK_2, lw=1, ls=(0, (3, 3)), alpha=0.55, zorder=1)
        ax.plot(x, d["raw"], color=c, lw=1, alpha=0.35, zorder=2, label="instantaneous speed")
        ax.plot(x, d["smooth"], color=c, lw=2, zorder=3, label=f"smoothed (Gaussian)")
        v = np.array(d["valleys"], dtype=int)
        if len(v):
            ax.scatter(v, d["smooth"][v], s=64, color=c, edgecolor=SURFACE, linewidth=2, zorder=4,
                       label="selected valley")
        ax.set_ylim(0, ymax)
        ax.set_ylabel("speed (px / frame)", color=INK_2, fontsize=9)
        ax.text(0.005, 0.95, f"{hand} hand", transform=ax.transAxes, va="top", fontsize=11,
                fontweight="bold", color=INK)
        missing = int(np.isnan(d["raw"]).sum())
        ax.text(0.995, 0.95, f"{missing} of {len(d['raw'])} steps without detection",
                transform=ax.transAxes, va="top", ha="right", fontsize=8, color=INK_2, zorder=5,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2))
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8)
    axes[-1].set_xlim(0, n - 1)
    axes[-1].set_xlabel("frame id  (keyframe ids listed below the axis)", color=INK_2, fontsize=9,
                        labelpad=20)
    top = axes[0].secondary_xaxis("top", functions=(lambda f: f / fps, lambda t: t * fps))
    top.set_xlabel("time (s)", color=INK_2, fontsize=9)
    top.tick_params(colors=INK_2, labelsize=8)
    top.spines["top"].set_visible(False)
    for k in keyframes:  # label keyframe ids along the bottom panel
        axes[-1].annotate(str(k), (k, 0), xytext=(0, -26), textcoords="offset points", ha="center",
                          fontsize=7, color=INK_2, annotation_clip=False)
    # neutral swatches: hue identifies the hand (named in each panel), the legend explains mark style
    handles = [
        plt.Line2D([], [], color=INK_2, lw=1, alpha=0.4),
        plt.Line2D([], [], color=INK_2, lw=2),
        plt.Line2D([], [], color=INK_2, marker="o", ls="", ms=7, mec=SURFACE, mew=1.5),
        plt.Line2D([], [], color=INK_2, lw=1, ls=(0, (3, 3))),
        plt.Rectangle((0, 0), 1, 1, color=GRID, alpha=0.6),
    ]
    labels = ["instantaneous speed", f"smoothed (Gaussian)", "selected valley",
              "keyframe", "no detection (interpolated)"]
    fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.91))
    fig.legend(handles, labels, loc="upper left", ncol=5, frameon=False, fontsize=8,
               labelcolor=INK_2, bbox_to_anchor=(0.005, 0.955))
    fig.savefig(png_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("video")
    p.add_argument("--out_dir", default=".", help="where to write the PNG/CSV (default: cwd)")
    p.add_argument("--sigma", type=float, default=5, help="Gaussian sigma, as in get_frames.py")
    p.add_argument("--min_gap", type=int, default=15, help="min frames between valleys of one hand")
    p.add_argument("--model", default=os.path.join(HERE, "hand_landmarker.task"))
    a = p.parse_args()

    stem = os.path.splitext(os.path.basename(a.video))[0]
    os.makedirs(a.out_dir, exist_ok=True)
    pos, fps, n = track_hands(a.video, a.model)
    curves, keyframes = analyze(pos, n, a.sigma, a.min_gap)
    if not curves:
        raise SystemExit("No hands detected in the video.")

    png = os.path.join(a.out_dir, f"{stem}_hand_velocity.png")
    plot(curves, keyframes, fps, n, f"Hand speed — {os.path.basename(a.video)}  ({n} frames @ {fps:g} fps)", png)

    csv_path = os.path.join(a.out_dir, f"{stem}_hand_velocity.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "time_s"] + [f"{h}_{k}" for h in curves for k in ("speed_raw", "speed_smooth")]
                    + ["is_keyframe"])
        for i in range(n - 1):
            row = [i, round(i / fps, 4)]
            for h in curves:
                r = curves[h]["raw"][i]
                row += ["" if np.isnan(r) else round(float(r), 4), round(float(curves[h]["smooth"][i]), 4)]
            wr.writerow(row + [int(i in keyframes)])

    print(f"keyframes: {json.dumps(keyframes)}")
    for h, d in curves.items():
        print(f"{h} valleys: {d['valleys']}")
    print(f"saved: {png}\nsaved: {csv_path}")


if __name__ == "__main__":
    main()
