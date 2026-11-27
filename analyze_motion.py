"""Extract jumping-jack poses and reproduce Assignment 3 figures."""

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

from p1_2_post_processing import create_height_corrected_target, export_unity_csv, load_raw_keypoints

VIDEO_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/5/57/"
    "Jumping_jacks_and_burpees.webm"
)
START = 0.5
STOP = 3.0
FEET = (31, 32)  # MediaPipe foot index joints


def extract(video, start, stop, output):
    """Extract 33 tracked world landmarks using the course y-axis convention."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Could not read video frame rate")
    frames = []
    coordinates = []
    visibility = []
    samples = []
    sample_times = np.arange(start, stop, max(1, (stop - start) / 12))
    wanted = set(np.round(sample_times * fps).astype(int))
    try:
        with mp.solutions.pose.Pose(
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            frame_index = 0
            while frame_index / fps < stop:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index / fps >= start:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if frame_index in wanted:
                        samples.append((frame_index / fps, rgb))
                    estimate = pose.process(rgb).pose_world_landmarks
                    frames.append(frame_index)
                    if estimate:
                        coordinates.append([(p.x, -p.y, p.z) for p in estimate.landmark])
                        visibility.append([p.visibility for p in estimate.landmark])
                    else:
                        coordinates.append(np.full((33, 3), np.nan))
                        visibility.append(np.zeros(33))
                frame_index += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError("No decoded frames in the selected time range")
    if samples:
        fig, axes = plt.subplots(3, 4, figsize=(10, 6))
        for ax in axes.flat:
            ax.axis("off")
        for ax, (time, frame) in zip(axes.flat, samples):
            ax.imshow(frame)
            ax.set_title(f"{time:.2f} s")
        fig.savefig(output / "frames.jpg", dpi=110, bbox_inches="tight")
        plt.close(fig)
    return np.array(frames), np.array(coordinates), np.array(visibility), fps


def chart(time, raw, corrected, output):
    fig, ax = plt.subplots(figsize=(8.6, 3.1), layout="constrained")
    ax.plot(time, raw, color="#a03e20", linewidth=1.5, label="Raw lower foot")
    ax.plot(time, corrected, color="#005d98", linewidth=1.65, label="Corrected lower foot")
    ax.set(xlabel="Source video time (s)", ylabel="Lower foot y (m)")
    ax.grid(alpha=0.23)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", frameon=False)
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)


def skeleton(xyz, output):
    fig = plt.figure(figsize=(4.5, 3.4), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    for i, j in sorted(mp.solutions.pose.POSE_CONNECTIONS):
        ax.plot(xyz[[i, j], 0], xyz[[i, j], 2], xyz[[i, j], 1], color="#146090")
    ax.scatter(xyz[:, 0], xyz[:, 2], xyz[:, 1], s=7, color="#146090")
    for joint in FEET:
        ax.scatter(*xyz[joint, [0, 2, 1]], s=35, color="#a03e20")
    ax.set_axis_off()
    ax.view_init(elev=13, azim=-67)
    ax.set_box_aspect(np.maximum(np.ptp(xyz[:, [0, 2, 1]], axis=0), 0.15))
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="Existing local source video")
    parser.add_argument("--start", type=float, default=START)
    parser.add_argument("--stop", type=float, default=STOP)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if not 0 <= args.start < args.stop:
        parser.error("Expected 0 <= start < stop")
    args.output.mkdir(parents=True, exist_ok=True)
    video = args.video
    if video is None:
        video = args.output / "source.webm"
        if not video.is_file():
            urlretrieve(VIDEO_URL, video)
    frames, raw, visibility, fps = extract(video, args.start, args.stop, args.output)
    report(frames, raw, visibility, fps, args.start, args.stop, args.output)


def report(frames, raw, visibility, fps, start, stop, output):
    detected = np.isfinite(raw).all(axis=(1, 2))
    if not detected.any():
        raise ValueError("No complete poses were detected")
    observed = raw[detected]
    rows = np.column_stack((
        np.repeat(frames[detected], 33),
        np.tile(np.arange(33), len(observed)),
        observed.reshape(-1, 3),
        visibility[detected].ravel(),
    ))
    raw_file = output / "course_raw.npz"
    np.savez_compressed(raw_file, data=rows)
    course_raw = load_raw_keypoints(raw_file)
    np.testing.assert_array_equal(course_raw, observed)
    course_corrected = create_height_corrected_target(course_raw)
    export_unity_csv(course_corrected, output / "course_target.csv")
    corrected = np.full_like(raw, np.nan)
    corrected[detected] = course_corrected
    np.testing.assert_array_equal(course_raw[:, :, [0, 2]], course_corrected[:, :, [0, 2]])
    raw_feet = np.min(raw[:, FEET, 1], axis=1)
    corrected_feet = np.min(corrected[:, FEET, 1], axis=1)
    np.testing.assert_array_equal(corrected_feet[detected], 0.0)
    time = frames / fps
    lowest = int(np.nanargmin(raw_feet))
    chart(time, raw_feet, corrected_feet, output / "comparison.png")
    skeleton(raw[lowest], output / "skeleton.png")
    clamped = course_raw[:, :, 1] < raw_feet[detected, None]
    metrics = {
        "fps": fps, "frame_count": len(frames), "detected_frames": int(detected.sum()),
        "video_start_s": start, "video_stop_s": stop,
        "lowest_frame": int(frames[lowest]), "lowest_time": float(time[lowest]),
        "raw_min_m": float(np.nanmin(raw_feet)), "raw_max_m": float(np.nanmax(raw_feet)),
        "corrected_min_m": float(np.nanmin(corrected_feet)),
        "corrected_max_m": float(np.nanmax(corrected_feet)),
        "raw_rms_m": float(np.sqrt(np.nanmean(raw_feet**2))),
        "corrected_rms_m": float(np.sqrt(np.nanmean(corrected_feet**2))),
        "clamped_joints": int(clamped.sum()),
        "clamped_frames": int(clamped.any(axis=1).sum()),
    }
    np.savez_compressed(
        output / "keypoints.npz", frame=frames, time_s=time,
        raw_xyz_m=raw, corrected_xyz_m=corrected, visibility=visibility,
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
