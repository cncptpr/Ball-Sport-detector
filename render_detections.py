"""Render annotated validation frames with ball detections overlaid."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from train_ball_detector import IMAGE_DIR, LABELS_FILE, OUTPUT_SIZE, load_split, find_peaks


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "ball_detector.keras"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--model", type=Path, default=MODEL_FILE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "detections")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")

    train_images, _, train_centers, _, val_images, _, val_centers, _ = load_split()
    centers = train_centers + val_centers
    model = tf.keras.models.load_model(args.model, compile=False)
    train_center_maps, train_radius_maps = model.predict(train_images, batch_size=4, verbose=0)
    val_center_maps, val_radius_maps = model.predict(val_images, batch_size=4, verbose=0)
    heatmaps = np.concatenate((train_center_maps[..., 0], val_center_maps[..., 0]))
    radius_maps = np.concatenate((train_radius_maps[..., 0], val_radius_maps[..., 0]))
    predictions = [find_peaks(hm, args.threshold) for hm in heatmaps]

    labels = json.loads(LABELS_FILE.read_text())["images"]
    train_paths = sorted(path for path in labels if not path.startswith("IMG_5158.MOV/"))
    val_paths = sorted(path for path in labels if path.startswith("IMG_5158.MOV/"))
    paths = train_paths + val_paths
    if len(paths) != len(centers):
        raise ValueError("Rendered frame order does not match the labeled dataset")

    val_predictions = predictions[len(train_paths):]
    precision, recall, f1, false_positives, true_positives = score(val_centers, val_predictions)
    print(
        f"Validation @ {args.threshold:.2f}: TP={true_positives}, FP={false_positives}, "
        f"precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}"
    )
    for threshold in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        result = score(val_centers, [find_peaks(hm, threshold) for hm in heatmaps[len(train_paths):]])
        print(f"  threshold {threshold:.3f}: TP={result[4]}, FP={result[3]}, "
              f"precision={result[0]:.3f}, recall={result[1]:.3f}, F1={result[2]:.3f}")

    count = 0
    for relative_path, truth, found in zip(paths, centers, predictions):
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe image path in labels: {relative_path}")
        image_path = IMAGE_DIR / relative
        with Image.open(image_path) as source:
            frame = source.convert("RGB")
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, frame.width, 38), fill="#202020")
        draw.text(
            (10, 10),
            f"{relative_path} | GT {len(truth)} | detections {len(found)} | threshold {args.threshold:.2f}",
            fill="white",
        )
        for x, y in truth:
            px = round(x * frame.width / OUTPUT_SIZE[1])
            py = round(y * frame.height / OUTPUT_SIZE[0])
            draw.line((px - 18, py, px + 18, py), fill="#00ff55", width=4)
            draw.line((px, py - 18, px, py + 18), fill="#00ff55", width=4)
        for confidence, x, y in found:
            px = round(x * frame.width / OUTPUT_SIZE[1])
            py = round(y * frame.height / OUTPUT_SIZE[0])
            draw.line((px - 12, py, px + 12, py), fill="#ff3030", width=4)
            draw.line((px, py - 12, px, py + 12), fill="#ff3030", width=4)
            radius = radius_maps[count, y, x] * frame.width / OUTPUT_SIZE[1]
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline="#ff6060", width=3)
            draw.text((px + 14, py - 12), f"{confidence:.2f} r={radius:.1f}", fill="#ff6060")
        output_path = args.output_dir / relative.with_suffix(".jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.save(output_path, quality=90)
        count += 1

    print(f"Saved {count} individual frame images under {args.output_dir}")


def score(centers, predictions, tolerance=3):
    matched = false_positives = targets = 0
    for truth, detections in zip(centers, predictions):
        used = set()
        for _, x, y in detections:
            nearest = min(
                (((x - gx) ** 2 + (y - gy) ** 2, i) for i, (gx, gy) in enumerate(truth) if i not in used),
                default=None,
            )
            if nearest is not None and nearest[0] <= tolerance**2:
                matched += 1
                used.add(nearest[1])
            else:
                false_positives += 1
        targets += len(truth)
    precision = matched / max(matched + false_positives, 1)
    recall = matched / max(targets, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1, false_positives, matched


if __name__ == "__main__":
    main()
