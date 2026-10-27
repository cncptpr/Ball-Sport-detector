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
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "validation_detections.jpg")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.limit < 1:
        parser.error("--threshold must be in [0, 1] and --limit must be positive")

    _, _, _, images, _, centers = load_split()
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    heatmaps = model.predict(images, verbose=0)[..., 0]
    predictions = [find_peaks(hm, args.threshold) for hm in heatmaps]

    labels = json.loads(LABELS_FILE.read_text())["images"]
    paths = sorted(path for path in labels if path.startswith("IMG_5158.MOV/"))
    precision, recall, f1, false_positives, true_positives = score(centers, predictions)
    print(
        f"Validation @ {args.threshold:.2f}: TP={true_positives}, FP={false_positives}, "
        f"precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}"
    )
    for threshold in (0.2, 0.4, 0.6, 0.8, 0.9):
        result = score(centers, [find_peaks(hm, threshold) for hm in heatmaps])
        print(f"  threshold {threshold:.1f}: precision={result[0]:.3f}, recall={result[1]:.3f}, F1={result[2]:.3f}")

    cols, tile_w, tile_h, header_h = 4, 384, 512, 26
    count = min(args.limit, len(paths))
    indices = np.linspace(0, len(paths) - 1, count, dtype=int)
    sheet = Image.new("RGB", (cols * tile_w, ((count + cols - 1) // cols) * (tile_h + header_h)), "#202020")
    draw = ImageDraw.Draw(sheet)
    for panel, index in enumerate(indices):
        row, col = divmod(panel, cols)
        x0, y0 = col * tile_w, row * (tile_h + header_h)
        path = IMAGE_DIR / paths[index]
        with Image.open(path) as frame:
            tile = frame.convert("RGB").resize((tile_w, tile_h))
        sheet.paste(tile, (x0, y0 + header_h))
        truth = centers[index]
        found = predictions[index]
        draw.text((x0 + 5, y0 + 5), f"{path.name}  GT:{len(truth)} pred:{len(found)}", fill="white")
        for x, y in truth:
            px = x0 + round(x * tile_w / OUTPUT_SIZE[1])
            py = y0 + header_h + round(y * tile_h / OUTPUT_SIZE[0])
            draw.ellipse((px - 8, py - 8, px + 8, py + 8), outline="#00ff55", width=3)
            draw.line((px - 10, py, px + 10, py), fill="#00ff55", width=2)
            draw.line((px, py - 10, px, py + 10), fill="#00ff55", width=2)
        for confidence, x, y in found:
            px = x0 + round(x * tile_w / OUTPUT_SIZE[1])
            py = y0 + header_h + round(y * tile_h / OUTPUT_SIZE[0])
            draw.ellipse((px - 7, py - 7, px + 7, py + 7), outline="#ff3030", width=3)
            draw.text((px + 8, py - 8), f"{confidence:.1f}", fill="#ff6060")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=90)
    print(f"Saved contact sheet: {args.output}")


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
