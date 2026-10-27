"""Train a small heatmap detector from data/manuel_labels.json."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "data" / "extract"
LABELS_FILE = ROOT / "data" / "manuel_labels.json"
INPUT_SIZE = (256, 192)  # height, width; preserve the source 3:4 aspect ratio
OUTPUT_STRIDE = 4
OUTPUT_SIZE = (INPUT_SIZE[0] // OUTPUT_STRIDE, INPUT_SIZE[1] // OUTPUT_STRIDE)


def load_split():
    labels = json.loads(LABELS_FILE.read_text())["images"]
    train_images, train_maps, train_centers = [], [], []
    val_images, val_maps, val_centers = [], [], []
    for relative_path, annotation in sorted(labels.items()):
        path = IMAGE_DIR / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Image from labels is missing: {path}")
        with Image.open(path) as frame:
            if frame.size != (annotation["width"], annotation["height"]):
                raise ValueError(f"Annotation dimensions do not match {path}")
            image = np.asarray(
                frame.convert("RGB").resize((INPUT_SIZE[1], INPUT_SIZE[0])),
                dtype=np.float32,
            ) / 255.0

        heatmap = np.zeros(OUTPUT_SIZE, dtype=np.float32)
        centers = []
        for ball in annotation["balls"]:
            if not (
                0 <= ball["x"] < annotation["width"]
                and 0 <= ball["y"] < annotation["height"]
            ):
                raise ValueError(f"Ball center is outside the image: {relative_path}: {ball}")
            x = ball["x"] * OUTPUT_SIZE[1] / annotation["width"]
            y = ball["y"] * OUTPUT_SIZE[0] / annotation["height"]
            centers.append((x, y))
            cx, cy = int(round(x)), int(round(y))
            left, right = max(0, cx - 3), min(OUTPUT_SIZE[1], cx + 4)
            top, bottom = max(0, cy - 3), min(OUTPUT_SIZE[0], cy + 4)
            yy, xx = np.mgrid[top:bottom, left:right]
            gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 2.0)
            heatmap[top:bottom, left:right] = np.maximum(
                heatmap[top:bottom, left:right], gaussian
            )

        # Hold out an entire clip so adjacent video frames cannot leak into validation.
        destination = (
            (val_images, val_maps, val_centers)
            if relative_path.startswith("IMG_5158.MOV/")
            else (train_images, train_maps, train_centers)
        )
        destination[0].append(image)
        destination[1].append(heatmap[..., np.newaxis])
        destination[2].append(centers)

    if not train_images or not val_images:
        raise ValueError("Expected training clips and the IMG_5158.MOV validation clip")
    return (
        np.asarray(train_images), np.asarray(train_maps), train_centers,
        np.asarray(val_images), np.asarray(val_maps), val_centers,
    )


def build_model():
    inputs = tf.keras.Input((*INPUT_SIZE, 3))
    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(24, 3, padding="same", activation="relu")(x)
    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)

    def weighted_bce(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(bce * (1.0 + 50.0 * y_true))

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=weighted_bce)
    return model


def find_peaks(heatmap, threshold=0.2, nms_radius=3):
    pooled = tf.nn.max_pool2d(heatmap[None, ..., None], 3, 1, "SAME").numpy()[0, ..., 0]
    ys, xs = np.where((heatmap == pooled) & (heatmap >= threshold))
    candidates = sorted(
        ((float(heatmap[y, x]), x, y) for y, x in zip(ys, xs)), reverse=True
    )
    selected = []
    for score, x, y in candidates:
        if all((x - px) ** 2 + (y - py) ** 2 >= nms_radius**2 for _, px, py in selected):
            selected.append((score, x, y))
    return selected


def validation_scores(model, images, centers, threshold=0.2):
    heatmaps = model.predict(images, verbose=0)[..., 0]
    matched = false_positives = targets = 0
    for heatmap, truth in zip(heatmaps, centers):
        used = set()
        for _, x, y in find_peaks(heatmap, threshold):
            nearest = min(
                ((x - gx) ** 2 + (y - gy) ** 2, i)
                for i, (gx, gy) in enumerate(truth) if i not in used
            ) if len(used) < len(truth) else None
            if nearest and nearest[0] <= 9.0:
                matched += 1
                used.add(nearest[1])
            else:
                false_positives += 1
        targets += len(truth)
    precision = matched / max(matched + false_positives, 1)
    recall = matched / max(targets, 1)
    return precision, recall, 2 * precision * recall / max(precision + recall, 1e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "ball_detector.keras")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(42)
    train_x, train_y, _, val_x, val_y, val_centers = load_split()
    print(f"Train: {len(train_x)} frames | validation: {len(val_x)} frames (held-out clip)")

    train = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(
        len(train_x), seed=42
    )

    def flip_pair(image, heatmap):
        seed = tf.random.uniform((2,), maxval=2**31 - 1, dtype=tf.int32)
        return (
            tf.image.stateless_random_flip_left_right(image, seed),
            tf.image.stateless_random_flip_left_right(heatmap, seed),
        )

    train = train.map(flip_pair, num_parallel_calls=1).batch(args.batch_size).prefetch(1)
    model = build_model()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        args.output, monitor="val_loss", save_best_only=True, verbose=1
    )
    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    model.fit(
        train,
        validation_data=(val_x, val_y),
        epochs=args.epochs,
        callbacks=[checkpoint, stop],
        shuffle=False,
        verbose=2,
    )
    model.save(args.output)
    precision, recall, f1 = validation_scores(model, val_x, val_centers)
    print(f"Validation detections (12px tolerance): precision={precision:.3f}, "
          f"recall={recall:.3f}, F1={f1:.3f}")
    print(f"Saved model: {args.output}")


if __name__ == "__main__":
    main()
