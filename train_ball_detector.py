"""Train a ball center-and-radius detector from the manually labeled frames."""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "data" / "extract"
LABELS_FILE = next(
    path for path in (ROOT / "data" / "manual_labels.json", ROOT / "data" / "manuel_labels.json")
    if path.is_file()
)
INPUT_SIZE = (512, 384)  # height, width; retain small balls after resizing
OUTPUT_STRIDE = 4
OUTPUT_SIZE = (INPUT_SIZE[0] // OUTPUT_STRIDE, INPUT_SIZE[1] // OUTPUT_STRIDE)


def load_split(validation_only=False):
    labels = json.loads(LABELS_FILE.read_text())["images"]
    train_images, train_maps, train_radius_maps, train_centers, train_radii = [], [], [], [], []
    val_images, val_maps, val_radius_maps, val_centers, val_radii = [], [], [], [], []
    for relative_path, annotation in sorted(labels.items()):
        is_validation = relative_path.startswith("IMG_5158.MOV/")
        if validation_only and not is_validation:
            continue
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
        radius_map = np.zeros(OUTPUT_SIZE, dtype=np.float32)
        centers = []
        radii = []
        for ball in annotation["balls"]:
            if not (
                0 <= ball["x"] < annotation["width"]
                and 0 <= ball["y"] < annotation["height"]
            ):
                raise ValueError(f"Ball center is outside the image: {relative_path}: {ball}")
            x = ball["x"] * OUTPUT_SIZE[1] / annotation["width"]
            y = ball["y"] * OUTPUT_SIZE[0] / annotation["height"]
            radius = ball.get("radius")
            if radius is None or radius <= 0:
                raise ValueError(f"Ball radius is missing or invalid: {relative_path}: {ball}")
            output_radius = radius * OUTPUT_SIZE[1] / annotation["width"]
            centers.append((x, y))
            radii.append(output_radius)
            cx = min(OUTPUT_SIZE[1] - 1, max(0, int(round(x))))
            cy = min(OUTPUT_SIZE[0] - 1, max(0, int(round(y))))
            left, right = max(0, cx - 3), min(OUTPUT_SIZE[1], cx + 4)
            top, bottom = max(0, cy - 3), min(OUTPUT_SIZE[0], cy + 4)
            yy, xx = np.mgrid[top:bottom, left:right]
            gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 2.0)
            heatmap[top:bottom, left:right] = np.maximum(
                heatmap[top:bottom, left:right], gaussian
            )
            radius_map[top:bottom, left:right] = np.maximum(
                radius_map[top:bottom, left:right], gaussian * output_radius
            )

        # Hold out an entire clip so adjacent video frames cannot leak into validation.
        destination = (
            (val_images, val_maps, val_radius_maps, val_centers, val_radii)
            if is_validation
            else (train_images, train_maps, train_radius_maps, train_centers, train_radii)
        )
        destination[0].append(image)
        destination[1].append(heatmap[..., np.newaxis])
        destination[2].append(radius_map[..., np.newaxis])
        destination[3].append(centers)
        destination[4].append(radii)

    if not val_images or (not validation_only and not train_images):
        raise ValueError("Expected training clips and the IMG_5158.MOV validation clip")
    return (
        np.asarray(train_images), (np.asarray(train_maps), np.asarray(train_radius_maps)), train_centers, train_radii,
        np.asarray(val_images), (np.asarray(val_maps), np.asarray(val_radius_maps)), val_centers, val_radii,
    )


def build_model():
    inputs = tf.keras.Input((*INPUT_SIZE, 3))
    normalized = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*INPUT_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    feature_extractor = tf.keras.Model(base.input, base.get_layer("block_6_expand_relu").output)
    x = feature_extractor(normalized, training=False)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(96, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(48, 3, padding="same", activation="relu")(x)
    center_output = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="center")(x)
    radius_output = tf.keras.layers.Conv2D(1, 1, activation="softplus", name="radius")(x)
    model = tf.keras.Model(inputs, (center_output, radius_output))

    def hard_negative_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        positive_loss = tf.reduce_sum(bce * y_true) / tf.maximum(tf.reduce_sum(y_true), 1.0)
        negative_scores = tf.reshape(y_pred * (1.0 - y_true), (tf.shape(y_pred)[0], -1))
        hardest = tf.math.top_k(negative_scores, k=64).values
        return positive_loss + 5.0 * tf.reduce_mean(hardest)

    def radius_loss(y_true, y_pred):
        weights = tf.cast(y_true > 0, tf.float32)
        return tf.reduce_sum(tf.abs(y_true - y_pred) * weights) / tf.maximum(tf.reduce_sum(weights), 1.0)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-4),
        loss={"center": hard_negative_loss, "radius": radius_loss},
        loss_weights={"center": 1.0, "radius": 0.25},
    )
    return model


def find_peaks(heatmap, threshold=0.4, nms_radius=6):
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


def validation_scores(model, images, centers, radii, threshold=0.4):
    heatmaps, radius_maps = model.predict(images, verbose=0)
    heatmaps, radius_maps = heatmaps[..., 0], radius_maps[..., 0]
    matched = false_positives = targets = 0
    radius_error = []
    for heatmap, radius_map, truth, truth_radii in zip(heatmaps, radius_maps, centers, radii):
        used = set()
        for _, x, y in find_peaks(heatmap, threshold):
            nearest = min(
                ((x - gx) ** 2 + (y - gy) ** 2, i)
                for i, (gx, gy) in enumerate(truth) if i not in used
            ) if len(used) < len(truth) else None
            if nearest and nearest[0] <= 9.0:
                matched += 1
                used.add(nearest[1])
                radius_error.append(abs(float(radius_map[y, x]) - truth_radii[nearest[1]]))
            else:
                false_positives += 1
        targets += len(truth)
    precision = matched / max(matched + false_positives, 1)
    recall = matched / max(targets, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1, false_positives, matched, radius_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "ball_detector.keras")
    parser.add_argument("--minutes", type=float, default=0, help="stop after this many minutes")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.minutes < 0:
        parser.error("--epochs and --batch-size must be positive")

    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(42)
    train_x, train_y, _, _, val_x, val_y, val_centers, val_radii = load_split()
    print(f"Train: {len(train_x)} frames | validation: {len(val_x)} frames (held-out clip)")

    train = tf.data.Dataset.from_tensor_slices((train_x, {"center": train_y[0], "radius": train_y[1]})).shuffle(
        len(train_x), seed=42
    )

    def flip_pair(image, targets):
        seed = tf.random.uniform((2,), maxval=2**31 - 1, dtype=tf.int32)
        return (
            tf.image.stateless_random_flip_left_right(image, seed),
            {
                name: tf.image.stateless_random_flip_left_right(target, seed)
                for name, target in targets.items()
            },
        )

    train = train.map(flip_pair, num_parallel_calls=1).batch(args.batch_size).prefetch(1)
    model = build_model()
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        args.output, monitor="val_loss", save_best_only=True, verbose=1
    )
    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=1e-5, patience=3, restore_best_weights=True
    )
    callbacks = [checkpoint]
    if args.minutes:
        started = time.monotonic()

        class TimeLimit(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                elapsed = (time.monotonic() - started) / 60
                print(f"Elapsed training time: {elapsed:.2f} minutes")
                if elapsed >= args.minutes:
                    self.model.stop_training = True

        callbacks.append(TimeLimit())
    else:
        callbacks.append(stop)
    model.fit(
        train,
        validation_data=(val_x, val_y),
        epochs=args.epochs,
        callbacks=callbacks,
        shuffle=False,
        verbose=2,
    )
    model.save(args.output)
    precision, recall, f1, false_positives, true_positives, radius_error = validation_scores(
        model, val_x, val_centers, val_radii
    )
    print(
        f"Validation @ 0.40 (45px tolerance): TP={true_positives}, FP={false_positives}, "
        f"precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}, "
        f"radius MAE={np.mean(radius_error) if radius_error else float('nan'):.2f}px"
    )
    print(f"Saved model: {args.output}")


if __name__ == "__main__":
    main()
