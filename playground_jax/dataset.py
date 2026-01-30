import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
import numpy as np


def _prep_images(ds, image_size: int):
    imgs = ds["image"].astype(np.float32) / 255.0  # [0,1]
    imgs = imgs * 2.0 - 1.0  # [-1,1]
    # resize to image_size using jax.image.resize on host
    imgs = jax.image.resize(imgs, (imgs.shape[0], image_size, image_size, 1), method="bilinear")
    return np.array(imgs)


def load_mnist(image_size: int, seed: int = 0):
    train_ds = tfds.load("mnist", split="train", as_supervised=False, batch_size=-1)
    test_ds = tfds.load("mnist", split="test", as_supervised=False, batch_size=-1)
    train_imgs = _prep_images(train_ds, image_size)
    test_imgs = _prep_images(test_ds, image_size)
    train_labels = np.array(train_ds["label"], dtype=np.int32)
    test_labels = np.array(test_ds["label"], dtype=np.int32)
    return (train_imgs, train_labels), (test_imgs, test_labels)


def split_per_class(imgs: np.ndarray, labels: np.ndarray, holdout_per_class: int, leave_out_digit: int | None = None, leaveout_eval_holdout: int = 0):
    rng = np.random.default_rng()
    train_indices = {}
    eval_indices = {}
    digits = list(range(10))
    for d in digits:
        idxs = np.where(labels == d)[0]
        rng.shuffle(idxs)
        if leave_out_digit is not None and d == leave_out_digit:
            eval_indices[d] = idxs[:leaveout_eval_holdout]
            train_indices[d] = np.array([], dtype=np.int64)
        else:
            eval_indices[d] = idxs[:holdout_per_class]
            train_indices[d] = idxs[holdout_per_class:]
    return train_indices, eval_indices


def sample_digit_batch(imgs: np.ndarray, indices_by_digit: dict[int, np.ndarray], digit: int, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    idx_pool = indices_by_digit[digit]
    replace = len(idx_pool) < batch_size
    idx = rng.choice(idx_pool, size=batch_size, replace=replace)
    return imgs[idx]
