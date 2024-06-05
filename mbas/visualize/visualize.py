import io

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import matplotlib
from monai.transforms.utils import rescale_array
from PIL import Image

from mbas.data.constants import MBAS_LABELS, MBAS_LABEL_COLORS

# mbas_colormap = matplotlib.colors.ListedColormap(MBAS_LABEL_COLORS.values(), name = "mbas")
# matplotlib.colormaps.register(mbas_colormap)


def plot_classification_grid(
    preds: np.ndarray, target: np.ndarray, patient_id: np.ndarray
) -> np.ndarray:
    assert len(preds) == len(target)
    assert len(preds) == len(patient_id)

    errors = np.abs(preds - target)
    errors_bucketed = np.digitize(errors, np.arange(0, 1.1, 0.1)) - 1  # 10 buckets
    zipped = zip(patient_id, errors_bucketed, target)
    zipped = sorted(zipped, key=lambda x: x[0])

    width = int(np.ceil(np.sqrt(len(preds))))
    grid = np.zeros((width, width), dtype=int)
    cmap = colors.LinearSegmentedColormap.from_list(
        "Custom", ["green", "white", "red"], N=10
    )
    fig, ax = plt.subplots()
    for i, (pid, error_bucket, tar) in enumerate(zipped):
        grid[i // width, i % width] = error_bucket
        methyl = "(M)" if tar == 1 else ""
        text = f"{pid}\n{methyl}"
        ax.text(i % width, i // width, text, ha="center", va="center", color="black")
    ax.imshow(grid, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def add_color_border(img, border_width=10, color="green"):
    """
    Assume img is shape (H,W,C)
    """
    assert isinstance(img, np.ndarray)
    height = img.shape[0]
    width = img.shape[1]
    frame_height = 2 * border_width + height
    frame_width = 2 * border_width + width
    framed_img = Image.new("RGB", (frame_width, frame_height), color)
    framed_img = np.array(framed_img)
    framed_img[border_width:-border_width, border_width:-border_width] = img
    return framed_img


def segmentation_label_color(labels=[0, 1, 2, 3], cmap="mbas", rgb_255=True):
    """
    Generate the label color corresponding to the cmap

    Visualize the color with Image.new("RGB", (50, 50), rgb_color)
    """
    labels_np = np.array(labels)
    label_np = rescale_array(labels_np)
    label_rgb = matplotlib.colormaps[cmap](label_np)
    if rgb_255:
        colors = [tuple((255.0 * l).astype(np.uint8)[:3]) for l in label_rgb]
    else:
        colors = [tuple(l[:3]) for l in label_rgb]
    return colors


def make_segmentation_legend(
    fig, fontsize=12, loc="upper left", bbox_to_anchor=(0.0, 0.0)
):
    patches = [
        mpatches.Patch(color=MBAS_LABEL_COLORS[i], label=f"({i}) {MBAS_LABELS[i]}")
        for i in [1, 2, 3]
    ]
    legend = fig.legend(
        handles=patches,
        loc=loc,
        fontsize=fontsize,
        title="MBAS Segmentation Labels",
        title_fontsize=fontsize,
        bbox_to_anchor=bbox_to_anchor,
    )
    return legend


def figure_to_array(fig, rgb_255=True):
    with io.BytesIO() as buff:
        fig.savefig(buff, format="png", bbox_inches="tight")
        buff.seek(0)
        im = plt.imread(buff, format="RGB")
    im = im[..., :3]  # from RGBA -> RGB
    if rgb_255:
        im = (255.0 * im).astype(np.uint8)
    return im
