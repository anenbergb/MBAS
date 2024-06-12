import io

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import matplotlib
from PIL import Image
import torch

from monai.transforms.utils import rescale_array
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.utils_pytorch_numpy_unification import repeat
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type

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


def blend_images(
    image: NdarrayOrTensor,
    label: NdarrayOrTensor,
    alpha: float | NdarrayOrTensor = 0.5,
    cmap: str = "hsv",
    rescale_arrays: bool = True,
    transparent_background: bool = True,
) -> NdarrayOrTensor:
    """
    Blend an image and a label. Both should have the shape CHW[D].
    The image may have C==1 or 3 channels (greyscale or RGB).
    The label is expected to have C==1.

    Args:
        image: the input image to blend with label data.
        label: the input label to blend with image data.
        alpha: this specifies the weighting given to the label, where 0 is completely
            transparent and 1 is completely opaque. This can be given as either a
            single value or an array/tensor that is the same size as the input image.
        cmap: specify colormap in the matplotlib, default to `hsv`, for more details, please refer to:
            https://matplotlib.org/2.0.2/users/colormaps.html.
        rescale_arrays: whether to rescale the array to [0, 1] first, default to `True`.
        transparent_background: if true, any zeros in the label field will not be colored.

    .. image:: ../../docs/images/blend_images.png

    """

    if label.shape[0] != 1:
        raise ValueError("Label should have 1 channel.")
    if image.shape[0] not in (1, 3):
        raise ValueError("Image should have 1 or 3 channels.")
    if image.shape[1:] != label.shape[1:]:
        raise ValueError("image and label should have matching spatial sizes.")
    if isinstance(alpha, (np.ndarray, torch.Tensor)):
        if (
            image.shape[1:] != alpha.shape[1:]
        ):  # pytype: disable=attribute-error,invalid-directive
            raise ValueError(
                "if alpha is image, size should match input image and label."
            )

    # rescale arrays to [0, 1] if desired
    if rescale_arrays:
        image = rescale_array(image)

    # label = rescale_array(label)
    # convert image to rgb (if necessary) and then rgba
    if image.shape[0] == 1:
        image = repeat(image, 3, axis=0)

    def get_label_rgb(cmap: str, label: NdarrayOrTensor) -> NdarrayOrTensor:
        _cmap = plt.colormaps.get_cmap(cmap)
        label_np, *_ = convert_data_type(label, np.ndarray)
        label_rgb_np = _cmap(label_np[0])
        label_rgb_np = np.moveaxis(label_rgb_np, -1, 0)[:3]
        label_rgb, *_ = convert_to_dst_type(label_rgb_np, label, np.float32)
        return label_rgb

    label_rgb = get_label_rgb(cmap, label)
    if isinstance(alpha, (torch.Tensor, np.ndarray)):
        w_label = alpha
    elif isinstance(label, torch.Tensor):
        w_label = torch.full_like(label, alpha)
    else:
        w_label = np.full_like(label, alpha, np.float32)
    if transparent_background:
        # where label == 0 (background), set label alpha to 0
        w_label[label == 0] = 0  # pytype: disable=unsupported-operands

    w_image = 1 - w_label
    return w_image * image + w_label * label_rgb
