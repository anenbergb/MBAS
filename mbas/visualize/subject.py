import io

import matplotlib.pyplot as plt
import numpy as np
import torchio as tio
from matplotlib import font_manager

# from monai.transforms.utils import rescale_array
from PIL import Image, ImageDraw, ImageFont
from torchio.transforms.preprocessing.spatial.to_canonical import ToCanonical
from torchio.visualization import color_labels, rotate

from .visualize import (
    add_color_border,
    figure_to_array,
    make_segmentation_legend,
    blend_images,
)


def plot_volume(
    image: tio.Image,
    radiological=True,
    channel=-1,  # default to foreground for binary maps
    axes=None,
    cmap=None,
    output_path=None,
    show=True,
    xlabels=True,
    percentiles=(0.5, 99.5),
    figsize=None,
    reorient=True,
    indices=None,
    single_axis=None,  # one of (Sagittal, Coronal, Axial)
):
    fig = None

    axes_names = "sagittal", "coronal", "axial"
    if single_axis is not None:
        assert single_axis in ("sagittal", "coronal", "axial")
        axes_names = (single_axis,)
    num_rows = len(axes_names)

    if axes is None:
        fig, axes = plt.subplots(1, num_rows, figsize=figsize, squeeze=False)
        axes = axes.flatten()

    # sag_axis, cor_axis, axi_axis = axes

    if reorient:
        image = ToCanonical()(image)  # type: ignore[assignment]
    data = image.data[channel]
    if indices is None:
        indices = np.array(data.shape) // 2
    i, j, k = indices
    slice_x = rotate(data[i, :, :], radiological=radiological)
    slice_y = rotate(data[:, j, :], radiological=radiological)
    slice_z = rotate(data[:, :, k], radiological=radiological)
    kwargs = {}
    is_label = isinstance(image, tio.LabelMap)
    if isinstance(cmap, dict):
        slices = slice_x, slice_y, slice_z
        slice_x, slice_y, slice_z = color_labels(slices, cmap)
    else:
        if cmap is None:
            cmap = "cubehelix" if is_label else "gray"
        kwargs["cmap"] = cmap
    if is_label:
        kwargs["interpolation"] = "none"

    sr, sa, ss = image.spacing
    kwargs["origin"] = "lower"

    if percentiles is not None and not is_label:
        p1, p2 = np.percentile(data, percentiles)
        kwargs["vmin"] = p1
        kwargs["vmax"] = p2

    axes_map = {
        "sagittal": {
            "aspect": ss / sa,
            "slice": slice_x,
            "xlabel": "A",
            "ylabel": "S",
            "title": "Sagittal",
        },
        "coronal": {
            "aspect": ss / sr,
            "slice": slice_y,
            "xlabel": "R",
            "ylabel": "S",
            "title": "Coronal",
        },
        "axial": {
            "aspect": sa / sr,
            "slice": slice_z,
            "xlabel": "R",
            "ylabel": "A",
            "title": "Axial",
        },
    }

    for i, axes_name in enumerate(axes_names):
        ax_map = axes_map[axes_name]
        axes[i].imshow(ax_map["slice"], aspect=ax_map["aspect"], **kwargs)
        if xlabels:
            axes[i].set_xlabel(ax_map["xlabel"])
        axes[i].set_ylabel(ax_map["ylabel"])
        axes[i].invert_xaxis()
        axes[i].set_title(ax_map["title"])

    if fig is not None:
        fig.tight_layout(pad=0)
    if output_path is not None and fig is not None:
        fig.savefig(output_path)
    if show:
        plt.show()
    return fig


def plot_volume_with_label(
    image: tio.Image,
    label: tio.LabelMap,
    radiological=True,
    channel=-1,  # default to foreground for binary maps
    axes=None,
    output_path=None,
    show=True,
    xlabels=True,
    figsize=None,
    reorient=True,
    indices=None,
    single_axis=None,  # one of (Sagittal, Coronal, Axial)
    border_color=None,
    border_width=10,
    blend_alpha=0.25,
    cmap="mbas",
):
    fig = None

    axes_names = "sagittal", "coronal", "axial"
    if single_axis is not None:
        assert single_axis in ("sagittal", "coronal", "axial")
        axes_names = (single_axis,)
    num_rows = len(axes_names)

    if axes is None:
        fig, axes = plt.subplots(1, num_rows, figsize=figsize, squeeze=False)
        axes = axes.flatten()

    def prepare_slices(im):
        if reorient:
            im = ToCanonical()(im)  # type: ignore[assignment]
        data = im.data[channel]
        indices_ = indices
        if indices_ is None:
            indices_ = np.array(data.shape) // 2
        i, j, k = indices_
        slice_x = rotate(data[i, :, :], radiological=radiological)
        slice_y = rotate(data[:, j, :], radiological=radiological)
        slice_z = rotate(data[:, :, k], radiological=radiological)
        return slice_x, slice_y, slice_z

    def blend_slice(mri_slice, label_slice):
        blend = blend_images(
            image=np.expand_dims(mri_slice, 0),
            label=np.expand_dims(label_slice, 0),
            alpha=blend_alpha,
            cmap=cmap,
            rescale_arrays=True,
        )
        blend = 255.0 * blend
        blend = blend.astype(np.uint8)
        blend = np.moveaxis(blend, 0, -1)  # CHW -> HWC
        blend = np.ascontiguousarray(blend)
        if border_color is not None:
            blend = add_color_border(blend, border_width, border_color)
        return blend

    image_slices = prepare_slices(image)
    label_slices = prepare_slices(label)
    slice_x, slice_y, slice_z = (
        blend_slice(i, l) for i, l in zip(image_slices, label_slices)
    )

    kwargs = {}
    sr, sa, ss = image.spacing
    kwargs["origin"] = "lower"

    axes_map = {
        "sagittal": {
            "aspect": ss / sa,
            "slice": slice_x,
            "xlabel": "A",
            "ylabel": "S",
            "title": "Sagittal",
        },
        "coronal": {
            "aspect": ss / sr,
            "slice": slice_y,
            "xlabel": "R",
            "ylabel": "S",
            "title": "Coronal",
        },
        "axial": {
            "aspect": sa / sr,
            "slice": slice_z,
            "xlabel": "R",
            "ylabel": "A",
            "title": "Axial",
        },
    }

    for i, axes_name in enumerate(axes_names):
        ax_map = axes_map[axes_name]
        axes[i].imshow(ax_map["slice"], aspect=ax_map["aspect"], **kwargs)
        if xlabels:
            axes[i].set_xlabel(ax_map["xlabel"])
        axes[i].set_ylabel(ax_map["ylabel"])
        axes[i].invert_xaxis()
        axes[i].set_title(ax_map["title"])

    if fig is not None:
        fig.tight_layout(pad=0)
    if output_path is not None and fig is not None:
        fig.savefig(output_path)
    if show:
        plt.show()
    return fig


def plot_subject(
    subject: tio.Subject,
    cmap_dict=None,
    show=False,
    figsize=None,
    clear_axes=True,
    return_fig=False,
    single_axis=None,  # one of (sagittal, coronal, axial)
    add_metadata=False,
    subject_include=None,
    **kwargs,
):
    num_images = len(subject) if subject_include is None else len(subject_include)
    many_images = num_images > 2
    subplots_kwargs = {"figsize": figsize}
    try:
        if clear_axes:
            subject.check_consistent_spatial_shape()
            subplots_kwargs["sharex"] = "row" if many_images else "col"
            subplots_kwargs["sharey"] = "row" if many_images else "col"
    except RuntimeError:  # different shapes in subject
        pass

    axes_names = "sagittal", "coronal", "axial"
    if single_axis is not None:
        assert single_axis in ("sagittal", "coronal", "axial")
        axes_names = (single_axis,)
    num_rows = len(axes_names)

    args = (num_rows, num_images) if many_images else (num_images, num_rows)
    fig, axes = plt.subplots(*args, **subplots_kwargs, squeeze=False)
    # The array of axes must be 2D so that it can be indexed correctly within
    # the plot_volume() function
    axes = axes.T if many_images else axes.reshape(-1, num_rows)
    iterable = enumerate(
        subject.get_images_dict(include=subject_include, intensity_only=False).items()
    )
    for image_index, (name, image) in iterable:
        image_axes = axes[image_index]
        cmap = None
        if cmap_dict is not None and name in cmap_dict:
            cmap = cmap_dict[name]
        last_row = image_index == len(axes) - 1
        plot_volume(
            image,
            axes=image_axes,
            show=False,
            cmap=cmap,
            xlabels=last_row,
            single_axis=single_axis,
            **kwargs,
        )
        for axis, axis_name in zip(image_axes, axes_names):
            axis.set_title(f"{name} ({axis_name})")

    if add_metadata:
        legend = make_subject_metadata_legend(
            fig, subject, loc="upper left", bbox_to_anchor=(0.0, 0.0)
        )
        fig.add_artist(legend)

    fig.tight_layout(pad=0)
    fig.canvas.draw()

    close_fig = True
    if show:
        plt.show()
        close_fig = False
    if return_fig:
        return fig
    data = figure_to_array(fig)
    if close_fig:
        plt.close(fig)
    return data


def plot_subject_with_label(
    subject: tio.Subject,
    show=False,
    figsize=None,
    clear_axes=True,
    return_fig=False,
    single_axis=None,  # one of (sagittal, coronal, axial)
    add_metadata=False,
    subject_include=None,
    label_key="label",
    cmap="mbas",
    add_segmentation_legend=False,
    **kwargs,
):
    assert label_key in subject

    num_images = len(subject.get_images(intensity_only=True))
    if subject_include is not None:
        num_images = len(subject_include)

    many_images = num_images > 2
    subplots_kwargs = {"figsize": figsize}
    try:
        if clear_axes:
            subject.check_consistent_spatial_shape()
            subplots_kwargs["sharex"] = "row" if many_images else "col"
            subplots_kwargs["sharey"] = "row" if many_images else "col"
    except RuntimeError:  # different shapes in subject
        pass

    axes_names = "sagittal", "coronal", "axial"
    if single_axis is not None:
        assert single_axis in ("sagittal", "coronal", "axial")
        axes_names = (single_axis,)
    num_rows = len(axes_names)

    args = (num_rows, num_images) if many_images else (num_images, num_rows)
    fig, axes = plt.subplots(*args, **subplots_kwargs, squeeze=False)
    # The array of axes must be 2D so that it can be indexed correctly within
    # the plot_volume() function
    axes = axes.T if many_images else axes.reshape(-1, num_rows)
    iterable = enumerate(
        subject.get_images_dict(include=subject_include, intensity_only=True).items()
    )
    label_image = subject[label_key]
    for image_index, (name, image) in iterable:
        image_axes = axes[image_index]
        last_row = image_index == len(axes) - 1
        plot_volume_with_label(
            image,
            label_image,
            axes=image_axes,
            show=False,
            cmap=cmap,
            xlabels=last_row,
            single_axis=single_axis,
            **kwargs,
        )
        for axis, axis_name in zip(image_axes, axes_names):
            axis.set_title(f"{name} ({axis_name})")

    if add_metadata:
        legend = make_subject_metadata_legend(
            fig, subject, loc="upper left", bbox_to_anchor=(0.0, 0.0)
        )
        fig.add_artist(legend)

    if add_segmentation_legend:
        loc, bbox_to_anchor = (
            ("upper right", (1.0, 0.0)) if add_metadata else ("upper left", (0.0, 0.0))
        )
        legend = make_segmentation_legend(fig, loc=loc, bbox_to_anchor=bbox_to_anchor)
        fig.add_artist(legend)

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    close_fig = True
    if show:
        plt.show()
        close_fig = False
    if return_fig:
        return fig
    data = figure_to_array(fig)
    if close_fig:
        plt.close(fig)
    return data


def render_subject_metadata(
    subject: tio.Subject,
    font="DejaVu Sans",
    font_size=16,
) -> np.ndarray:
    from mbas.data.subject_utils import get_subject_nonimages

    font_type = font_manager.FontProperties(family=font, weight="normal")
    font_file = font_manager.findfont(font_type)
    image_font = ImageFont.truetype(font_file, font_size)
    height = image_font.getmetrics()[0] + 1

    max_width = 0

    rows = []
    for name, value in get_subject_nonimages(subject).items():
        row_str = f"{name}: {value}"
        (width, _), _ = image_font.font.getsize(row_str)
        max_width = max(max_width, width)
        rows.append(row_str)

    info_image = Image.new("RGB", (max_width, height * len(rows)), "white")
    draw = ImageDraw.Draw(info_image)
    for i, row in enumerate(rows):
        draw.text((0, i * height), row, font=image_font, fill=(0, 0, 0))
    arr = np.asarray(info_image)
    return arr


def make_subject_metadata_legend(
    fig, subject: tio.Subject, fontsize=12, loc="upper left", bbox_to_anchor=(0.0, 0.0)
):
    import matplotlib.patches as mpatches

    from mbas.data.subject_utils import get_subject_nonimages

    patches = []
    for name, value in get_subject_nonimages(subject).items():
        label = f"{name}: {value}"
        patch = mpatches.Patch(color="black", label=label)
        patches.append(patch)

    legend = fig.legend(
        handles=patches, loc=loc, fontsize=fontsize, bbox_to_anchor=bbox_to_anchor
    )
    return legend
