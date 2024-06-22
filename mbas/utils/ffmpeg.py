import io

import ffmpeg
import matplotlib
import numpy as np


def figure_to_array(fig):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="raw")
    io_buf.seek(0)
    arr = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )
    arr = arr[..., :3]  # HWC, C = RGB
    io_buf.close()
    return arr


class FfmpegWriter:
    def __init__(
        self,
        save_path,
        width,
        height,
        framerate=30,
        vcodec="libx264",
        crf=20,
    ):
        self.width = width
        self.height = height
        self.process = (
            ffmpeg.input(
                "pipe:",
                r=framerate,
                format="rawvideo",
                pix_fmt="rgb24",
                s=f"{width}x{height}",
            )
            .output(save_path, vcodec=vcodec, pix_fmt="yuv420p", crf=crf)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def write(self, image):
        if isinstance(image, matplotlib.figure.Figure):
            image = figure_to_array(image)

        assert isinstance(image, np.ndarray)
        buffer = image.astype(np.uint8).tobytes()
        self.process.stdin.write(buffer)

    def close(self):
        self.process.stdin.close()
        self.process.wait()
