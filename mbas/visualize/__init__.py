import matplotlib
from mbas.data.constants import MBAS_LABEL_COLORS

mbas_colormap = matplotlib.colors.ListedColormap(
    MBAS_LABEL_COLORS.values(), name="mbas"
)
matplotlib.colormaps.register(mbas_colormap)
