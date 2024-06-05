import matplotlib
from mbas.data.constants import MBAS_LABEL_COLORS, MBAS_HIERARCHICAL_LABEL_COLORS

mbas_colormap = matplotlib.colors.ListedColormap(
    MBAS_LABEL_COLORS.values(), name="mbas"
)
matplotlib.colormaps.register(mbas_colormap)

mbas_hierarchical_colormap = matplotlib.colors.ListedColormap(
    MBAS_HIERARCHICAL_LABEL_COLORS.values(), name="mbas_hierarchical"
)
matplotlib.colormaps.register(mbas_hierarchical_colormap)
