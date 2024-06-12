import matplotlib
from mbas.data.constants import (
    MBAS_LABEL_COLORS,
    MBAS_HIERARCHICAL_LABEL_COLORS,
    MBAS_ONLY_ATRIUM_LABEL_COLORS,
)

matplotlib.colormaps.register(
    matplotlib.colors.ListedColormap(MBAS_LABEL_COLORS.values(), name="mbas")
)

matplotlib.colormaps.register(
    matplotlib.colors.ListedColormap(
        MBAS_HIERARCHICAL_LABEL_COLORS.values(), name="mbas_hierarchical"
    )
)

matplotlib.colormaps.register(
    matplotlib.colors.ListedColormap(
        MBAS_ONLY_ATRIUM_LABEL_COLORS.values(), name="mbas_only_atrium"
    )
)
