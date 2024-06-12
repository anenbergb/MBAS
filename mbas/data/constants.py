from collections import OrderedDict

# Labels according to the website
# MBAS_LABELS = OrderedDict({
#     0: "Background",
#     1: "Right Atrium Cavity",
#     2: "Left Atrium Cavity",
#     3: "Left & Right Atrium Wall",
# })

# MBAS_LABEL_COLORS = OrderedDict({
#     0: "black",
#     1: "red",
#     2: "blue",
#     3: "yellow"
# })

MBAS_LABELS = OrderedDict(
    {
        0: "Background",
        1: "Left & Right Atrium Wall",
        2: "Right Atrium Cavity",
        3: "Left Atrium Cavity",
    }
)

MBAS_LABEL_COLORS = OrderedDict({0: "black", 1: "yellow", 2: "blue", 3: "red"})


MBAS_HIERARCHICAL_LABELS = OrderedDict(
    {
        0: "Background",
        1: "Left Atrium Wall",
        2: "Right Atrium Cavity",
        3: "Left Atrium Cavity",
        4: "Right Atrium Wall",
    }
)

MBAS_HIERARCHICAL_LABEL_COLORS = OrderedDict(
    {0: "black", 1: "yellow", 2: "red", 3: "blue", 4: "green"}
)

MBAS_ONLY_ATRIUM_LABELS = OrderedDict(
    {
        0: "Background",
        1: "Left Atrium",
        2: "Right Atrium",
    }
)

MBAS_ONLY_ATRIUM_LABEL_COLORS = OrderedDict({0: "black", 1: "blue", 2: "red"})
