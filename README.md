# MBAS
MICCAI MBAS 2024 - Multi-class Bi-Atrial Segmentation Challenge

In collaboration with [Peter Chang](https://www.faculty.uci.edu/profile/?facultyId=6569) and the [Center for Artificial Intelligence in Diagnostic Medicine](https://www.caidm.som.uci.edu/).


## Installation 
```
# create virtual environment
pip install -e .[dev,notebook]
```
### Jupyter notebook development
```
jupyter nbextension install itkwidgets --user --py
jupyter nbextension enable itkwidgets --user --py

jupyter notebook
```
### Tensorboard
```
tensorboard --logdir=/path/to/logs --port 6006
```
