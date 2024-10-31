# plotme

Easy plotting of common graphs with TSV inputs

## Installation
```
pip install git+https://github.com/supernifty/plotme
```


### bar

### box
Generate a box plot with main category (x) and sub-category (y) with values z

```
python plotme/box.py --x x --y y --z z < test/box.tsv
```

### density.py

### fit.py

### heatmap.py
Plot a heatmap.

```
python plotme/heatmap.py --x x --y y --z z < test/box.tsv
```

### hist
Plot a histogram.

```
python plotme/hist.py --label x --value z < test/box.tsv
```

### pair_plot.py

### pca.py

### pie.py

### scatter.py

### scatter_with_error.py

### segplot.py

### swimmer.py
