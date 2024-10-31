# plotme

Easy plotting of common graphs with TSV inputs

## Installation
```
pip install git+https://github.com/supernifty/plotme
```


### bar

### box
Generate a box plot with main category (x) and optional sub-category (y) with values z

```
python plotme/box.py --x x --y y --z z < test/box.tsv
```

### density.py
Generate a density graph grouped by column group, and with values from column value.

```
python plotme/density.py --group x --value z < test/box.tsv
```

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
