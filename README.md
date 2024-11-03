# plotme

Easy plotting of common graphs with TSV inputs

## Installation
```
pip install git+https://github.com/supernifty/plotme
```


### bar
Generate a bar chart with main category (x) and optional sub-category (y) with values z

```
python plotme/bar.py --x x --y y --z z < test/box.tsv
```

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
Fits a list of data points to a given distribution.

```
python plotme/fit.py --xs 1 2 3 4 8 9 10 --distribution lognormal
```

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
Generate a pair plot.

```
python plotme/pair_plot.py --x1 x --x2 y --x1_name Apples --x2_name Oranges < test/pair.tsv
```

### pca.py

### pie.py

### scatter.py

### scatter_with_error.py

### segplot.py

### swimmer.py
