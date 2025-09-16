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
python plotme/box.py --x Class --z 'Sepal Length' --points --points_jitter 0.1 < test/iris.tsv
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

### pie.py
Use col as the category column and val as the value column.
```
python plotme/pie.py --col Class --val 'Sepal Length' < test/iris.tsv
```

### scatter.py
Generate a scatter plot.

```
python plotme/scatter.py --x 'Sepal Length' --y 'Sepal Width' --z Class --z_color --figsize 8 --line_of_best_fit_by_category < test/iris.tsv
```

Generate a (simple) 3D scatter plot:
```
python plotme/scatter.py --x 'Sepal Length' --y 'Sepal Width' --projection 'Petal Length' --z Class --z_color --figsize 8 < test/iris.tsv
```

with density overlay
```
python plotme/scatter.py --x 'Sepal Length' --y 'Sepal Width' --z Class --z_color --figsize 8 --line_of_best_fit_by_category --density < test/iris.tsv
```

### scatter_with_error.py
Generate a scatter plot with error bars on y.

```
python plotme/scatter_with_error.py --x 'Sepal Length' --y 'Sepal Width' --yl 'Sepal Low' --yh 'Sepal High' < test/err.tsv
```

### segplot.py
Generate a segment plot (horizontal barchart with error bars).

```
python plotme/segplot.py --x c --y sc --lower l --mean m --upper u --separator --height 4 < test/seg.tsv
```

### swimmer.py
Generate a swimmer plot.

```
python plotme/swimmer.py --data test/swimmer.tsv --indicator Sex --start 50
```

### umap_helper.py
Add umap columns to a table

```
python plotme/umap_helper.py --cols 'Sepal Length' 'Sepal Width' 'Petal Length' 'Petal Width' --cluster --normalise < test/iris.tsv > iris.umap.tsv
```

Now you can view the clustering as a scatter plot:
```
python plotme/scatter.py --x 'umap0' --y 'umap1' --z cluster --z_color --figsize 8 < test/iris.umap.tsv
```

Or see the true classes:
```
python plotme/scatter.py --x 'umap0' --y 'umap1' --z Class --z_color --figsize 8 < test/iris.umap.tsv
```
