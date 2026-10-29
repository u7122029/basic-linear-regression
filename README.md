# Basic Linear Regression

Slides, exercises and a workshop notebook on fitting a line of best fit.

| File | Contents |
|---|---|
| `main.tex` | Slides |
| `exercises.tex` | Pen-and-paper exercises |
| `workshop.ipynb` | Workshop: a guided toy example, then house prices on your own |
| `data/` | `toy.csv` and `house_prices.csv` |
| `src/lr_slides/lr_dataset_generator.py` | Generates `house_prices.csv` |

## Usage

Build the PDFs:

```sh
latexmk -pdf main.tex
latexmk -pdf exercises.tex
```

Run the notebook:

```sh
uv run --with jupyter jupyter lab workshop.ipynb
```
