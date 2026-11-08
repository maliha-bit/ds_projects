

Six projects covering machine learning, deep learning and Python tooling. Each folder has its own README with details and results.

| Project | What it is | Key result |
|---|---|---|
| [p1-house-price](p1-house-price) | End-to-end regression pipeline comparing 4 models | Random Forest, test R2 = 0.806 |
| [p2-lenet-mnist](p2-lenet-mnist) | LeNet-5 reproduction vs a modern CNN on MNIST | 98.73% vs 99.12% accuracy |
| [p3-plant-disease](p3-plant-disease) | ResNet18 transfer learning, 38 plant disease classes | 71% (small subset, 2 epochs) |
| [p4-book-scraper](p4-book-scraper) | Web scraper plus price analysis | Cleaned dataset and charts |
| [p5-expense-tracker](p5-expense-tracker) | Command-line expense tracker | CLI with summaries and charts |
| [p6-csv-explorer](p6-csv-explorer) | Streamlit app to explore any CSV and train a baseline | Interactive web app |

## Setup
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Then run the script inside each project folder (see its README).

