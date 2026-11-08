# Python Project: Command-Line Expense Tracker

A small command-line tool to record expenses and see where the money goes, built with Python, pandas and matplotlib.

## Features
- Add an expense with amount, category, note and optional date
- List expenses (optionally for one month)
- Summary by category: total, average, and the biggest category
- Pie chart of spending by category
- Delete an entry by ID
- Data is stored in a simple `expenses.csv` file

## Usage
```
python p5_expense_tracker.py add 250 food "lunch"
python p5_expense_tracker.py add 1200 transport "bus pass" --date 2026-11-01
python p5_expense_tracker.py list
python p5_expense_tracker.py summary --month 2026-11
python p5_expense_tracker.py chart --month 2026-11
python p5_expense_tracker.py delete 3
```

## Skills shown
Command-line interfaces with `argparse` subcommands, file persistence with CSV, pandas groupby, and plotting.
