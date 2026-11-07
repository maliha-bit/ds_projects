"""
Python Project B: Command-line expense tracker (pandas + argparse + matplotlib).

    python p5_expense_tracker.py add 250 food "lunch"
    python p5_expense_tracker.py add 1200 transport "monthly pass" --date 2026-11-01
    python p5_expense_tracker.py list
    python p5_expense_tracker.py summary --month 2026-11
    python p5_expense_tracker.py chart --month 2026-11
    python p5_expense_tracker.py delete 3

Data is stored in expenses.csv next to the script.
"""
import argparse
import os
from datetime import date

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILE = "expenses.csv"
COLUMNS = ["id", "date", "amount", "category", "note"]


def load():
    if os.path.exists(FILE):
        return pd.read_csv(FILE)
    return pd.DataFrame(columns=COLUMNS)


def save(df):
    df.to_csv(FILE, index=False)


def filter_month(df, month):
    if month:
        df = df[df["date"].astype(str).str.startswith(month)]
    return df


def cmd_add(args):
    df = load()
    new_id = int(df["id"].max()) + 1 if len(df) else 1
    row = {
        "id": new_id,
        "date": args.date or date.today().isoformat(),
        "amount": args.amount,
        "category": args.category.lower(),
        "note": args.note,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save(df)
    print(f"Added #{new_id}: {args.amount} on {args.category}")


def cmd_list(args):
    df = filter_month(load(), args.month)
    print(df.to_string(index=False) if len(df) else "No expenses found.")


def cmd_summary(args):
    df = filter_month(load(), args.month)
    if df.empty:
        print("No expenses found.")
        return
    by_cat = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    print(by_cat.to_string())
    print(f"\nTotal: {df['amount'].sum():.2f}")
    print(f"Average per entry: {df['amount'].mean():.2f}")
    print(f"Biggest category: {by_cat.index[0]} ({by_cat.iloc[0] / by_cat.sum():.0%})")


def cmd_chart(args):
    df = filter_month(load(), args.month)
    if df.empty:
        print("No expenses found.")
        return
    by_cat = df.groupby("category")["amount"].sum()
    by_cat.plot(kind="pie", autopct="%1.0f%%", figsize=(5, 5))
    plt.ylabel("")
    plt.title(f"Spending by category {args.month or '(all time)'}")
    plt.tight_layout()
    plt.savefig("expense_chart.png", dpi=150)
    print("Saved expense_chart.png")


def cmd_delete(args):
    df = load()
    if args.id not in df["id"].values:
        print("ID not found.")
        return
    save(df[df["id"] != args.id])
    print(f"Deleted #{args.id}")


def main():
    ap = argparse.ArgumentParser(description="Expense tracker")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add")
    a.add_argument("amount", type=float)
    a.add_argument("category")
    a.add_argument("note", nargs="?", default="")
    a.add_argument("--date")
    a.set_defaults(func=cmd_add)

    for name, fn in [("list", cmd_list), ("summary", cmd_summary), ("chart", cmd_chart)]:
        p = sub.add_parser(name)
        p.add_argument("--month", help="YYYY-MM")
        p.set_defaults(func=fn)

    d = sub.add_parser("delete")
    d.add_argument("id", type=int)
    d.set_defaults(func=cmd_delete)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
