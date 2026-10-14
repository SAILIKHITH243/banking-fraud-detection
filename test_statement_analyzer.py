
import pandas as pd

from statement_analyzer import (
    extract_amount,
    categorize_transaction,
    build_expense_summary,
    build_category_summary,
    detect_suspicious_transactions,
)

# Test 1: Amount conversion
print("TEST 1: Amount conversion")
print(extract_amount("₹1,250.50"))
print()

# Test 2: Expense categories
print("TEST 2: Expense categories")
print(categorize_transaction("Swiggy food order"))
print(categorize_transaction("Amazon shopping"))
print()

# Test 3: Sample transactions
sample = pd.DataFrame({
    "date": [
        "01/10/2026",
        "02/10/2026",
        "03/10/2026",
        "03/10/2026",
    ],
    "description": [
        "Swiggy food order",
        "Monthly salary credit",
        "Amazon shopping",
        "Amazon shopping",
    ],
    "debit": [450, 0, 1500, 1500],
    "credit": [0, 30000, 0, 0],
})

print("TEST 3: Weekly and monthly reports")
weekly, monthly = build_expense_summary(sample)
print("Weekly report:")
print(weekly)
print("\nMonthly report:")
print(monthly)

print("\nTEST 4: Spending by category")
print(build_category_summary(sample))

print("\nTEST 5: Transactions flagged for review")
all_transactions, flagged = detect_suspicious_transactions(sample)
print(flagged[["date", "description", "debit", "review_reason"]])

print("\nAll basic tests completed.")
