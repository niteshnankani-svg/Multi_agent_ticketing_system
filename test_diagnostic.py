from test_pipeline import classify, ingest_ticket

tickets = [
    "My paycheck was short this month, who do I contact?",
    "Paycheck short this month. Contact required regarding discrepancy in salary payment.",
    "Regarding: Salary Payment Discrepancy - Employee Payroll Department Notification",
]

for text in tickets:
    state = ingest_ticket("t", text)
    cat, conf = classify(state["clean_text"])
    print(f"{cat:9} ({conf:.1%})  {text}")
