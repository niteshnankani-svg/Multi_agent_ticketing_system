import pandas as pd

new_examples = [
    # internal - plain English, HR/employee topics
    ("My paycheck was short this month, who do I contact?", "internal"),
    ("I need to update my emergency contact information.", "internal"),
    ("Can I request time off for next week?", "internal"),
    ("I was denied access to the shared drive, can IT fix my permissions?", "internal"),
    ("How do I enroll in the company health insurance plan?", "internal"),
    ("My badge isn't working to enter the building.", "internal"),
    ("I need to report a workplace conflict with a coworker.", "internal"),
    ("When is the deadline to submit my expense report?", "internal"),
    ("I haven't received my onboarding paperwork yet.", "internal"),
    ("Can you update my mailing address in the employee system?", "internal"),
    ("I need approval for remote work next month.", "internal"),
    ("My manager hasn't approved my timesheet yet.", "internal"),
    ("I want to check my remaining vacation days.", "internal"),
    ("How do I reset my company email password?", "internal"),
    ("I need a copy of my employment verification letter.", "internal"),

    # general - plain English, not fitting other categories
    ("What are your business hours?", "general"),
    ("Do you offer weekend support?", "general"),
    ("Where is your office located?", "general"),
    ("Can I schedule a demo of your product?", "general"),
    ("How do I contact your sales team?", "general"),
    ("What languages does your support team speak?", "general"),
    ("Do you have a mobile app?", "general"),
    ("Can you send me your company brochure?", "general"),
    ("Is there a phone number I can call for help?", "general"),
    ("What is your typical response time?", "general"),
    ("Do you offer training sessions for new users?", "general"),
    ("Can I get a tour of your facility?", "general"),
    ("What industries do you typically work with?", "general"),
    ("How long has your company been in business?", "general"),
    ("Can I speak with a manager?", "general"),
]

new_df = pd.DataFrame(new_examples, columns=["text", "category"])

train = pd.read_csv("data/raw/train.csv")

# duplicate these 15x each so they have real weight against 48,830 existing rows
boosted_new = pd.concat([new_df] * 15, ignore_index=True)

combined = pd.concat([train, boosted_new], ignore_index=True)
combined = combined.sample(frac=1, random_state=42)

print("added rows:", len(boosted_new))
print(combined["category"].value_counts())
combined.to_csv("data/raw/train_boosted2.csv", index=False)
print("saved: data/raw/train_boosted2.csv")
