"""Questions written by me AFTER tuning, to check generalisation (not part of eval.jsonl).
(question, should_answer, expected substring or None)"""
DEV = [
    ("what's the price of pickup point delivery?", True, "3.50"),
    ("Do you accept Apple Pay?", False, None),
    ("how many characters must my password have", True, "10 characters"),
    ("Can I pay with a gift card and then get my money back in cash?", True, "cannot be exchanged for cash"),
    ("what size helmet for a 60 cm head", True, "size l"),
    ("how long do you keep my invoices", True, "10 years"),
    ("does the student code work with another promo code", True, "cannot be combined"),
    ("Where is my order? tracking number is 12 digits", True, "tracking"),
    ("Do you sell electric scooters?", False, None),
    ("when are pre-orders charged", True, "ships"),
    ("how long does a replacement lock key take to arrive", True, "10 working days"),
    ("can I ship a battery to Germany", True, "inside france only"),
    ("How do I contact you by phone?", False, None),
    ("What's the delivery time to Spain?", True, "3 to 7"),
    ("how fast do you answer emails", True, "24 hours"),
    ("will my account be deleted immediately", True, "30 days"),
]
