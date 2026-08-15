from semantic_cache import check_cache, store_in_cache

print("First check (should be None, nothing cached yet):")
print(check_cache("How long does a refund take?"))

store_in_cache(
    "How long does a refund take?",
    "Refunds are processed within 5-7 business days."
)

print("\nParaphrased question (should return the cached answer):")
print(check_cache("How many days does a refund usually take?"))

print("\nUnrelated question (should be None):")
print(check_cache("Is the VPN down right now?"))
