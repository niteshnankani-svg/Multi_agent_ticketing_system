from ingest import ingest_ticket

messy = "   Hi,   my invoice is wrong!!\n\n\n   Please contact me at john.doe@email.com or 555-123-4567.\n   My card ending is 4111111111111111. Thanks.\n"

result = ingest_ticket('t-001', messy)
print('CLEAN:', result['clean_text'])
