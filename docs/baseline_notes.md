# Baseline smoke reference

Environment:
- train_gpt.py from parameter-golf main
- 1 GPU
- TRAIN_BATCH_TOKENS=65536
- VAL_BATCH_SIZE=65536
- ITERATIONS=100

Observed:
- initial val_bpb: 4.1077
- step 25 val_bpb: 3.1714
- step 50 val_bpb: 3.0110
- step 75 val_bpb: 2.7713
- step 100 val_bpb: 2.7061
- peak memory allocated: 1552 MiB
- int8+zlib artifact: 5578868 bytes