

## Project structure

```text
CAMMDA/
├── data/
│   ├── HMDD v2.0/
│   └── HMDD v3.2/
├── result/                      # Predictions/metrics written by train.py
├── eval.py                      # Run entry, same default evaluation parameters
├── CAMMDA.py                    # Original CAMMDA four-channel model, 2 API aliases
├── train.py                     # MAMFGAT-aligned KFold/graph/metrics + CAMMDA BCE
├── utils.py                     # Exact copy of the uploaded MAMFGAT utils.py
├── requirements.txt
└── README.md
```

