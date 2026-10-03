# Gate SFT Train/Dev/Test Split Report

- Source: `gate_final_2855.sft.jsonl`
- Seed: `20261001`
- Total rows: **2855**
- Train / Dev / Test: **2287 / 284 / 284**

## Split policy

- LoCoMo grouped by conversation (`source.sample_id`).
- LongMemEval grouped by `source.question_id`.
- PersonaMem-v2 grouped by `source.persona_id`.
- RHELM grouped by `source.source_file` and kept in train.
- Logical groups that share an **exactly identical model input** were merged before splitting.
- This stricter duplicate rule prevents an identical prompt from appearing in both train and evaluation splits.

## Statistics

| split   |   rows |   retrieve_true |   retrieve_false |   zh |   en |   LoCoMo |   LongMemEval |   PersonaMem |   RHELM |
|:--------|-------:|----------------:|-----------------:|-----:|-----:|---------:|--------------:|-------------:|--------:|
| train   |   2287 |            1145 |             1142 | 1110 | 1177 |      980 |           556 |          503 |     248 |
| dev     |    284 |             142 |              142 |  138 |  146 |        0 |           220 |           64 |       0 |
| test    |    284 |             142 |              142 |  138 |  146 |        0 |           220 |           64 |       0 |

## Leakage checks

- Logical group overlap across splits: **0**
- Exact model-input duplicates across splits: **0**
- Resolvable `paired_positive_id` crossing splits: **0**

## Important consequence

The current dataset reuses some generic negative prompts across sources. Under strict no-exact-input-leakage grouping, those duplicates connect all LoCoMo conversations to the train-only RHELM component. Therefore this clean split keeps LoCoMo and RHELM in train, while dev/test are composed of held-out LongMemEval questions and PersonaMem personas.