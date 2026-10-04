Settings (chosen on clean dev questions): `{"k": 10, "echo": true, "inject": true, "tau": 0.926, "min_sources": 1, "conflict": 99.0, "cluster": true}`

## PoisonedRAG's 100 questions (the paper's fakes, and three attacks built on them)

**Attack success rate: the attacker's answer was given (100 questions; lower is better)**

| system | poisonedrag | paraphrase | inject | diverse |
|---|---|---|---|---|
| vanilla k=5 | 96% | 87% | 99% | 56% |
| vanilla k=10 | 61% | 60% | 92% | 56% |
| filters + vanilla k=5 | 2% | 86% | 2% | 55% |
| isolate-vote k=10 | 85% | 80% | 97% | 52% |
| quorum vote | 1% | 40% | 1% | 42% |
| quorum guard | 2% | 35% | 2% | 37% |
| vote - echo filter | 41% | 40% | 34% | 42% |
| vote - injection filter | 1% | 40% | 1% | 42% |
| vote - merging | 1% | 80% | 1% | 52% |
| vote - both filters | 41% | 40% | 41% | 42% |

**Correct**

| system | clean | poisonedrag | paraphrase | inject | diverse |
|---|---|---|---|---|---|
| vanilla k=5 | 66% | 4% | 12% | 1% | 31% |
| vanilla k=10 | 77% | 36% | 36% | 8% | 38% |
| filters + vanilla k=5 | 66% | 66% | 13% | 66% | 32% |
| isolate-vote k=10 | 65% | 13% | 16% | 3% | 35% |
| quorum vote | 66% | 66% | 42% | 66% | 42% |
| quorum guard | 65% | 65% | 12% | 65% | 32% |
| vote - echo filter | 66% | 45% | 42% | 50% | 42% |
| vote - injection filter | 66% | 66% | 42% | 66% | 42% |
| vote - merging | 65% | 65% | 16% | 65% | 35% |
| vote - both filters | 66% | 45% | 42% | 44% | 42% |

**Abstained ("I'm not sure")**

| system | clean | poisonedrag | paraphrase | inject | diverse |
|---|---|---|---|---|---|
| vanilla k=5 | 7% | 0% | 1% | 0% | 1% |
| vanilla k=10 | 6% | 1% | 2% | 0% | 1% |
| filters + vanilla k=5 | 7% | 7% | 1% | 7% | 1% |
| isolate-vote k=10 | 5% | 0% | 0% | 0% | 0% |
| quorum vote | 5% | 5% | 0% | 5% | 0% |
| quorum guard | 8% | 8% | 53% | 8% | 20% |
| vote - echo filter | 5% | 0% | 0% | 0% | 0% |
| vote - injection filter | 5% | 5% | 0% | 5% | 0% |
| vote - merging | 5% | 5% | 0% | 5% | 0% |
| vote - both filters | 5% | 0% | 0% | 0% | 0% |

## The cost when nobody attacks

**500 clean test questions**

| system | correct | abstained | other wrong |
|---|---|---|---|
| vanilla k=5 | 60% | 10% | 30% |
| vanilla k=10 | 62% | 8% | 31% |
| filters + vanilla k=5 | 60% | 10% | 30% |
| isolate-vote k=10 | 46% | 3% | 51% |
| quorum vote | 46% | 3% | 51% |
| quorum guard | 60% | 10% | 30% |
| vote - echo filter | 46% | 3% | 51% |
| vote - injection filter | 46% | 3% | 51% |
| vote - merging | 46% | 3% | 51% |
| vote - both filters | 46% | 3% | 51% |

## Held-out set (made after the first results; fakes written by the local model)

**Attack success rate (90 questions; lower is better)**

| system | ho-poisonedrag | ho-paraphrase | ho-inject | ho-diverse |
|---|---|---|---|---|
| vanilla k=5 | 88% | 80% | 99% | 66% |
| vanilla k=10 | 56% | 50% | 89% | 54% |
| filters + vanilla k=5 | 1% | 80% | 1% | 66% |
| isolate-vote k=10 | 83% | 77% | 89% | 60% |
| quorum vote | 2% | 36% | 2% | 44% |
| quorum guard | 1% | 28% | 1% | 49% |
| vote - echo filter | 39% | 37% | 34% | 44% |
| vote - injection filter | 2% | 36% | 2% | 44% |
| vote - merging | 1% | 76% | 1% | 60% |
| vote - both filters | 39% | 37% | 40% | 44% |

**Correct**

| system | ho-clean | ho-poisonedrag | ho-paraphrase | ho-inject | ho-diverse |
|---|---|---|---|---|---|
| vanilla k=5 | 63% | 9% | 14% | 1% | 17% |
| vanilla k=10 | 64% | 31% | 33% | 7% | 26% |
| filters + vanilla k=5 | 62% | 63% | 14% | 63% | 17% |
| isolate-vote k=10 | 52% | 8% | 11% | 8% | 17% |
| quorum vote | 51% | 53% | 36% | 53% | 28% |
| quorum guard | 62% | 63% | 13% | 63% | 17% |
| vote - echo filter | 51% | 31% | 34% | 38% | 28% |
| vote - injection filter | 51% | 53% | 36% | 53% | 28% |
| vote - merging | 52% | 53% | 11% | 53% | 17% |
| vote - both filters | 51% | 31% | 34% | 31% | 28% |

**Abstained**

| system | ho-clean | ho-poisonedrag | ho-paraphrase | ho-inject | ho-diverse |
|---|---|---|---|---|---|
| vanilla k=5 | 2% | 0% | 0% | 0% | 0% |
| vanilla k=10 | 2% | 0% | 0% | 0% | 0% |
| filters + vanilla k=5 | 2% | 2% | 0% | 2% | 0% |
| isolate-vote k=10 | 1% | 0% | 0% | 0% | 0% |
| quorum vote | 1% | 1% | 0% | 1% | 0% |
| quorum guard | 2% | 2% | 54% | 2% | 20% |
| vote - echo filter | 1% | 0% | 0% | 0% | 0% |
| vote - injection filter | 1% | 1% | 0% | 1% | 0% |
| vote - merging | 1% | 1% | 0% | 1% | 0% |
| vote - both filters | 1% | 0% | 0% | 0% | 0% |

## Retrieval

| condition | questions with a poison in the top 5 | poisons in the top 5 (mean) | gold passage in the top 10 |
|---|---|---|---|
| clean | 0 | 0.00 | 71% |
| poisonedrag | 100 | 4.95 | 59% |
| paraphrase | 98 | 4.63 | 59% |
| inject | 100 | 4.93 | 59% |
| diverse | 97 | 3.46 | 63% |
| clean-test | 0 | 0.00 | 71% |
| ho-clean | 0 | 0.00 | 70% |
| ho-poisonedrag | 90 | 4.64 | 61% |
| ho-paraphrase | 90 | 4.38 | 62% |
| ho-inject | 90 | 4.62 | 61% |
| ho-diverse | 90 | 3.51 | 63% |
