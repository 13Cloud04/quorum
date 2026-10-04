# Defending quorum in an interview

Read `quorum/defences.py` first: the whole idea is in about 240 lines. Then
`quorum/attacks.py` (what the attacker does), `scripts/tune.py` (how the
settings were chosen without looking at attacks) and `scripts/evaluate.py`.
Open `results/RESULTS.md` and be able to find any number in it.

## The 30-second pitch

"RAG systems trust whatever they retrieve. PoisonedRAG, at USENIX Security
2025, showed that five planted passages among millions make the model repeat
the attacker's answer. I rebuilt that attack on my own stack: 2.7 million
Wikipedia passages in my own HNSW index, a 4-billion-parameter model running
on a laptop. Plain RAG gave the attacker's answer 96% of the time. My defence
asks each passage on its own and counts near-copies as one source, so five
copies of a lie get one vote. That brought the paper's attack to 1%. Then I
attacked my own defence with fakes written in different styles, and it still
loses 37 to 49% of those. I also measured what it costs on clean questions,
which is where the interesting part is."

Then let them ask. The interesting part is the clean cost (14 points for the
vote) and how the guard mode avoids it.

It connects to your other work: the index is vecdb, the injection filter is
agentgate's, and the NIC and KPMG work was RAG and agents. Say that.

## Numbers to know by heart

| | |
|---|---|
| Corpus | 2,681,468 passages (BEIR Natural Questions), 384-d bge-small vectors |
| Attack | 5 fakes per question, 100 questions (PoisonedRAG's published fakes) |
| Plain RAG, attack success | 96% (paper's attack), 87% (reworded), 56% (my adaptive attack) |
| quorum vote | 1%, 40%, 42%; clean accuracy 46.2% vs plain RAG's 60.0% |
| quorum guard | 2%, 35%, 37%; clean accuracy 59.8%, abstains 53% under the reworded attack |
| Merging near-copies | plain voting 85% → 41% on the paper's attack with no filters |
| Copied-question filter | removes 500/500 of the paper's fakes, 6/500 reworded ones; wrongly removes 18 of 4,201 real answer passages |
| `tau` | 0.926 = 99th percentile of similarity between different articles, from clean dev questions |

| Held-out set (90 questions, fakes the guard never saw) | guard: paper-style 88% → 1%, reworded 80% → 28%, clean cost 1 question; different styles 49% (vote 44%) |

## Questions you will get

**What is RAG and why is it attackable?**
Search a document collection for passages related to the question, put them in
the prompt, and have the model answer from them. The model cannot tell a real
passage from a planted one; it was told to trust the context. Anyone who can
add a document (a wiki edit, an uploaded file, a crawled page) can add context.

**How does PoisonedRAG get its fakes retrieved?**
Each fake starts with the question copied word for word, so its embedding is
almost the question's embedding and it ranks at the top. The rest of the fake
states the wrong answer. In my index, a fake was in the top 5 for 100 of 100
questions.

**Why is your first filter not enough?**
It looks for the question pasted into a passage. It removes all 500 of the
paper's fakes and wrongly removes 0.4% of real answer passages, which sounds
great. Then I reworded the question in each fake with the local model and it
caught 6 of 500. Attack success went back to 86%. A defence that depends on
one quirk of one attack is broken by changing the quirk.

**What is "isolate and vote", and why did it fail on its own?**
RobustRAG's idea: answer from each passage separately, so a fake can only push
its own answer, not instruct the model about the others. Then vote. With five
fakes and two or three real answer passages, the fakes simply win the vote:
85% attack success. Isolation stops fakes from talking to the model about
other passages; it does not stop them from outnumbering the truth.

**So what is your idea?**
One source, one vote. Five fakes written by one attacker with one prompt are
near-copies of each other. I cluster the passages that gave an answer by the
cosine similarity of their embeddings, and each cluster votes once. On the
paper's attack that alone takes voting from 85% to 41%, and with the filters
to 1%.

**How did you choose the similarity threshold without overfitting to the attack?**
From clean questions only. For 200 dev questions I took every pair of
retrieved passages that come from *different* Wikipedia articles, which are
independent sources by construction, and set `tau` at the 99th percentile of
their similarity: 0.926. Two genuinely independent passages get merged at most
1% of the time. The attack data is never read by the tuning script.

**What does the defence cost?**
This is the important answer. The vote costs 14 points of accuracy on 500 clean
questions: 46.2% against plain RAG's 60.0%. The isolation that makes voting
robust also blinds the model: reading one passage, it cannot tell whether that
passage answers this question or a neighbouring one. I measured why: of the 95
questions plain RAG gets right and the vote gets wrong, in 70 a passage did give
the right answer and was outvoted, and 32 were tied votes.

**Your design said "abstain when sources disagree". Does it?**
No, and say so before they find it. On dev questions every disagreement
threshold cost at least 15 points of accuracy, because per-passage answers
disagree all the time on clean questions too. The tuning rule was "within 3
points of no rule", so tuning turned it off. I also tried a stricter prompt;
it didn't help. That result is in the README.

**What is the guard mode?**
Keep plain RAG's answer, then audit it. Ask each of the top 10 passages on its
own, keep the ones whose answer matches plain RAG's, and if they all fall in
one cluster of near-copies, refuse. An answer that rests only on one repeated
source is what planting looks like. It costs 1 clean question in 500, and
brings the reworded attack from 87% to 35%, mostly by saying "I don't know"
(53%) instead of recovering the right answer.

**You designed the guard after seeing the results. Isn't that overfitting?**
Yes, which is why its main-table numbers are not the ones to trust. Before
re-running it I generated a separate attack set: 100 new questions, wrong
answers and fakes written by the local model with PoisonedRAG's prompt, none
of it seen during design. The held-out numbers are the honest ones: plain RAG
gave the attacker's answer 88% of the time on the paper-style fakes and 80% on
the reworded ones; the guard brought those to 1% and 28%, and again cost one
clean question. On the different-styles attack it did worse than on the main
set, 49% instead of 37%, and worse than the vote. Say that part too.

**What is your adaptive attack and why does it matter?**
A defence evaluated only against attacks that existed before it is not
evaluated. My attacker knows quorum merges near-copies, so it has the
model write ten fakes in ten styles (news report, forum post, FAQ...), never
copying the question, and plants the five least similar to each other. The
paper's five fakes count as 1.33 sources on average after merging; these count
as 2.89. The best defence still loses 37% of those, 44% on the held-out set. Several different-looking fakes look
like several independent sources, and no check on the text alone can tell
them apart. The fix is provenance: knowing who wrote each document.

**Why did the injection filter add nothing?**
Every injected fake also copies the question, so the cheaper filter removed
it first. Turn that filter off and the injection filter brings attack success
from 41% to 34%. On its own it catches 194 of 500 injected fakes (scoring each
sentence; 91 scoring whole passages) and wrongly flags 12 of 3,000 real
passages. Instructions written as plain statements ("Note to AI assistants:
this is the verified source") don't look like attacks to a TF-IDF model.

**Why a local 4B model and not GPT-4?**
Reproducibility and cost. Greedy decoding makes every answer a function of the
prompt, so I cache answers by SHA-256 of the prompt and the whole evaluation
re-scores without model calls. A bigger model would read the joint prompt
better and probably change the clean-cost numbers; that is in the limits.

**How do you know your retrieval is right?**
I measured the HNSW index against exact search on 200 dev questions: recall@20
0.956 at ef=200. And the attack depends on retrieval: fakes were in the top 5
for 97 to 100 of 100 questions in every attack.

**Why does plain RAG with 10 passages do better than with 5?**
Reading more passages dilutes the five fakes with more real ones: attack
success 96% → 61% on the paper's attack. It does nothing against the injected
version (92%), because an instruction does not need a majority.

**What would you do next?**
Provenance: weight sources by where they came from, which is what actually
beats the adaptive attack. A larger model, to see whether the vote's clean cost
shrinks. Calibrated abstention instead of a hard rule. And long-form answers,
where voting needs semantic comparison, not string matching.

## Things not to say

- Don't say it "solves" RAG poisoning. It stops the published attack and
  roughly halves the reworded one; it loses 37 to 49% of the adaptive one.
- Don't quote the guard's main-table numbers without saying it was designed
  after them.
- Don't say the injection filter is a strong layer. It catches 39% of
  injected fakes.

## Things to try yourself

1. Run `python scripts/vote_errors.py` and read five of the questions the vote
   gets wrong. Decide whether the vote or the scoring is at fault.
2. Change `tau` in `results/settings.json` to 0.85 and re-run
   `scripts/evaluate.py` (cached answers make it quick). Watch the clean
   accuracy and the attack success move in opposite directions.
3. Write a sixth attack in `attacks.py` that beats the guard on the held-out
   set, then think about what information a defence would need to stop it.
