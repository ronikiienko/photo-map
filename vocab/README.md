# Label vocabulary

Plain English labels for naming groups of photos (`src/name_regions.py`). Files written in rounds by a language model,
each round less common than the one before: `curated2_round1.txt` (the 1000 most basic things in photos), `round2` (the next
1000), `round3` (the next 1000). Added later: `round1b` (more basic phrases, counted as common as round 1), `round4` and `round5`
(rarer than round 3); most of what was written for them was already in the earlier rounds, so only about 380 phrases are new. Format: a `# general` block (broad labels: animal, vehicle, food) then a `# specific` block
(owl, bicycle, pizza), one phrase per line. A phrase is kept once (the earliest round).

Style: lowercase, singular, 1-3 words, no proper names, no adjectives on their own, plain words, no filler.

The round counts as how common a phrase is: `src/name_regions.py` adds `COMMON * log10(1000 | 100 | 10)` to a phrase's score.
Edit the files freely; `vocab_embed.py` re-embeds them when they change (vectors are cached in data/vocab/, not in git).

## How the lists were made

Each round was one message to the same language-model agent, which could not read the project (only write the output file):

> Write a vocabulary of plain English labels from your own knowledge. It will be used to name groups of photos from any photo
> collection: a text-embedding model matches each label against a group's average, and the best match becomes the group's name.
> Every label should be a natural word or short phrase an ordinary person would use to say what a photo shows. It must work for
> any collection, so assume no particular source, theme or country.
> Round 1: the 1000 most common, most basic things people photograph or that appear in photos, across everything (no categories,
> no balancing). Round 2/3: the next 1000, less common than the earlier rounds but still plain things an ordinary person would
> recognise; nothing repeated from earlier rounds. Two tiers: general (about one in eight) and specific. Style rules as above.

Tried and left out: ImageNet-21k class names (WordNet: rare species and odd senses win, no plain "lake"/"forest"), Places365 scene
names alone (too few, indoor scenes), common phrases of the Wikivoyage captions (fragments, adjectives), and a first hand-balanced
list of 18 domains (misses areas, equal quotas are wrong for real collections).
