# Tip Consolidation

Deduplicate accumulated tips without losing distinct signals.

## Problem

Running `extract-all` across many sessions will produce near-duplicate tips — same advice
extracted multiple times from similar situations. Without consolidation the store grows
linearly and retrieval quality degrades (top-5 slots filled with variations of the same tip).

## Approach

Two-stage pipeline:

1. **Similarity pass** — find candidate clusters of tips with cosine similarity ≥ 0.88
2. **LLM merge pass** — for each cluster, call LLM to synthesise a canonical tip or
   confirm tips are distinct enough to keep separate

The LLM is the safety valve. High similarity flags candidates; the LLM decides.
Prompt biases strongly toward "keep separate" — asymmetric risk (merging destroys signal,
keeping a near-duplicate just adds noise).

## Schema

Add two columns to `tips`:

```sql
ALTER TABLE tips ADD COLUMN status TEXT NOT NULL DEFAULT 'active';
-- 'active' | 'merged'
ALTER TABLE tips ADD COLUMN merged_into TEXT REFERENCES tips(id);
```

All existing queries get `WHERE status = 'active'`. Merged tips are soft-deleted:
`status = 'merged'`, `merged_into = <canonical_tip_id>`. Reversible by flipping status back.

Migration goes in `store.migrate_add_consolidation_columns()`, called from
`fm db migrate`.

## Clustering Algorithm

1. Load all active tips with embeddings (same provider)
2. Compute pairwise cosine similarity — O(n²), fine for current corpus size
3. Build clusters greedily: if sim(A, B) ≥ 0.88, they belong to the same cluster
   (union-find to handle transitive membership: if A~B and B~C, all three in one cluster)
4. Discard singleton clusters (nothing to merge)

## LLM Merge Pass

For each cluster of ≥ 2 tips, call the LLM with:
- Full details of each tip: content, purpose, trigger, steps, negative_example, priority
- Instruction: determine if these are genuinely the same advice or distinct

LLM returns one of:
- `{"action": "merge", "content": "...", "purpose": "...", "trigger": "...", "steps": [...], "negative_example": "...", "priority": "...", "reasoning": "..."}` — synthesised canonical tip
- `{"action": "keep", "reasoning": "..."}` — tips are distinct, leave as-is

On `merge`:
- Create a new `Tip` with synthesised content, new UUID, `status = 'active'`
- Priority = highest priority among merged tips
- `source_session_id` = comma-joined source session IDs
- Mark all original tips: `status = 'merged'`, `merged_into = new_tip.id`
- Embed the new tip immediately

On `keep`: no changes.

## CLI

```
fm tips consolidate [--threshold FLOAT] [--dry-run] [--model TEXT]
```

- `--threshold`: cosine similarity cutoff for candidates (default: 0.88)
- `--dry-run`: show proposed merges without applying, exit 0
- `--model`: Claude model for merge decisions (default: sonnet — synthesis quality matters)

### Dry-run output format

```
Cluster 1 (3 tips, max_sim=0.94):
  [abc12345] [HIGH strategy] When reading files, check existence before opening...
  [def67890] [HIGH strategy] Before reading a file, verify the path exists to avoid...
  [ghi11121] [MEDIUM strategy] Always check file existence prior to read operations...
  → PROPOSED MERGE: "Before reading a file, verify the path exists..."

Cluster 2 (2 tips, max_sim=0.91):
  [jkl31415] [HIGH recovery] After a failed tool call, check stderr before retrying...
  [mno92653] [CRITICAL recovery] When a tool call fails, inspect the error output...
  → PROPOSED MERGE: "When a tool call fails, inspect stderr before retrying..."

2 clusters, 5 tips → 2 canonical tips (saves 3 slots)
Run without --dry-run to apply.
```

### Live run output

```
Cluster 1/2: merging 3 tips → 1 canonical... done
Cluster 2/2: merging 2 tips → 1 canonical... done
Consolidated 5 tips into 2. 3 tips soft-deleted.
```

## Implementation Steps

- [ ] `store.migrate_add_consolidation_columns()` — add status/merged_into columns, backfill existing rows with status='active'
- [ ] Update `store.list_tips()`, `store.get_tips_with_embeddings()` to filter `WHERE status = 'active'`
- [ ] `store.add_tip()` — no changes needed (new tips default to 'active')
- [ ] `store.mark_merged(tip_ids: list[str], canonical_id: str)` — batch soft-delete
- [ ] `fm/consolidator.py` — similarity pass + clustering logic
- [ ] `fm/prompts/consolidate.py` — LLM merge prompt
- [ ] `fm tips consolidate` CLI command
- [ ] `fm db migrate` calls new migration
- [ ] Tests

## Out of Scope

- Cross-provider consolidation (tips embedded by different providers aren't comparable)
- Scheduled / automatic consolidation (manual command only for now)
- Conflict resolution beyond priority (no "success-derived beats failure-derived" logic yet)
