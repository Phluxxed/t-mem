# PRD: Memory Selection Adapter for t-mem

## Overview

Build a modular memory selection adapter that sits between `t-mem` and the agent harness.

`t-mem` remains responsible for:
- extracting trajectory-informed tips from prior sessions
- storing and embedding those tips
- retrieving a candidate set via semantic similarity

The adapter becomes responsible for:
- rescoring or filtering the `t-mem` candidate set
- selecting which tips are actually injected into the harness
- evolving independently through offline evaluation and periodic model updates

The first target harness is Claude Code. The design should not depend on Claude-specific internals beyond the existing hook-style retrieval flow.

## Problem

`t-mem` currently retrieves tips using cosine similarity over embedded tip keys, with thresholding and top-k truncation. This is simple, legible, and useful, but it optimises for semantic resemblance rather than downstream utility.

Failure modes:
- tips that look similar but are not actually useful get injected
- useful tips may be missed because they are phrased differently
- cosine similarity ignores metadata already present in the tip schema
- prompt budget is spent on plausible-but-low-value injections
- retrieval quality is difficult to improve without modifying `t-mem` itself

The goal is to improve injection quality without turning `t-mem` into a broader memory platform or contaminating its simple design.

## Goal

Create a thin, modular layer that improves the quality of injected tips while preserving `t-mem`’s current flavour and fallback behavior.

Success means:
- better tip selection than raw cosine top-k
- minimal or no invasive changes to `t-mem`
- the ability to evolve the selector weekly using offline DS workflows
- clean fallback to native `t-mem` retrieval if the adapter is disabled

## Non-goals

Not in scope for v1:
- replacing `t-mem` extraction
- replacing `t-mem` storage
- replacing embedding-based candidate generation
- building full long-term memory or stateful agent memory
- LLM fine-tuning
- real-time optimisation on every run
- RL-based policy optimisation unless later justified by evidence
- cross-store memory federation beyond `t-mem`

## Product shape

### Current flow

User prompt → `t-mem` retrieves tips → tips injected into Claude Code

### Proposed flow

User prompt → `t-mem` retrieves candidate tips → adapter scores, filters, or reranks → selected tips injected into Claude Code

This makes the adapter a selection membrane rather than a new memory system.

## Design principles

### Preserve flavour

`t-mem` remains understandable, opinionated, and useful on its own.

### Bounded component

The adapter optimises one narrow decision: which candidate tips deserve prompt space.

### Reversible

If the adapter fails, disable it and fall back to `t-mem` cosine retrieval.

### Offline evolution

Model improvement happens on a weekly cadence using logged retrieval data and DS experimentation.

### Harness-light contract

The adapter should rely on a small stable interface so it can later sit between other memory stores and other harnesses.

## Users

### Primary user

You, operating an agent harness and iterating on retrieval quality for your own workflows.

### Secondary user

Future versions of your broader memory stack, which may reuse the same adapter pattern across other memory stores.

## Core use cases

### UC1: Standard retrieval with adapter enabled
1. Harness receives user prompt
2. `t-mem` returns top-N candidate tips with metadata and cosine scores
3. Adapter rescoring chooses top-K final tips
4. Selected tips are injected into prompt context

### UC2: Adapter disabled
1. Harness receives user prompt
2. `t-mem` returns final tips using native threshold and top-k
3. No adapter intervention

### UC3: Weekly optimisation cycle
1. Retrieval and injection events are logged
2. DS agent trains and evaluates a new selector offline
3. New selector is compared against incumbent on held-out data
4. New selector is promoted only if it wins baseline metrics

### UC4: Diagnostics
1. Operator inspects why a tip was selected or rejected
2. Adapter exposes component scores or ranked-feature contributions
3. Poor selector behavior is traceable without reading tea leaves

## Functional requirements

### 1. Candidate intake

The adapter must accept a candidate set from `t-mem`, where each candidate includes:
- tip ID
- content
- title if present
- category
- priority
- trigger
- steps if present
- negative example if present
- subtask description if present
- source project
- source session ID
- created timestamp
- cosine similarity score from `t-mem`

### 2. Query context intake

The adapter must accept:
- raw prompt or query text
- session ID if available
- optional harness metadata such as project or repo context when easy to provide

### 3. Selection modes

The adapter must support at least:
- passthrough mode
- heuristic rerank mode
- learned rerank mode

### 4. Output

The adapter must return:
- selected tips for injection
- final score per tip
- optional debug metadata explaining selection

### 5. Logging

The system must log enough information to support offline analysis and model training, including:
- query text or query snippet
- candidate set before final selection
- selected set
- scores used for ranking
- run metadata such as timestamp and session ID
- later session-level outcome signals where available

### 6. Safety and fallback

If scoring fails, the adapter must return a safe fallback result rather than block retrieval. Preferred fallback:
- raw `t-mem` ordering
- or heuristic-only scoring if learned scorer is unavailable

## v1 scoring strategy

The first version should not replace `t-mem` retrieval. It should rerank `t-mem` candidates.

### Phase 1

Hand-built heuristic scorer over the candidate pool.

Illustrative shape:

`score = a*cosine + b*priority + c*project_match + d*trigger_overlap + e*subtask_overlap - f*redundancy`

### Phase 2

Small learned reranker trained offline on logged examples.

Candidate model families:
- logistic regression
- gradient boosted trees
- pairwise ranker
- small MLP only if simpler models plateau

Default recommendation for the first learned version:
- gradient boosted trees over tabular retrieval features

Reason:
- fast iteration
- handles nonlinear interactions
- interpretable enough to debug
- well matched to sparse, mixed metadata features

## Features for scoring

The adapter should support a feature pipeline that can evolve independently of `t-mem`.

### Initial feature groups

#### Similarity features
- cosine similarity from `t-mem`
- lexical overlap between prompt and tip trigger
- lexical overlap between prompt and tip content
- lexical overlap between prompt and subtask description

#### Metadata features
- tip category
- tip priority
- source project match
- recency
- age bucket
- presence or absence of steps
- presence or absence of negative example
- title present

#### Historical usefulness features
- prior retrieval count
- prior injection count
- historical acceptance rate if later added
- historical usefulness label if available
- repeated selection frequency in the same or similar context

#### Redundancy features
- pairwise similarity against already selected tips
- duplicate or near-duplicate content indicator
- same subtask cluster indicator

#### Session-aware features
- session ID repeat suppression
- whether already injected in the current session window
- match against recent prior prompts in the same session if available

## Labels and training targets

The learned selector should not initially try to predict relevance in the abstract. It should predict utility for injection.

### Initial labels

Because high-quality causal labels will be sparse at first, v1 should support weak and intermediate supervision.

#### Weak labels
- retrieved and injected
- retrieved but not selected
- high cosine candidate not selected
- selected by heuristic baseline

These are easy but noisy.

#### Better labels over time
- injected tip followed by lower recovery burden in session
- injected tip followed by fewer repeated failed operations
- injected tip aligned with later successful subtask resolution
- candidate missed in selection but later clearly relevant from session evidence

#### Pairwise preference labels

For a given candidate set:
- candidate A preferred over candidate B if A was selected and B was not under a stronger selector
- candidate A preferred if A historically correlates with better downstream outcomes than B in similar contexts

Pairwise training is likely more natural than absolute classification once enough data exists.

## Evaluation

The selector must be evaluated against strong baselines, not against vibes in a trench coat.

### Baselines
- raw `t-mem` cosine threshold plus top-k
- heuristic reranker
- heuristic reranker with redundancy penalty

### Offline metrics
- precision at k for selected tips
- recall of later-judged-useful tips where labels exist
- pairwise ranking accuracy
- NDCG or MAP if ranking labels become available
- selector agreement and disagreement against baseline

### Downstream proxy metrics
Using session-level telemetry where possible:
- recovery rate
- repeated-op rate
- retries after error
- session length
- turns to completion
- prompt token cost from injected memory
- rate of obviously irrelevant injections

### Acceptance rule
A newly trained selector is promoted only if it beats the incumbent on offline metrics and does not worsen key downstream proxies beyond agreed tolerance.

## Logging requirements

This is the load-bearing piece.

### Minimum v1 log event
For each retrieval event:
- retrieval event ID
- timestamp
- session ID if present
- query snippet
- candidate tip IDs
- raw cosine scores
- final selected tip IDs
- selector mode used
- selector version
- feature snapshot used for selection if cheap enough

### Nice-to-have later
- prompt token cost added by injected tips
- whether selected tips were repeated in the same session
- session-level outcome summary
- eventual label backfills from later analysis jobs

### Constraint
Logging should be additive and should not force invasive redesign of `t-mem`.

## Architecture

### Components

#### `t-mem`
Owns:
- extraction
- persistence
- embeddings
- initial candidate retrieval

#### Adapter
Owns:
- feature generation from prompt and candidate metadata
- heuristic or learned scoring
- final selection
- logging of selection decisions

#### Harness integration
Owns:
- invoking `t-mem`
- invoking the adapter
- injecting final selected tips into Claude Code prompt path

#### Weekly DS workflow
Owns:
- training data assembly
- feature experiments
- model training
- offline evaluation
- model packaging and versioning
- promotion or rollback decision support

## API contract

### Candidate request
Harness calls the adapter with:
- `query`
- `session_id`
- optional `context`
- `candidates[]`

### Candidate object
Each candidate should include:
- `tip_id`
- `content`
- `title`
- `category`
- `priority`
- `trigger`
- `steps`
- `negative_example`
- `subtask_description`
- `source_project`
- `source_session_id`
- `created_at`
- `cosine_score`

### Adapter response
- `selected_tip_ids`
- `scores_by_tip`
- `selector_version`
- `mode`
- optional `reasons_by_tip`

The contract should stay serializable and harness-agnostic.

## Rollout plan

### Milestone 1
Adapter shell with passthrough and heuristic mode.

Success:
- can run between `t-mem` and the harness
- no behavior regression in passthrough mode
- heuristic mode produces debuggable rankings

### Milestone 2
Richer logging and offline analysis pipeline.

Success:
- can reconstruct candidate pools and selections from logs
- can run baseline comparisons offline

### Milestone 3
First learned reranker.

Success:
- beats heuristic baseline on held-out retrieval data
- safe fallback exists

### Milestone 4
Shadow deployment.

Success:
- learned selector scores live candidate sets without affecting production output
- disagreement analysis reveals sensible behavior

### Milestone 5
Controlled activation.

Success:
- learned selector influences live injections
- rollback path works
- downstream proxy metrics remain neutral or improve

## Risks

### Label fraud
Selected does not mean useful.

Mitigation:
- add downstream and comparative signals over time
- treat early labels as weak supervision only

### Overfitting to phrasing
Model may learn stylistic affinity rather than utility.

Mitigation:
- evaluate against downstream proxies
- test on held-out session families
- include project and subtask generalization features

### Adapter bloat
The adapter could metastasize into a second memory system.

Mitigation:
- keep candidate generation in `t-mem`
- keep persistence out of the adapter
- enforce a narrow API boundary

### Debug opacity
A learned scorer can become harder to reason about than cosine.

Mitigation:
- start with a heuristic baseline
- prefer interpretable model classes initially
- emit debug scores and feature contributions

### Harness coupling
The adapter may become too dependent on Claude Code specifics.

Mitigation:
- keep the data contract generic
- only require query, session ID, and candidate set

## Open questions

These do not block building the system, but they affect later quality.

- What exact downstream signals from Claude Code sessions are easiest to harvest reliably?
- Should ranking be global or split by tip type such as strategy versus recovery versus optimisation?
- Should the adapter ever abstain and inject nothing even when `t-mem` found candidates?
- Is top-k fixed or should the adapter learn a variable injection budget?
- When broader memory arrives, should the same adapter rank across stores or should each store have its own scorer?

## Recommended v1 decisions

Lock these in for the first pass:
- `t-mem` remains unchanged except for additive logging where needed
- adapter is a separate module or package
- `t-mem` still performs candidate recall
- adapter reranks top 20 to 50 candidates
- first production scorer is heuristic
- first learned scorer is gradient boosted trees
- model retraining cadence is weekly, manual or semi-automated
- fallback is raw `t-mem` ordering

## Definition of done for v1

v1 is done when:
- the adapter can sit between `t-mem` and Claude Code without breaking the existing flow
- passthrough mode exactly preserves baseline behavior
- heuristic mode works and is inspectable
- logging captures candidate pools and final selections
- offline DS workflow can train and compare a learned reranker
- learned reranker can be shadow-tested and safely rolled back
- the system improves or holds downstream quality without inflating irrelevant injections
