# Quality

## Universal Principles

- **Validate at boundaries.** Validate external input (user input, API responses, file reads) at the point of entry. Trust internal code after that — don't re-validate internally.
- **YAGNI.** Don't build what isn't needed yet. Solve the problem in front of you, not the hypothetical future one.

## Coding Standards

These are enforced by CI — code that doesn't meet them will not pass pipelines.

### All Languages

- **File size:** No source file may exceed 400 lines. Split files before they get there.
- **No debug prints:** Do not leave `print()`, `console.log()`, `fmt.Print`, or `println!` in source files. Use structured logging.

### Python

- **Type hints:** All function signatures must have type hints.
- **Linting:** ruff with rules E, F, D, UP, N (see `ruff.toml`). Docstrings follow Google convention.
- **Type checking:** mypy --strict. No untyped functions, no implicit Any.
- **Tests:** pytest, test files in `tests/`.

## Taste Invariants

- Prefer explicit over clever. Readable code beats concise code.
- One concept per function. If you need "and" to describe it, split it.
- Name things for what they are, not what they do.

## Definition of Done

- [ ] All CI stages pass (lint, type-check, tests)
- [ ] No debug prints in source
- [ ] No file exceeds 400 lines
- [ ] New behaviour has tests
