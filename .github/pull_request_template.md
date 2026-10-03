## Why
<!-- The problem, and what prompted it. Link the issue or HANDOFF.md item. -->

## What
<!-- The change, briefly. One concern per PR. -->

## Verification
<!-- Commands run and what they showed. For a fix: the test that fails without it. -->

## Not verified
<!-- Be explicit. Anything that needs the GPU VM, a real browser, or the network goes here. -->

## Checklist
- [ ] A test fails without this change and passes with it
- [ ] Each fix was mutation-checked (re-broken in a scratch copy; a test failed)
- [ ] `scripts/check.sh` passes (`--full` when deps, pins, Docker or the stack changed)
- [ ] `CHANGELOG.md` `[Unreleased]` updated
- [ ] CLAUDE.md invariants kept, or the PR explains why one changes
- [ ] Docs updated (README, CLAUDE.md, HANDOFF.md) where behaviour or commands changed
