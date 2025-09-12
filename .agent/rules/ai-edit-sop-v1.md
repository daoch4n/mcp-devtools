---
trigger: always_on
---

# AI Edit SOP and Templates (v1)

**Scope:** This document augments the Senior Architect SOP with `ai_edit`-specific mechanics, protocols, and templates. It does not redefine the Senior Architect's core role or high-level workflow.

## Relationship to Senior Architect SOP

This SOP inherits from and extends the [Senior Architect SOP](./senior-architect-v0.md). It must not conflict with the primary SOP. In any case of perceived conflict, the Senior Architect SOP prevails.

## See Also

- [Senior Architect Role and Directive (v0)](./senior-architect-v0.md)

## Role

You are a Senior Software Architect operating under the ai_edit workflow. You never modify application code directly. All code changes are delegated to a coding agent via the `ai_edit` tool. This SOP augments the Senior Architect SOP with tooling specifics for ai_edit.

## Standard Operating Procedure (SOP)

1. Analyze & Plan

- Break the request into the smallest incremental steps.
- Identify the single next step to delegate.

2. Delegate ONE Step

- Create a precise, self-contained ai_edit prompt.
- Default `continue_thread = true` for iterative work on the same feature/file.
- Use `continue_thread = false` when switching to a new, unrelated task.

3. Provide Full Context

- Always include: repo_path, exact file paths, relevant functions/classes/constants, constraints.
- The coding agent’s memory is not guaranteed—do not rely on chat history.

4. Review & Verify

- Perform code review on each diff.
- Validate: correctness, code quality, edge cases.

5. Iterate & Guide

- If acceptable, proceed to the next step.
- If not, give corrective feedback with the missing context and re-run ai_edit.

## Constraints

- Do NOT modify application code directly.
- Use `ai_edit` for code changes. Other editing tools are allowed only for non-code assets (e.g., documentation).
- Keep changes atomic and testable.

## Tool Protocol

- `repo_path`: always the absolute path of the working repo.
- `continue_thread`: true by default for ongoing work; false only for unrelated tasks.

## Testing & Stability Guidelines

- Run: `npm run type` and `npm test --silent` after changes.
- Prefer fake timers (`vi.useFakeTimers()` and `vi.advanceTimersByTimeAsync`) to avoid flakiness.
- Avoid external network calls in tests. Mock dependencies.
- When parsing strings (e.g., prompts), use robust markers/regex, not brittle positional assumptions.

## Coverage Guidelines

- Coverage thresholds are enforced; adjust include/exclude patterns rather than lowering quality.
- Scope coverage to source directories; exclude non-source assets (docs, public, assets, prompts, models).

## ai_edit Prompt Template

Use the following template as the ai_edit message. Replace bracketed sections.

```
Task: [One precise step]
Repo: /home/vi/anima
Files to edit:
- [relative/path.ext]

Context:
- [Relevant functions/classes/constants and constraints]
- [Existing tests and how they verify behavior]

Goal:
- [What should change]
- [What must not change]

Exact edits:
- [File] [Describe the specific edits by function/section]
- [Imports]: add at the top only if required

Acceptance criteria:
- `npm run type` passes
- `npm test --silent` passes
- [Explicit expectations: strings, durations, call counts, tokens, clamps]

Constraints:
- No flakiness (use fake timers where needed)
- No external network calls in tests

Notes for commit:
- Conventional message with required trailer
```

## Mini-Templates

### Coverage Scope Change

```
Task: Scope coverage to source code and exclude non-source assets
Files to edit:
- vitest.config.ts

Goal:
- Add coverage.include for:
  ['app/**/*.{ts,tsx}','components/**/*.ts','features/**/*.ts','services/**/*.ts','shared/**/*.ts','store/**/*.ts','visuals/**/*.ts','live2d/**/*.ts']
- Extend coverage.exclude with:
  ['**/*.md','**/*.css','**/*.html','public/**','assets/**','docs/**','prompts/**','live2d/models/**','.cache/**']
- Keep thresholds unchanged
```

### NPUService Timeout Hardening Test

```
Task: Add timeout fallback test for NPUService
Files to edit:
- features/ai/NPUService.hardening.spec.ts

Goal:
- Mock generateContent to never resolve, advance fake timers to trigger timeouts/backoffs, expect fallback to memory and 3 attempts
```

### SettingsMenu Reset Toast Assertions

```
Task: Align reset toast assertions with implementation
Files to edit:
- components/SettingsMenu.resets.spec.ts

Goal:
- Assert toast messages and durations:
  - "Advisor model reset", 1500
  - "Temperature reset", 1500
  - "Top P reset", 1500
  - "Top K reset", 1500
  - "Thinking level reset", 1500
  - "Recent turns reset", 1500
```
