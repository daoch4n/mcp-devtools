# Scope

This document defines the Senior Agent's v3 behavior within the Windsurf framework. It extends [Senior Agent v2](.windsurf/rules/senior-agent-v2.md) and complements the [Senior Architect SOP](.windsurf/rules/senior-architect-v0.md) and the [AI Edit SOP](.windsurf/rules/ai-edit-sop-v1.md) by providing agent-level operational guidelines.

## See Also

- [Senior Agent Role and Directive (v2)](.windsurf/rules/senior-agent-v2.md)
- [Senior Architect Role and Directive (v0)](.windsurf/rules/senior-architect-v0.md)
- [AI Edit SOP and Templates (v1)](.windsurf/rules/ai-edit-sop-v1.md)

This v3 rule inherits all directives from Senior Agent v2. Only the following additions/overrides apply; if unspecified, follow v2 verbatim.

## Additions & Overrides

- **Library Research**: Researching external dependencies is mandatory. Follow the "Library and API Research" guidelines in the Senior Agent v2 document. Your primary tool is `context7`, not the browser.
- **Planning**: For complex planning, the `sequentialthinking` tool from the `sequentialthinking` server is the preferred and mandatory tool. Use it to structure thoughts, reflect between steps, and ensure a robust plan before execution.
- **Todo List**: Use the `update_todo_list` tool to manage your plan. Emoji status markers are allowed for readability.
- **Test Discipline**: Follow v2 Test Discipline. In research-heavy work, ensure tests remain deterministic (fake timers, mocks), maintain baseline coverage, and raise opportunistically—avoid brittle or low-value tests.
- **Code Hygiene**: Follow v2 Code Hygiene; always lint and type locally (`npm run lint`, `npm run type`); Husky is a safety net only, not the primary enforcement.
- **Commit Discipline**: Follow v2 and [.windsurf/rules/llm-tagger-v0.md](.windsurf/rules/llm-tagger-v0.md). Stage and commit each step with the required trailer. Prefer `git_stage_and_commit` from the `devtools` server.
- **Core Workflow**: For making code changes, debugging, codebase navigation, and running commands, follow the directives in Senior Agent v2 and the referenced SOPs (Senior Architect and AI Edit). These sections are not duplicated here.
