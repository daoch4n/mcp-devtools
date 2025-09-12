---
trigger: always_on
---

# Commit Convention

For the name of science, your every step must be committed. The `Co-authored-by: gpt-5-high (planning) && gemini-2.5-pro-low (coding) <189301087+windsurf-bot[bot]@users.noreply.github.com>` line must be a trailer on every commit you make. For proper formatting, either git_stage_and_commit tool must be used, or if you must use shell do it like so: `printf "type(task): update some stuff\n\n- Add/refresh stuff to align with some past/future stuff\n- These changes are part of the big stuff\n\nCo-authored-by: gpt-5-high (planning) && gemini-2.5-pro-low (coding) <189301087+windsurf-bot[bot]@users.noreply.github.com>\n" > .git/COMMIT_MSG && git commit -F .git/COMMIT_MSG`
