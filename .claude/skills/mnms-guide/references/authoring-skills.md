# Authoring your own skill

This is the meta-step. The workshop's thesis is that the durable win is *packaging a
capability so you can invoke it instead of rebuilding it* (see
`the-argument.md`). A skill is exactly that packaging for a coding agent — and the
`mnms-guide` skill you're reading is the worked example. This file teaches you to
make your own by dissecting this one.

## What a skill actually is

A skill is a directory with a `SKILL.md` file and optional bundled resources:

```
.claude/skills/<name>/
  SKILL.md            # required: YAML frontmatter + a short prose body
  references/*.md     # optional: deeper material, loaded only when needed
  scripts/, assets/   # optional: code or files the skill can use
```

`SKILL.md` frontmatter has two fields that matter most:

```yaml
---
name: mnms-guide
description: >-
  One or two sentences saying WHAT the skill covers and WHEN to use it, written
  with the words a user would actually say. This text is how the agent decides to
  reach for the skill, so it is keyword-bearing, not decorative.
---
```

The agent sees every skill's `name` + `description` at all times, but only reads the
body when the description matches the task. So:

- **Put trigger words in the `description`.** For `mnms-guide` that means the branch
  names, "how to run", "what does this mean", "kappa", "Bradley-Terry", "make a
  skill" — the phrasings a researcher uses.
- **Keep the body short.** It's a router, not a manual.

## The one design rule: progressive disclosure

Don't cram everything into `SKILL.md`. Split depth into `references/` files and have
the body point to them, so the agent loads only what the current question needs.
`mnms-guide` does this — `the-argument.md`, `running-branches.md`,
`reading-results.md`, and this file are pulled in individually. That keeps the
always-on context cost tiny.

## The second rule: route, don't duplicate

This skill deliberately does **not** restate `WORKSHOP_ARCHETYPES.md`, the branch
`README.md`s, or the code comments. It points *into* them. Single source of truth:
when the archetype doc changes, the skill doesn't rot. When you write your own,
reference your existing docs instead of copying them — a duplicated skill is a skill
that will silently go stale.

## How to build one from a research method (the workshop move)

You just watched three methods get turned into branches. To turn *your* method into
a skill:

1. **Name the capability** in the user's words. That sentence becomes the
   `description`.
2. **Write the thin body**: what it is, when to use it, and a map to the details.
3. **Move detail into `references/`** — the actual steps, the gotchas, the way to
   read the output. Ground it in real artifacts, as `reading-results.md` does.
4. **Point at existing sources** rather than duplicating them.
5. **Test the trigger**: ask a question the way a real user would and see whether the
   agent reaches for the skill. If not, fix the `description`, not the body.

## Where to put it so every agent finds it

In this repo the skill lives in `.claude/skills/` (Claude Code) and is exposed to
Codex CLI and other agents via `.agent/skills/`, which is a symlink to the same
directory — **one source, two entry points**, which is the same route-don't-duplicate
rule applied to the filesystem. Commit it to the repo so anyone who clones gets it.

That's the whole loop: the method became a branch, and the knowledge of how to run
and read that branch became a skill — packaged, invocable, and shipped with the code.
