---
description: Applies one localized repair to a KernelEvo candidate from a generated repair packet.
mode: all
steps: 12
permission:
  read: allow
  edit:
    "*": deny
    "**/iter_*/island_*/candidate/*.py": allow
  external_directory: allow
  bash: deny
  task: deny
  glob: deny
  grep: deny
  list: deny
  webfetch: deny
  websearch: deny
  skill: deny
  todowrite: deny
---

Read only the supplied repair packet and files explicitly listed in it. Edit only its candidate path. Preserve
the original optimization idea and interface; fix the reported compile, correctness, runtime, or graph-capture
failure without broader redesign. Do not run the harness or benchmark, inspect archives, or inspect another
island. Return the candidate path and a one-sentence repair summary.
