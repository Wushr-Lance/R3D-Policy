# tmux Mouse Scrolling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable mouse-wheel scrolling immediately in the live tmux server, including the `r3d-place-shoe-0000` session, and retain 50,000 lines in panes created afterward.

**Architecture:** Store the persistent global options in `/home/shenrui/.tmux.conf`, then source that file into the already-running default tmux server. Verify both the server-wide option values and the continued existence of all live sessions without restarting panes.

**Tech Stack:** tmux 3.4 configuration and command-line interface

## Global Constraints

- Change only `/home/shenrui/.tmux.conf`; do not alter project runtime code or system-wide configuration.
- Set `mouse` to `on` and `history-limit` to exactly `50000`.
- Apply mouse support to current sessions without restarting any session or pane.
- Accept that the larger history limit applies only to panes created after the reload.

---

### Task 1: Persistent and live tmux configuration

**Files:**
- Create: `/home/shenrui/.tmux.conf`
- Test: live default tmux server options and session list

**Interfaces:**
- Consumes: the default tmux server socket containing `data_download`, `r3d-place-shoe-0000`, and `tactile_vae`
- Produces: persistent global tmux options and an immediately updated live server

- [ ] **Step 1: Capture the failing precondition**

Run:

```bash
tmux show-options -gv mouse
tmux show-options -gv history-limit
```

Expected before configuration:

```text
off
2000
```

- [ ] **Step 2: Create the persistent configuration**

Create `/home/shenrui/.tmux.conf` with exactly:

```tmux
set -g mouse on
set -g history-limit 50000
```

- [ ] **Step 3: Apply it to the running server**

Run:

```bash
tmux source-file /home/shenrui/.tmux.conf
```

Expected: exit status 0 with no output. This updates the live global options and does not restart sessions or panes.

- [ ] **Step 4: Verify the effective options**

Run:

```bash
tmux show-options -gv mouse
tmux show-options -gv history-limit
tmux show-options -Av -t r3d-place-shoe-0000 mouse
```

Expected:

```text
on
50000
on
```

- [ ] **Step 5: Verify that live sessions remain available**

Run:

```bash
tmux list-sessions -F '#{session_name}'
```

Expected output includes all of:

```text
data_download
r3d-place-shoe-0000
tactile_vae
```
