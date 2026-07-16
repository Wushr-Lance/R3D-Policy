# tmux Mouse Scrolling Design

## Goal

Enable mouse-wheel scrolling through tmux pane history on this cluster while retaining enough output for long-running jobs.

## Configuration

Create `~/.tmux.conf` with:

```tmux
set -g mouse on
set -g history-limit 50000
```

The mouse option lets the wheel enter and navigate tmux copy mode. The history limit retains up to 50,000 lines per pane without adding custom mouse bindings.

## Activation and verification

Reload the configuration for a running tmux server, if one exists. Then query tmux for the effective `mouse` and `history-limit` global options. Existing panes may need to be recreated for the larger history limit to take full effect.

## Scope

This setup changes only the user's home-directory tmux configuration. It does not alter project runtime code, key bindings, terminal configuration, or system-wide cluster settings.
