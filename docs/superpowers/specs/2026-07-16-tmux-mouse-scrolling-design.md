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

Source the configuration into the currently running default tmux server so mouse scrolling becomes available immediately in its live sessions, including `data_download`, `r3d-place-shoe-0000`, and `tactile_vae`. Then query tmux for the effective `mouse` and `history-limit` global options.

Mouse scrolling takes effect in existing panes without restarting them. The larger history limit applies to panes created after the reload; existing panes keep the limit they had when created. No session or pane will be restarted merely to enlarge its history.

## Scope

This setup changes only the user's home-directory tmux configuration. It does not alter project runtime code, key bindings, terminal configuration, or system-wide cluster settings.
