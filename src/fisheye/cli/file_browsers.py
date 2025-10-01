# src/fisheye/cli/file_browsers.py
"""
File browser utilities for the interactive launcher.

Provides Textual-based file/directory pickers for zarr, video, and config files.
"""

import os
from pathlib import Path
from typing import Optional, Callable

try:
    from textual.app import App, ComposeResult
    from textual.widgets import DirectoryTree, Footer, Header, Static
    from textual.containers import Horizontal
    from textual.reactive import reactive
except ImportError:
    print(" Textual not installed. Install with: pip install textual rich")
    raise


def is_zarr_root(p: str) -> bool:
    """Check if a path is a zarr root directory."""
    p = os.path.expanduser(p)
    if os.path.isdir(p) and p.endswith(".zarr"):
        return True
    return os.path.isfile(os.path.join(p, ".zarray")) or os.path.isfile(os.path.join(p, ".zgroup"))


def is_video_file(p: str) -> bool:
    """Check if a path is a video file."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.mpeg', '.mpg', '.wmv', '.flv', '.webm'}
    return Path(p).suffix.lower() in video_extensions


def is_yaml_file(p: str) -> bool:
    """Check if a path is a YAML file."""
    return Path(p).suffix.lower() in {'.yaml', '.yml'}


class FileBrowserApp(App):
    """Generic file browser with custom filter function."""
    
    CSS = """
    Screen { layout: vertical; }
    #status { height: 3; }
    #bar { height: 3; }
    DirectoryTree { height: 1fr; }
    """
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("escape", "quit", "Quit"),
        ("enter", "toggle", "Open/close"),
        ("right", "open_dir", "Expand"),
        ("left", "close_dir", "Collapse"),
        ("l", "open_dir", "Open"),
        ("h", "close_dir", "Close"),
        ("s", "select", "Select"),
    ]
    
    current_path = reactive("")
    
    def __init__(
        self, 
        start_dir: str, 
        title: str,
        filter_func: Optional[Callable[[str], bool]] = None,
        instructions: str = "Navigate and press S to select"
    ):
        super().__init__()
        self._start_dir = start_dir
        self._title = title
        self._filter_func = filter_func
        self._instructions = instructions
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(
            self._instructions,
            id="status",
        )
        yield DirectoryTree(self._start_dir, id="tree")
        yield Horizontal(
            Static(
                "Navigate: ↑↓   Expand: →/Enter   Collapse: ←   Select: S   Quit: Q/Esc",
                id="bar"
            )
        )
        yield Footer()
    
    def on_mount(self) -> None:
        tree = self.query_one("#tree", DirectoryTree)
        tree.focus()
        tree.can_focus = True
    
    def on_key(self, event) -> None:
        """Track cursor position and update status."""
        tree = self.query_one("#tree", DirectoryTree)
        cursor = tree.cursor_node if hasattr(tree, 'cursor_node') else None
        
        if cursor:
            # Try to get path
            path = None
            if hasattr(cursor, "data") and cursor.data and hasattr(cursor.data, "path"):
                path = str(cursor.data.path)
            elif hasattr(cursor, "label"):
                label = str(cursor.label)
                # Try to construct path from label
                if self.current_path and self.current_path.endswith(label):
                    path = self.current_path
                else:
                    path = label
            
            if path:
                self.current_path = path
                
                # Check if it matches our filter
                if self._filter_func and self._filter_func(path):
                    status = self.query_one("#status", Static)
                    filename = os.path.basename(path)
                    status.update(f"[b green]✓ Found:[/b green] {filename} - Press S to select")
                else:
                    status = self.query_one("#status", Static)
                    status.update(self._instructions)
    
    def action_toggle(self) -> None:
        tree = self.query_one("#tree", DirectoryTree)
        if tree.cursor_node:
            tree.cursor_node.toggle()
    
    def action_open_dir(self) -> None:
        tree = self.query_one("#tree", DirectoryTree)
        if tree.cursor_node:
            if tree.cursor_node.allow_expand and not tree.cursor_node.is_expanded:
                tree.cursor_node.expand()
    
    def action_close_dir(self) -> None:
        tree = self.query_one("#tree", DirectoryTree)
        if tree.cursor_node:
            if tree.cursor_node.is_expanded:
                tree.cursor_node.collapse()
    
    def action_select(self) -> None:
        tree = self.query_one("#tree", DirectoryTree)
        node = tree.cursor_node
        if node:
            # Try to get path
            path = None
            if hasattr(node, "data") and node.data and hasattr(node.data, "path"):
                path = str(node.data.path)
            elif self.current_path:
                path = self.current_path
            
            if path and os.path.exists(path):
                # Check if it matches our filter
                if self._filter_func:
                    if self._filter_func(path):
                        self.exit(path)
                        return
                    else:
                        status = self.query_one("#status", Static)
                        status.update(f"[b red]⚠ Not a valid selection:[/b red] {os.path.basename(path)}")
                        return
                else:
                    # No filter, accept any path
                    self.exit(path)
                    return
            
            status = self.query_one("#status", Static)
            status.update("[b red]⚠ Could not determine path[/b red]")


def pick_zarr_path(start_dir: str = "~/Desktop") -> Optional[str]:
    """
    Open a TUI to pick a Zarr directory.
    
    Args:
        start_dir: Starting directory for browser
        
    Returns:
        Selected zarr path, or None if cancelled
    """
    start_dir = os.path.expanduser(start_dir)
    app = FileBrowserApp(
        start_dir=start_dir,
        title="Select Zarr Directory",
        filter_func=is_zarr_root,
        instructions="↑↓ Navigate | →/Enter Expand | ← Collapse | S Select Zarr | Q/Esc Quit"
    )
    
    try:
        return app.run()
    except KeyboardInterrupt:
        return None


def pick_video_path(start_dir: str = "~/Desktop") -> Optional[str]:
    """
    Open a TUI to pick a video file.
    
    Args:
        start_dir: Starting directory for browser
        
    Returns:
        Selected video path, or None if cancelled
    """
    start_dir = os.path.expanduser(start_dir)
    app = FileBrowserApp(
        start_dir=start_dir,
        title="Select Video File",
        filter_func=is_video_file,
        instructions="↑↓ Navigate | →/Enter Expand | ← Collapse | S Select Video | Q/Esc Quit"
    )
    
    try:
        return app.run()
    except KeyboardInterrupt:
        return None


def pick_config_path(start_dir: str = "~/Desktop") -> Optional[str]:
    """
    Open a TUI to pick a config YAML file.
    
    Args:
        start_dir: Starting directory for browser
        
    Returns:
        Selected config path, or None if cancelled
    """
    start_dir = os.path.expanduser(start_dir)
    app = FileBrowserApp(
        start_dir=start_dir,
        title="Select Config File",
        filter_func=is_yaml_file,
        instructions="↑↓ Navigate | →/Enter Expand | ← Collapse | S Select YAML | Q/Esc Quit"
    )
    
    try:
        return app.run()
    except KeyboardInterrupt:
        return None


# Convenience function for testing
if __name__ == "__main__":
    import sys
    
    print("Testing file browsers...\n")
    
    print("1. Pick a Zarr directory:")
    zarr = pick_zarr_path()
    print(f"   Selected: {zarr}\n")
    
    print("2. Pick a video file:")
    video = pick_video_path()
    print(f"   Selected: {video}\n")
    
    print("3. Pick a config file:")
    config = pick_config_path()
    print(f"   Selected: {config}\n")