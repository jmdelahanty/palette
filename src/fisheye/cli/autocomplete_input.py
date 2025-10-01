# src/fisheye/cli/autocomplete_input.py
"""
Custom Input widget with filesystem path autocomplete.

Provides Ctrl+Space completion for file and directory paths.
"""

import os
from pathlib import Path
from typing import Optional, Callable

try:
    from textual.widgets import Input
    from textual.suggester import Suggester
    from textual import events
except ImportError:
    print("⚠️ Textual not installed. Install with: pip install textual")
    raise


class PathSuggester(Suggester):
    """Suggests filesystem paths based on current input."""
    
    def __init__(self, filter_func: Optional[Callable[[str], bool]] = None):
        """
        Initialize the path suggester.
        
        Args:
            filter_func: Optional function to filter suggestions (e.g., only .zarr files)
        """
        super().__init__()
        self.filter_func = filter_func
    
    async def get_suggestion(self, value: str) -> Optional[str]:
        """
        Get path suggestion based on current input.
        
        Args:
            value: Current input value
            
        Returns:
            Suggested completion or None
        """
        if not value:
            return None
        
        # Expand ~ to home directory
        expanded = os.path.expanduser(value)
        
        # Get the directory and partial filename
        if os.path.isdir(expanded):
            # If it's already a complete directory, look inside it
            directory = expanded
            partial = ""
        else:
            # Split into directory and partial filename
            directory = os.path.dirname(expanded) or "."
            partial = os.path.basename(expanded)
        
        # Get suggestions from directory
        try:
            if not os.path.exists(directory):
                return None
            
            items = os.listdir(directory)
            
            # Filter by partial match (case-insensitive)
            if partial:
                matches = [item for item in items if item.lower().startswith(partial.lower())]
            else:
                matches = items
            
            # Apply custom filter if provided
            if self.filter_func:
                matches = [
                    item for item in matches 
                    if self.filter_func(os.path.join(directory, item))
                ]
            
            # Sort: directories first, then files
            def sort_key(item):
                full_path = os.path.join(directory, item)
                is_dir = os.path.isdir(full_path)
                return (not is_dir, item.lower())
            
            matches.sort(key=sort_key)
            
            if matches:
                # Return the first match
                first_match = matches[0]
                
                # Build full path
                if directory == ".":
                    suggestion = first_match
                else:
                    suggestion = os.path.join(directory, first_match)
                
                # Add trailing slash for directories
                if os.path.isdir(suggestion):
                    suggestion += os.sep
                
                # Convert back if we expanded ~
                if value.startswith("~"):
                    home = os.path.expanduser("~")
                    if suggestion.startswith(home):
                        suggestion = "~" + suggestion[len(home):]
                
                return suggestion
            
        except (PermissionError, OSError):
            return None
        
        return None


class PathInput(Input):
    """Input widget with filesystem path autocomplete."""
    
    BINDINGS = [
        ("ctrl+space", "complete_path", "Complete path"),
    ]
    
    def __init__(
        self,
        value: str = "",
        placeholder: str = "",
        filter_func: Optional[Callable[[str], bool]] = None,
        **kwargs
    ):
        """
        Initialize path input with autocomplete.
        
        Args:
            value: Initial value
            placeholder: Placeholder text
            filter_func: Optional function to filter suggestions
            **kwargs: Additional Input arguments
        """
        super().__init__(
            value=value,
            placeholder=placeholder,
            **kwargs
        )
        self.path_suggester = PathSuggester(filter_func=filter_func)
        self._last_suggestion = None
    
    def action_complete_path(self) -> None:
        """Complete the path (triggered by Ctrl+Space)."""
        if not self.value:
            self.app.notify("Type a path first!", severity="warning")
            return
        
        # Run completion in a worker
        self.run_worker(self._do_completion(), exclusive=True)
    
    async def _do_completion(self) -> None:
        """Perform the path completion."""
        current = self.value
        suggestion = await self.path_suggester.get_suggestion(current)
        
        if suggestion and suggestion != current:
            self.value = suggestion
            self.cursor_position = len(suggestion)
            self.app.notify(f"Completed: {suggestion}", severity="information", timeout=2)
        elif suggestion == current:
            self.app.notify("Already at suggestion", severity="information", timeout=1)
        else:
            self.app.notify("No completion found", severity="warning", timeout=1)


def is_zarr_dir(path: str) -> bool:
    """Filter for zarr directories."""
    if not os.path.exists(path):
        return True  # Allow incomplete paths
    if os.path.isdir(path):
        return (
            path.endswith(".zarr") or
            os.path.exists(os.path.join(path, ".zarray")) or
            os.path.exists(os.path.join(path, ".zgroup"))
        )
    return False


def is_video_file(path: str) -> bool:
    """Filter for video files."""
    if not os.path.exists(path):
        return True  # Allow incomplete paths
    if os.path.isfile(path):
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.mpeg', '.mpg', '.wmv', '.flv', '.webm'}
        return Path(path).suffix.lower() in video_extensions
    return os.path.isdir(path)  # Allow navigating through directories


def is_yaml_file(path: str) -> bool:
    """Filter for YAML config files."""
    if not os.path.exists(path):
        return True  # Allow incomplete paths
    if os.path.isfile(path):
        return Path(path).suffix.lower() in {'.yaml', '.yml'}
    return os.path.isdir(path)  # Allow navigating through directories


# Example usage and testing
if __name__ == "__main__":
    from textual.app import App, ComposeResult
    from textual.widgets import Header, Footer, Label
    from textual.containers import Vertical
    
    class PathInputTest(App):
        """Test app for PathInput widget."""
        
        CSS = """
        Screen {
            layout: vertical;
        }
        
        Vertical {
            height: auto;
            padding: 1;
        }
        
        PathInput {
            margin: 1 0;
        }
        """
        
        def compose(self) -> ComposeResult:
            yield Header()
            
            with Vertical():
                yield Label("Test Path Autocomplete Input")
                yield Label("Type a path and press Ctrl+Space for suggestions:")
                
                yield Label("\n1. Any path:")
                yield PathInput(
                    placeholder="Type any path...",
                    id="any_path"
                )
                
                yield Label("\n2. Zarr files only:")
                yield PathInput(
                    placeholder="Type zarr path...",
                    filter_func=is_zarr_dir,
                    id="zarr_path"
                )
                
                yield Label("\n3. Video files only:")
                yield PathInput(
                    placeholder="Type video path...",
                    filter_func=is_video_file,
                    id="video_path"
                )
                
                yield Label("\n4. YAML files only:")
                yield PathInput(
                    placeholder="Type config path...",
                    filter_func=is_yaml_file,
                    id="yaml_path"
                )
            
            yield Footer()
    
    app = PathInputTest()
    app.run()