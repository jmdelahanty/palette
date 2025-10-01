# src/fisheye/tune/base.py
"""
Base class for all interactive parameter tuners.

Provides common functionality for OpenCV-based interactive tuning tools.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import zarr
import cv2
import numpy as np


class BaseTuner(ABC):
    """
    Abstract base class for interactive parameter tuners.
    
    All tuners should inherit from this class and implement the required methods.
    """
    
    def __init__(
        self, 
        zarr_path: str, 
        config_path: Optional[str] = None,
        frame_idx: Optional[int] = None
    ):
        """
        Initialize the tuner.
        
        Args:
            zarr_path: Path to the Zarr file
            config_path: Path to YAML config file (default: src/pipeline_config.yaml)
            frame_idx: Specific frame index to use (None for automatic selection)
        """
        self.zarr_path = Path(zarr_path)
        self.config_path = Path(config_path) if config_path else Path("src/pipeline_config.yaml")
        self.frame_idx = frame_idx
        
        # Will be set by subclasses
        self.window_name = "Parameter Tuner"
        self.zarr_root = None
        self.config = {}
        
        # Load config if it exists
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        
    def load_zarr(self, mode='r'):
        """Open the zarr file."""
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr file not found: {self.zarr_path}")
        self.zarr_root = zarr.open(str(self.zarr_path), mode=mode)
        return self.zarr_root
    
    @abstractmethod
    def create_trackbars(self):
        """Create OpenCV trackbars for parameters. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame with current parameters and return visualization.
        
        Args:
            frame: Input frame to process
            
        Returns:
            Processed frame for display
        """
        pass
    
    @abstractmethod
    def get_current_params(self) -> Dict[str, Any]:
        """Get current parameter values. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_config_section(self) -> str:
        """
        Get the name of the config section where parameters should be saved.
        
        Returns:
            Config section name (e.g., 'detect', 'crop', 'mask')
        """
        pass
    
    def save_to_yaml(self, params: Dict[str, Any]) -> bool:
        """
        Save parameters to YAML config file.
        
        Args:
            params: Parameters to save
            
        Returns:
            True if successful
        """
        try:
            # Load existing config
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
            else:
                config = {}
            
            # Update the appropriate section
            section = self.get_config_section()
            if section not in config:
                config[section] = {}
            config[section].update(params)
            
            # Save updated config
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            print(f"✅ Parameters saved to {self.config_path}")
            print(f"   Section: {section}")
            for key, value in params.items():
                print(f"   {key}: {value}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving parameters: {e}")
            return False
    
    def save_to_zarr(self, params: Dict[str, Any]) -> bool:
        """
        Save parameters to zarr metadata.
        
        Args:
            params: Parameters to save
            
        Returns:
            True if successful
        """
        try:
            # Open in append mode
            zarr_root = zarr.open(str(self.zarr_path), mode='a')
            
            # Ensure analysis_metadata group exists
            if 'analysis_metadata' not in zarr_root:
                zarr_root.create_group('analysis_metadata')
            
            analysis_meta = zarr_root['analysis_metadata']
            
            # Save to appropriate attribute
            section = self.get_config_section()
            analysis_meta.attrs[section] = params
            
            print(f"✅ Parameters saved to zarr: {self.zarr_path}")
            print(f"   Group: analysis_metadata")
            print(f"   Attribute: {section}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving to zarr: {e}")
            return False
    
    def save_params(self, save_to: str = 'both') -> bool:
        """
        Save current parameters to config and/or zarr.
        
        Args:
            save_to: Where to save ('yaml', 'zarr', or 'both')
            
        Returns:
            True if all requested saves succeeded
        """
        params = self.get_current_params()
        success = True
        
        if save_to in ['yaml', 'both']:
            success &= self.save_to_yaml(params)
        
        if save_to in ['zarr', 'both']:
            success &= self.save_to_zarr(params)
        
        return success
    
    def run(self):
        """
        Run the interactive tuning session.
        
        This is the main loop that displays the OpenCV window and handles user input.
        Subclasses can override this if they need custom behavior.
        """
        # Load zarr
        self.load_zarr()
        
        # Create window and trackbars
        cv2.namedWindow(self.window_name)
        self.create_trackbars()
        
        # Display instructions
        self._print_instructions()
        
        # Main loop
        while True:
            # Get frame (subclass should implement frame selection)
            frame = self._get_frame()
            if frame is None:
                print("❌ Could not load frame")
                break
            
            # Process and display
            display = self.process_frame(frame)
            cv2.imshow(self.window_name, display)
            
            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('s'):  # save
                self.save_params(save_to='both')
            elif key == ord('h'):  # help
                self._print_instructions()
            
            # Let subclasses handle additional keys
            self._handle_key(key)
        
        cv2.destroyAllWindows()
        print("\n✅ Tuner closed.")
    
    def _get_frame(self) -> Optional[np.ndarray]:
        """
        Get the current frame to display. Should be implemented by subclasses.
        
        Returns:
            Frame as numpy array, or None if unavailable
        """
        # Default implementation - subclasses should override
        return None
    
    def _handle_key(self, key: int):
        """
        Handle additional keyboard input. Can be overridden by subclasses.
        
        Args:
            key: Key code from cv2.waitKey
        """
        pass
    
    def _print_instructions(self):
        """Print instructions for using the tuner."""
        print("\n🎛️  Parameter Tuner Controls:")
        print("  - Adjust sliders to tune parameters")
        print("  - Press 's' to SAVE parameters")
        print("  - Press 'h' to show this help")
        print("  - Press 'q' or ESC to quit")
        print()