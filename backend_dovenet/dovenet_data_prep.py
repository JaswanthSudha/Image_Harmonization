import os
import shutil
import uuid
from pathlib import Path

class DoveNetDataPreparer:
    """
    Prepares data in the format expected by DoveNet for harmonization
    """
    
    def __init__(self, workspace_dir="./dovenet_workspace"):
        """
        Initialize data preparer
        
        Args:
            workspace_dir: Directory to create DoveNet workspace
        """
        self.workspace_dir = Path(workspace_dir)
        self.dataset_root = self.workspace_dir / "custom_data"
        self.composite_dir = self.dataset_root / "composite_images"
        self.masks_dir = self.dataset_root / "masks"
        self.results_dir = self.dataset_root / "results"
        
    def setup_workspace(self):
        """Create required directory structure"""
        # Create directories
        self.composite_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Created DoveNet workspace at: {self.workspace_dir}")
        
    def prepare_data(self, composite_path, mask_path):
        """
        Prepare single image pair for DoveNet processing
        
        Args:
            composite_path: Path to composite image from Nuke
            mask_path: Path to mask image from Nuke
            
        Returns:
            dict: Prepared file paths and metadata
        """
        # Generate unique identifier for this harmonization
        unique_id = str(uuid.uuid4())[:8]
        
        # DoveNet expected naming convention
        composite_name = f"sample_{unique_id}_2.jpg"  # _2 indicates composite
        mask_name = f"sample_{unique_id}.png"          # base name for mask
        
        # Copy files to DoveNet structure
        composite_dest = self.composite_dir / composite_name
        mask_dest = self.masks_dir / mask_name
        
        # Copy files (convert format if needed)
        shutil.copy2(composite_path, composite_dest)
        shutil.copy2(mask_path, mask_dest)
        
        # Create IHD_test.txt file
        test_file = self.dataset_root / "IHD_test.txt"
        with open(test_file, 'w') as f:
            f.write(f"composite_images/{composite_name}\n")
        
        return {
            "unique_id": unique_id,
            "dataset_root": str(self.dataset_root),
            "composite_file": str(composite_dest),
            "mask_file": str(mask_dest),
            "test_file": str(test_file),
            "expected_output": str(self.results_dir / f"sample_{unique_id}_2_harmonized.jpg")
        }
    
    def cleanup_workspace(self):
        """Clean up temporary files"""
        if self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir)
            print(f"Cleaned up workspace: {self.workspace_dir}")

def prepare_dovenet_data(composite_path, mask_path, workspace_dir="./dovenet_workspace"):
    """
    Convenience function to prepare data for DoveNet
    
    Args:
        composite_path: Path to composite image
        mask_path: Path to mask image 
        workspace_dir: Workspace directory
        
    Returns:
        dict: Prepared data information
    """
    preparer = DoveNetDataPreparer(workspace_dir)
    preparer.setup_workspace()
    return preparer.prepare_data(composite_path, mask_path)