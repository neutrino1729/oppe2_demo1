"""
Setup Feast for CI environment
Creates necessary directories and initializes registry
"""

import os
from pathlib import Path

def setup_feast_ci():
    """Prepare Feast for CI environment"""
    
    # Get the feature_repo directory
    feature_repo_dir = Path(__file__).parent
    
    print(f"Setting up Feast in: {feature_repo_dir}")
    
    # Create data directory if it doesn't exist
    data_dir = feature_repo_dir / "data"
    data_dir.mkdir(exist_ok=True)
    print(f"✅ Created data directory: {data_dir}")
    
    # Initialize empty registry if it doesn't exist
    registry_path = feature_repo_dir / "registry.db"
    if not registry_path.exists():
        registry_path.touch()
        print(f"✅ Initialized registry: {registry_path}")
    
    # Initialize empty online store if it doesn't exist
    online_store_path = feature_repo_dir / "online_store.db"
    if not online_store_path.exists():
        online_store_path.touch()
        print(f"✅ Initialized online store: {online_store_path}")
    
    print("✅ Feast CI setup complete!")

if __name__ == "__main__":
    setup_feast_ci()
