"""
Report Organizer for Hierarchical Merkle Tree Research
Automatically organizes all result files into timestamped report folders.
"""

import os
import time
import shutil
from datetime import datetime
from pathlib import Path

class ReportOrganizer:
    """Organizes all research output files into timestamped report directories."""
    
    def __init__(self, base_dir=None):
        self.timestamp = int(time.time())
        self.datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{self.datetime_str}_{self.timestamp}"
        
        # Create main report directory structure
        self.base_report_dir = Path(base_dir) if base_dir else Path("report")
        self.current_run_dir = self.base_report_dir / self.run_id
        
        # Create subdirectories
        self.subdirs = {
            "verification_results": self.current_run_dir / "verification_results",
            "benchmark_results": self.current_run_dir / "benchmark_results", 
            "generated_data": self.current_run_dir / "generated_data",
            "performance_reports": self.current_run_dir / "performance_reports",
            "analysis": self.current_run_dir / "analysis",
            "logs": self.current_run_dir / "logs"
        }
        
        # Create all directories
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Created report structure: {self.current_run_dir}")
    
    def get_organized_filepath(self, filename, file_type="verification_results"):
        """Get the organized path for a result file."""
        if file_type not in self.subdirs:
            file_type = "verification_results"  # Default
        
        return self.subdirs[file_type] / filename
    
    def get_run_info(self):
        """Get information about the current run."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "datetime": self.datetime_str,
            "base_dir": str(self.current_run_dir)
        }
    
    def save_run_metadata(self, metadata):
        """Save metadata about the current run."""
        metadata_file = self.current_run_dir / "run_metadata.json"
        
        run_info = self.get_run_info()
        run_info.update(metadata)
        
        import json
        with open(metadata_file, 'w') as f:
            json.dump(run_info, f, indent=2)
        
        print(f"💾 Saved run metadata: {metadata_file}")
    
    def organize_existing_files(self):
        """Move existing result files to organized structure."""
        current_dir = Path(".")
        
        # File patterns to organize
        file_patterns = {
            "verification_results": [
                "large_scale_verification_results_*.json",
                "*verification_results*.json"
            ],
            "benchmark_results": [
                "benchmark_results_*.json",
                "*benchmark*.json"
            ],
            "generated_data": [
                "large_scale_documents_*.json",
                "large_scale_traffic_*.json",
                "*documents*.json",
                "*traffic*.json"
            ],
            "performance_reports": [
                "complete_large_scale_report_*.json",
                "*performance*.json",
                "*report*.json"
            ]
        }
        
        moved_files = []
        
        for file_type, patterns in file_patterns.items():
            for pattern in patterns:
                for file_path in current_dir.glob(pattern):
                    if file_path.is_file() and not file_path.name.startswith('.'):
                        target_path = self.subdirs[file_type] / file_path.name
                        try:
                            shutil.move(str(file_path), str(target_path))
                            moved_files.append((str(file_path), str(target_path)))
                            print(f"📦 Moved: {file_path.name} → {file_type}/")
                        except Exception as e:
                            print(f"❌ Error moving {file_path}: {e}")
        
        if moved_files:
            print(f"✅ Organized {len(moved_files)} existing files")
        else:
            print("ℹ️ No existing result files found to organize")
        
        return moved_files

# Global instance for use across scripts
_report_organizer = None

def get_report_organizer(base_dir=None):
    """Get the global report organizer instance."""
    global _report_organizer
    if _report_organizer is None:
        _report_organizer = ReportOrganizer(base_dir)
    return _report_organizer

def get_organized_filepath(filename, file_type="verification_results", base_dir=None):
    """Convenience function to get organized file path."""
    return get_report_organizer(base_dir).get_organized_filepath(filename, file_type)

def save_organized_file(data, filename, file_type="verification_results", base_dir=None):
    """Save data to an organized file path."""
    import json
    
    filepath = get_organized_filepath(filename, file_type, base_dir)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Saved: {file_type}/{filename}")
    return filepath

def organize_all_existing_files():
    """Organize all existing result files."""
    organizer = get_report_organizer()
    return organizer.organize_existing_files()

if __name__ == "__main__":
    # When run directly, organize existing files
    print("🗂️ ORGANIZING EXISTING RESULT FILES")
    print("=" * 50)
    
    organizer = ReportOrganizer()
    moved_files = organizer.organize_existing_files()
    
    # Save organization log
    if moved_files:
        log_data = {
            "organization_timestamp": organizer.timestamp,
            "organization_datetime": organizer.datetime_str,
            "moved_files": moved_files,
            "total_files_moved": len(moved_files)
        }
        
        log_file = organizer.current_run_dir / "logs" / "file_organization.json"
        import json
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"📋 Organization log saved: {log_file}")
    
    print(f"\n✅ File organization completed!")
    print(f"📁 Report directory: {organizer.current_run_dir}")
