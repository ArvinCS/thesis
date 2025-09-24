#!/usr/bin/env python3
"""
Learning Configuration for Huffman Optimization

This module provides easy configuration switching between different learning modes:
1. Daily learning (rebuild once per day)
2. Immediate learning (rebuild after every verification)
3. Batch learning (rebuild after N verifications)
4. Hybrid learning (immediate for small changes, batched for larger ones)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional

class LearningMode(Enum):
    """Different learning modes for tree optimization."""
    DAILY = "daily"           # Rebuild once per day (current implementation)
    IMMEDIATE = "immediate"   # Rebuild after every verification
    BATCH = "batch"          # Rebuild after N verifications  
    HYBRID = "hybrid"        # Smart combination of immediate + batch
    DISABLED = "disabled"    # No learning (static tree)

@dataclass
class LearningConfig:
    """Configuration for learning behavior."""
    mode: LearningMode = LearningMode.DAILY
    
    # Batch learning settings
    batch_size: int = 10           # Rebuild after N verifications (for BATCH mode)
    
    # Hybrid learning settings  
    immediate_threshold: int = 5   # Use immediate learning for events with < N properties
    batch_threshold: int = 20      # Use batch learning for events with >= N properties
    
    # Performance settings
    min_improvement_threshold: float = 0.01  # Only rebuild if expected >1% improvement
    max_rebuilds_per_day: int = 100         # Limit rebuilds to prevent performance issues
    
    # Debugging
    verbose_logging: bool = False   # Enable detailed learning logs
    track_performance: bool = True  # Track learning effectiveness

# Global configuration - easy to switch
LEARNING_CONFIG = LearningConfig(
    mode=LearningMode.DAILY,        # 🔧 SWITCH THIS TO CHANGE BEHAVIOR
    batch_size=10,
    verbose_logging=True,
    track_performance=True
)

def set_learning_mode(mode: LearningMode, **kwargs):
    """Convenient function to change learning mode."""
    global LEARNING_CONFIG
    LEARNING_CONFIG.mode = mode
    for key, value in kwargs.items():
        if hasattr(LEARNING_CONFIG, key):
            setattr(LEARNING_CONFIG, key, value)
    print(f"🔧 Learning mode switched to: {mode.value}")

def get_learning_config() -> LearningConfig:
    """Get current learning configuration."""
    return LEARNING_CONFIG

# Convenience functions for common configurations
def enable_immediate_learning():
    """Switch to immediate learning after every verification."""
    set_learning_mode(LearningMode.IMMEDIATE, verbose_logging=True)

def enable_batch_learning(batch_size: int = 10):
    """Switch to batch learning after N verifications."""
    set_learning_mode(LearningMode.BATCH, batch_size=batch_size, verbose_logging=True)

def enable_daily_learning():
    """Switch to daily learning (current behavior)."""
    set_learning_mode(LearningMode.DAILY, verbose_logging=True)

def disable_learning():
    """Disable learning completely (static tree)."""
    set_learning_mode(LearningMode.DISABLED, verbose_logging=False)

def reset_to_default_config():
    """Reset configuration to default values."""
    global LEARNING_CONFIG
    LEARNING_CONFIG = LearningConfig()

if __name__ == "__main__":
    # Test different configurations
    print("🧪 Testing learning configurations:")
    print(f"Default: {get_learning_config()}")
    
    enable_immediate_learning()
    print(f"Immediate: {get_learning_config()}")
    
    enable_batch_learning(batch_size=5)
    print(f"Batch: {get_learning_config()}")
    
    enable_daily_learning()
    print(f"Daily: {get_learning_config()}")
