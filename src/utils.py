"""
Utility Functions for Curriculum Mapping Tool

This module provides common utility functions, configuration management,
validation helpers, and data transformation utilities used across the
curriculum mapping application.

Key Features:
- Configuration loading and validation
- Data validation and sanitization utilities
- File handling helpers
- Progress tracking and logging utilities
- Error handling and recovery functions
- Performance monitoring utilities
"""

import os
import json
import yaml
import logging
import re
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import pandas as pd
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]


class ConfigManager:
    """Configuration management utility."""
    
    def __init__(self, config_path: str = "data/config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logger()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            return self._get_default_config()
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration when file is not found."""
        return {
            'api': {
                'anthropic': {
                    'model': 'claude-3-sonnet-20240229',
                    'max_tokens': 4000,
                    'temperature': 0.1,
                    'rate_limit_delay': 1.0
                }
            },
            'data_processing': {
                'min_description_length': 50,
                'max_description_length': 5000,
                'quality_threshold': 70.0
            },
            'paths': {
                'data_dir': 'data',
                'output_dir': 'outputs',
                'template_dir': 'templates'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger based on configuration."""
        logger = logging.getLogger(__name__)
        
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        logger.setLevel(level)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(log_config.get('format', 
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler if enabled
            if log_config.get('file_logging', False):
                log_dir = self.get_path('log_dir', 'logs')
                os.makedirs(log_dir, exist_ok=True)
                
                log_file = os.path.join(log_dir, f"curriculum_mapper_{datetime.now().strftime('%Y%m%d')}.log")
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        return logger
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_path(self, path_key: str, default: str = "") -> str:
        """
        Get file path from configuration, ensuring it exists.
        
        Args:
            path_key: Key for path in paths section
            default: Default path if not found
            
        Returns:
            Absolute path string
        """
        path = self.get(f'paths.{path_key}', default)
        return os.path.abspath(path)
    
    def validate_config(self) -> ValidationResult:
        """
        Validate configuration completeness and correctness.
        
        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Check required sections
        required_sections = ['api', 'data_processing', 'paths']
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required configuration section: {section}")
        
        # Validate API configuration
        api_config = self.config.get('api', {})
        if 'anthropic' not in api_config:
            errors.append("Missing Anthropic API configuration")
        
        # Validate paths
        paths_config = self.config.get('paths', {})
        for path_key in ['data_dir', 'output_dir']:
            if path_key not in paths_config:
                warnings.append(f"Missing path configuration: {path_key}")
        
        # Check data processing thresholds
        data_config = self.config.get('data_processing', {})
        min_len = data_config.get('min_description_length', 0)
        max_len = data_config.get('max_description_length', 0)
        
        if min_len >= max_len:
            errors.append("min_description_length must be less than max_description_length")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )


class DataValidator:
    """Data validation utilities."""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize data validator.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.logger = config_manager.logger
    
    def validate_course_id(self, course_id: str) -> ValidationResult:
        """
        Validate course ID format.
        
        Args:
            course_id: Course identifier to validate
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        suggestions = []
        
        if not course_id or not isinstance(course_id, str):
            errors.append("Course ID must be a non-empty string")
            return ValidationResult(False, errors, warnings, suggestions)
        
        # Check pattern if configured
        pattern = self.config.get('validation.course_id_pattern', r'^[A-Z0-9-]+$')
        if not re.match(pattern, course_id):
            errors.append(f"Course ID '{course_id}' does not match required pattern")
            suggestions.append("Course IDs should contain only uppercase letters, numbers, and hyphens")
        
        # Check length
        if len(course_id) < 3:
            warnings.append("Course ID is very short")
        elif len(course_id) > 20:
            warnings.append("Course ID is very long")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def validate_course_title(self, title: str) -> ValidationResult:
        """
        Validate course title.
        
        Args:
            title: Course title to validate
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        suggestions = []
        
        if not title or not isinstance(title, str):
            errors.append("Course title must be a non-empty string")
            return ValidationResult(False, errors, warnings, suggestions)
        
        title = title.strip()
        
        # Check length requirements
        min_len = self.config.get('validation.title_min_length', 5)
        max_len = self.config.get('validation.title_max_length', 200)
        
        if len(title) < min_len:
            errors.append(f"Course title too short (minimum {min_len} characters)")
        elif len(title) > max_len:
            errors.append(f"Course title too long (maximum {max_len} characters)")
        
        # Check for meaningful content
        if title.lower() in ['tbd', 'to be determined', 'n/a', 'none', 'untitled']:
            errors.append("Course title contains placeholder text")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def validate_credits(self, credits: Union[int, float, str]) -> ValidationResult:
        """
        Validate course credits.
        
        Args:
            credits: Credits value to validate
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        suggestions = []
        
        try:
            credits_num = float(credits)
        except (ValueError, TypeError):
            errors.append("Credits must be a numeric value")
            return ValidationResult(False, errors, warnings, suggestions)
        
        # Check range
        min_credits, max_credits = self.config.get('validation.credits_range', [1, 20])
        
        if credits_num < min_credits:
            errors.append(f"Credits too low (minimum {min_credits})")
        elif credits_num > max_credits:
            errors.append(f"Credits too high (maximum {max_credits})")
        
        # Check for reasonable increments
        if credits_num != int(credits_num) and credits_num * 2 != int(credits_num * 2):
            warnings.append("Unusual credit increment (not whole or half credits)")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text input for processing.
        
        Args:
            text: Raw text to sanitize
            
        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove potentially problematic characters
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/]', ' ', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'[''']', "'", text)
        
        return text


class FileManager:
    """File handling utilities."""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize file manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.logger = config_manager.logger
    
    def ensure_directory(self, path: str) -> str:
        """
        Ensure directory exists, creating if necessary.
        
        Args:
            path: Directory path
            
        Returns:
            Absolute path to directory
        """
        abs_path = os.path.abspath(path)
        os.makedirs(abs_path, exist_ok=True)
        return abs_path
    
    def safe_filename(self, filename: str) -> str:
        """
        Create safe filename by removing problematic characters.
        
        Args:
            filename: Original filename
            
        Returns:
            Safe filename
        """
        # Remove dangerous characters
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Limit length
        if len(safe_name) > 200:
            name, ext = os.path.splitext(safe_name)
            safe_name = name[:200-len(ext)] + ext
        
        return safe_name
    
    def get_unique_filename(self, directory: str, base_name: str, extension: str = "") -> str:
        """
        Generate unique filename in directory.
        
        Args:
            directory: Target directory
            base_name: Base filename
            extension: File extension (with or without dot)
            
        Returns:
            Unique filename path
        """
        if not extension.startswith('.') and extension:
            extension = '.' + extension
        
        base_path = os.path.join(directory, base_name + extension)
        
        if not os.path.exists(base_path):
            return base_path
        
        counter = 1
        while True:
            new_name = f"{base_name}_{counter}{extension}"
            new_path = os.path.join(directory, new_name)
            
            if not os.path.exists(new_path):
                return new_path
            
            counter += 1
    
    def backup_file(self, file_path: str) -> Optional[str]:
        """
        Create backup of existing file.
        
        Args:
            file_path: Path to file to backup
            
        Returns:
            Path to backup file or None if backup failed
        """
        if not os.path.exists(file_path):
            return None
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{file_path}.backup_{timestamp}"
            
            import shutil
            shutil.copy2(file_path, backup_path)
            
            self.logger.info(f"Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to create backup of {file_path}: {e}")
            return None


class ProgressTracker:
    """Progress tracking utility."""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total_items: Total number of items to process
            description: Description of the operation
        """
        self.total_items = total_items
        self.description = description
        self.completed_items = 0
        self.start_time = time.time()
        self.last_update = self.start_time
        
    def update(self, increment: int = 1) -> None:
        """
        Update progress counter.
        
        Args:
            increment: Number of items completed
        """
        self.completed_items += increment
        current_time = time.time()
        
        # Update every second or on completion
        if current_time - self.last_update >= 1.0 or self.completed_items >= self.total_items:
            self._print_progress()
            self.last_update = current_time
    
    def _print_progress(self) -> None:
        """Print current progress status."""
        if self.total_items <= 0:
            return
        
        percentage = (self.completed_items / self.total_items) * 100
        elapsed_time = time.time() - self.start_time
        
        if self.completed_items > 0:
            estimated_total = elapsed_time * self.total_items / self.completed_items
            remaining_time = estimated_total - elapsed_time
            
            print(f"{self.description}: {self.completed_items}/{self.total_items} "
                  f"({percentage:.1f}%) - ETA: {self._format_time(remaining_time)}")
        else:
            print(f"{self.description}: {self.completed_items}/{self.total_items} "
                  f"({percentage:.1f}%)")
    
    def _format_time(self, seconds: float) -> str:
        """Format time duration as human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def finish(self) -> None:
        """Mark progress as complete and print final status."""
        self.completed_items = self.total_items
        elapsed_time = time.time() - self.start_time
        print(f"{self.description}: Completed {self.total_items} items in {self._format_time(elapsed_time)}")


class PerformanceMonitor:
    """Performance monitoring utility."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str) -> None:
        """
        Start timing an operation.
        
        Args:
            operation: Name of the operation to time
        """
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """
        End timing an operation and record duration.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Duration in seconds
        """
        if operation not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append(duration)
        del self.start_times[operation]
        
        return duration
    
    def get_statistics(self, operation: str) -> Dict[str, float]:
        """
        Get performance statistics for an operation.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Dictionary with performance statistics
        """
        if operation not in self.metrics or not self.metrics[operation]:
            return {}
        
        durations = self.metrics[operation]
        
        return {
            'count': len(durations),
            'total_time': sum(durations),
            'average_time': sum(durations) / len(durations),
            'min_time': min(durations),
            'max_time': max(durations)
        }
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance summary for all operations.
        
        Returns:
            Dictionary with statistics for all operations
        """
        return {op: self.get_statistics(op) for op in self.metrics.keys()}


def create_data_hash(data: Union[str, Dict, List]) -> str:
    """
    Create hash of data for caching/comparison purposes.
    
    Args:
        data: Data to hash
        
    Returns:
        Hexadecimal hash string
    """
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, sort_keys=True)
    else:
        data_str = str(data)
    
    return hashlib.md5(data_str.encode('utf-8')).hexdigest()


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format percentage value for display.
    
    Args:
        value: Percentage value (0-100)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimals}f}%"


def format_score(score: float, scale: int = 10) -> str:
    """
    Format score for display.
    
    Args:
        score: Score value
        scale: Maximum score value
        
    Returns:
        Formatted score string
    """
    return f"{score:.1f}/{scale}"


if __name__ == "__main__":
    # Example usage and testing
    
    # Test configuration manager
    config = ConfigManager()
    print("Configuration loaded successfully")
    
    # Test validation
    validation_result = config.validate_config()
    print(f"Configuration valid: {validation_result.is_valid}")
    
    if validation_result.warnings:
        print("Warnings:")
        for warning in validation_result.warnings:
            print(f"  - {warning}")
    
    # Test data validator
    validator = DataValidator(config)
    
    # Test course ID validation
    id_result = validator.validate_course_id("CS-101")
    print(f"Course ID 'CS-101' valid: {id_result.is_valid}")
    
    # Test performance monitor
    monitor = PerformanceMonitor()
    monitor.start_timer("test_operation")
    time.sleep(0.1)  # Simulate work
    duration = monitor.end_timer("test_operation")
    print(f"Test operation took {duration:.3f} seconds")
    
    # Test progress tracker
    tracker = ProgressTracker(5, "Test progress")
    for i in range(5):
        time.sleep(0.1)
        tracker.update()
    tracker.finish()