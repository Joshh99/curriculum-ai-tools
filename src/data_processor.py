"""
Data Processing Pipeline for Curriculum Mapping

This module provides comprehensive data cleaning, validation, and preparation
functionality for curriculum data before AI processing. It includes data quality
assessment, standardization, and reporting capabilities.

Key Features:
- Data cleaning and standardization
- Quality validation and scoring
- Missing data handling
- Format standardization
- Data quality reporting
- Batch processing capabilities
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class DataQualityReport:
    """Represents a comprehensive data quality assessment."""
    total_records: int
    valid_records: int
    missing_data_percentage: float
    duplicate_records: int
    quality_score: float
    issues: List[str]
    recommendations: List[str]
    timestamp: str


@dataclass
class ProcessingStats:
    """Statistics from data processing operations."""
    original_count: int
    processed_count: int
    cleaned_count: int
    removed_count: int
    processing_time: float


class DataProcessor:
    """
    Comprehensive data processing pipeline for curriculum data.
    
    This class handles cleaning, validation, and preparation of course data
    for AI-assisted curriculum mapping, ensuring data quality and consistency.
    """
    
    def __init__(self, min_description_length: int = 50, max_description_length: int = 5000):
        """
        Initialize the data processor.
        
        Args:
            min_description_length: Minimum required length for course descriptions
            max_description_length: Maximum allowed length for course descriptions
        """
        self.min_description_length = min_description_length
        self.max_description_length = max_description_length
        self.logger = self._setup_logging()
        
        # Common academic terms and abbreviations
        self.academic_abbreviations = {
            'prereq': 'prerequisite',
            'coreq': 'corequisite',
            'hr': 'hour',
            'hrs': 'hours',
            'cr': 'credit',
            'crs': 'credits',
            'sem': 'semester',
            'yr': 'year',
            'dept': 'department',
            'prog': 'program',
            'req': 'required',
            'opt': 'optional',
            'elec': 'elective'
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def load_data(self, file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Load curriculum data from various file formats.
        
        Args:
            file_path: Path to the data file
            encoding: File encoding (default: utf-8)
            
        Returns:
            DataFrame with loaded data
        """
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, encoding=encoding)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path, encoding=encoding)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            self.logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def validate_data_structure(self, df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that the DataFrame has the required structure.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check for empty DataFrame
        if df.empty:
            issues.append("DataFrame is empty")
        
        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            issues.append("Duplicate column names found")
        
        return len(issues) == 0, issues
    
    def clean_text(self, text: str) -> str:
        """
        Clean and standardize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and standardized text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to string and strip whitespace
        text = str(text).strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        
        # Expand common abbreviations
        for abbrev, full_form in self.academic_abbreviations.items():
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, full_form, text, flags=re.IGNORECASE)
        
        # Capitalize sentences properly
        sentences = text.split('.')
        sentences = [s.strip().capitalize() for s in sentences if s.strip()]
        text = '. '.join(sentences)
        
        return text
    
    def validate_course_description(self, description: str) -> Tuple[bool, List[str]]:
        """
        Validate individual course descriptions for quality and completeness.
        
        Args:
            description: Course description to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not description or pd.isna(description):
            issues.append("Description is empty or missing")
            return False, issues
        
        # Check length requirements
        if len(description) < self.min_description_length:
            issues.append(f"Description too short (< {self.min_description_length} characters)")
        
        if len(description) > self.max_description_length:
            issues.append(f"Description too long (> {self.max_description_length} characters)")
        
        # Check for meaningful content
        word_count = len(description.split())
        if word_count < 10:
            issues.append("Description has too few words (< 10)")
        
        # Check for common issues
        if description.lower().strip() in ['tbd', 'to be determined', 'n/a', 'none']:
            issues.append("Description contains placeholder text")
        
        # Check for academic content indicators
        academic_indicators = [
            'student', 'learn', 'course', 'study', 'understand', 'analyze',
            'develop', 'skill', 'knowledge', 'concept', 'theory', 'practice'
        ]
        
        has_academic_content = any(
            indicator in description.lower() for indicator in academic_indicators
        )
        
        if not has_academic_content:
            issues.append("Description lacks clear academic content indicators")
        
        return len(issues) == 0, issues
    
    def clean_course_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize course data.
        
        Args:
            df: DataFrame with raw course data
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Starting data cleaning process")
        
        # Create a copy to avoid modifying original data
        cleaned_df = df.copy()
        
        # Standardize column names
        cleaned_df.columns = [col.lower().strip().replace(' ', '_') for col in cleaned_df.columns]
        
        # Clean text fields
        text_columns = ['title', 'description', 'department', 'instructor']
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].apply(self.clean_text)
        
        # Standardize course IDs
        if 'id' in cleaned_df.columns:
            cleaned_df['id'] = cleaned_df['id'].astype(str).str.upper().str.strip()
        
        # Handle credits/units
        if 'credits' in cleaned_df.columns:
            cleaned_df['credits'] = pd.to_numeric(cleaned_df['credits'], errors='coerce')
        
        # Remove rows with empty critical fields
        critical_fields = ['id', 'title', 'description']
        for field in critical_fields:
            if field in cleaned_df.columns:
                cleaned_df = cleaned_df[cleaned_df[field].notna()]
                cleaned_df = cleaned_df[cleaned_df[field].str.strip() != '']
        
        self.logger.info(f"Cleaning completed. {len(cleaned_df)} records remaining")
        return cleaned_df
    
    def assess_data_quality(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Perform comprehensive data quality assessment.
        
        Args:
            df: DataFrame to assess
            
        Returns:
            DataQualityReport with detailed quality metrics
        """
        total_records = len(df)
        issues = []
        recommendations = []
        
        # Check for missing data
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data.sum() / (len(df) * len(df.columns))) * 100
        
        if missing_percentage > 10:
            issues.append(f"High missing data rate: {missing_percentage:.1f}%")
            recommendations.append("Consider data imputation or additional data collection")
        
        # Check for duplicates
        if 'id' in df.columns:
            duplicate_count = df['id'].duplicated().sum()
            if duplicate_count > 0:
                issues.append(f"Found {duplicate_count} duplicate course IDs")
                recommendations.append("Remove or consolidate duplicate records")
        else:
            duplicate_count = 0
        
        # Validate course descriptions
        valid_descriptions = 0
        if 'description' in df.columns:
            for desc in df['description']:
                is_valid, _ = self.validate_course_description(desc)
                if is_valid:
                    valid_descriptions += 1
        
        # Calculate quality score
        quality_factors = []
        
        # Missing data factor (0-30 points)
        missing_score = max(0, 30 - missing_percentage)
        quality_factors.append(missing_score)
        
        # Duplicate factor (0-20 points)
        duplicate_score = max(0, 20 - (duplicate_count / total_records * 100))
        quality_factors.append(duplicate_score)
        
        # Description quality factor (0-50 points)
        if total_records > 0:
            desc_quality_score = (valid_descriptions / total_records) * 50
        else:
            desc_quality_score = 0
        quality_factors.append(desc_quality_score)
        
        quality_score = sum(quality_factors)
        
        # Add general recommendations
        if quality_score < 70:
            recommendations.append("Overall data quality is below acceptable threshold")
        
        if missing_percentage > 5:
            recommendations.append("Review data collection processes to reduce missing data")
        
        return DataQualityReport(
            total_records=total_records,
            valid_records=valid_descriptions,
            missing_data_percentage=missing_percentage,
            duplicate_records=duplicate_count,
            quality_score=quality_score,
            issues=issues,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
    
    def prepare_for_ai_processing(self, df: pd.DataFrame) -> Tuple[List[Dict], ProcessingStats]:
        """
        Prepare cleaned data for AI processing.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Tuple of (list of course dictionaries, processing statistics)
        """
        start_time = datetime.now()
        original_count = len(df)
        
        # Ensure required columns exist
        required_columns = ['id', 'title', 'description']
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter valid records
        valid_records = []
        
        for _, row in df.iterrows():
            # Validate description
            is_valid, issues = self.validate_course_description(row['description'])
            
            if is_valid:
                course_dict = {
                    'id': row['id'],
                    'title': row['title'],
                    'description': row['description']
                }
                
                # Add optional fields if available
                optional_fields = ['department', 'credits', 'level', 'instructor']
                for field in optional_fields:
                    if field in df.columns and pd.notna(row[field]):
                        course_dict[field] = row[field]
                
                valid_records.append(course_dict)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        stats = ProcessingStats(
            original_count=original_count,
            processed_count=len(valid_records),
            cleaned_count=len(valid_records),
            removed_count=original_count - len(valid_records),
            processing_time=processing_time
        )
        
        self.logger.info(f"Prepared {len(valid_records)} courses for AI processing")
        return valid_records, stats
    
    def export_quality_report(self, report: DataQualityReport, output_path: str):
        """
        Export data quality report to JSON file.
        
        Args:
            report: DataQualityReport to export
            output_path: Output file path
        """
        report_dict = {
            'data_quality_assessment': {
                'total_records': report.total_records,
                'valid_records': report.valid_records,
                'missing_data_percentage': report.missing_data_percentage,
                'duplicate_records': report.duplicate_records,
                'quality_score': report.quality_score,
                'quality_grade': self._get_quality_grade(report.quality_score),
                'issues': report.issues,
                'recommendations': report.recommendations,
                'assessment_timestamp': report.timestamp
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Quality report exported to {output_path}")
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def process_pipeline(self, input_path: str, output_dir: str) -> Tuple[List[Dict], DataQualityReport]:
        """
        Execute the complete data processing pipeline.
        
        Args:
            input_path: Path to input data file
            output_dir: Directory for output files
            
        Returns:
            Tuple of (processed course data, quality report)
        """
        self.logger.info("Starting complete data processing pipeline")
        
        # Load data
        raw_data = self.load_data(input_path)
        
        # Validate structure
        required_columns = ['id', 'title', 'description']
        is_valid, issues = self.validate_data_structure(raw_data, required_columns)
        
        if not is_valid:
            raise ValueError(f"Data structure validation failed: {issues}")
        
        # Clean data
        cleaned_data = self.clean_course_data(raw_data)
        
        # Assess quality
        quality_report = self.assess_data_quality(cleaned_data)
        
        # Export quality report
        quality_report_path = f"{output_dir}/data_quality_report.json"
        self.export_quality_report(quality_report, quality_report_path)
        
        # Prepare for AI processing
        processed_courses, stats = self.prepare_for_ai_processing(cleaned_data)
        
        # Export processed data
        processed_data_path = f"{output_dir}/processed_courses.json"
        with open(processed_data_path, 'w', encoding='utf-8') as f:
            json.dump(processed_courses, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Pipeline completed. Processed {len(processed_courses)} courses")
        self.logger.info(f"Quality score: {quality_report.quality_score:.1f}/100")
        
        return processed_courses, quality_report


if __name__ == "__main__":
    # Example usage
    processor = DataProcessor()
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'id': ['CS101', 'CS102', 'MATH201'],
        'title': ['Intro to Programming', 'Data Structures', 'Calculus II'],
        'description': [
            'Introduction to programming concepts using Python. Students will learn basic programming constructs, problem-solving techniques, and software development practices.',
            'Study of fundamental data structures including arrays, linked lists, stacks, queues, trees, and graphs. Implementation and analysis of algorithms.',
            'Continuation of calculus including integration techniques, applications of integrals, and infinite series.'
        ],
        'credits': [3, 4, 3],
        'department': ['Computer Science', 'Computer Science', 'Mathematics']
    })
    
    # Process the data
    quality_report = processor.assess_data_quality(sample_data)
    print(f"Data Quality Score: {quality_report.quality_score:.1f}/100")
    print(f"Quality Grade: {processor._get_quality_grade(quality_report.quality_score)}")
    
    if quality_report.issues:
        print("Issues found:")
        for issue in quality_report.issues:
            print(f"  - {issue}")
    
    # Prepare for AI processing
    processed_courses, stats = processor.prepare_for_ai_processing(sample_data)
    print(f"Prepared {len(processed_courses)} courses for AI processing")