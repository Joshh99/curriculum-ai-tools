"""
AI-Assisted Curriculum Mapping Tool

This module provides functionality to map course descriptions to institutional learning
competencies using Anthropic's Claude API. It includes confidence scoring, batch processing,
and multiple output formats for comprehensive curriculum analysis.

Key Features:
- Claude API integration for intelligent course analysis
- Confidence scoring for mapping reliability
- Batch processing for multiple courses
- Multiple export formats (CSV, JSON)
- Comprehensive error handling and logging
"""

import json
import csv
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import anthropic


@dataclass
class CourseMapping:
    """Represents a mapping between a course and learning competencies."""
    course_id: str
    course_title: str
    course_description: str
    mappings: List[Dict[str, any]]
    overall_confidence: float
    processing_time: float
    timestamp: str


class CurriculumMapper:
    """
    Main class for AI-assisted curriculum mapping using Claude API.
    
    This class handles the integration with Claude API to analyze course descriptions
    and map them to predefined learning competencies with confidence scores.
    """
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        """
        Initialize the curriculum mapper.
        
        Args:
            api_key: Anthropic API key for Claude access
            model: Claude model to use for analysis
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.logger = self._setup_logging()
        
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
    
    def _create_mapping_prompt(self, course_description: str, competencies: List[Dict]) -> str:
        """
        Create the prompt for Claude API to analyze course mapping.
        
        Args:
            course_description: Description of the course to analyze
            competencies: List of institutional learning competencies
            
        Returns:
            Formatted prompt for Claude API
        """
        competencies_text = "\n".join([
            f"- {comp['name']}: {comp['description']}"
            for comp in competencies
        ])
        
        prompt = f"""
Analyze this course description and map it to the following institutional learning competencies. 
For each competency, provide a relevance score (0-10) and brief justification.

Course Description:
{course_description}

Institutional Learning Competencies:
{competencies_text}

Instructions:
- Score each competency from 0 (not relevant) to 10 (highly relevant)
- Provide a brief justification (1-2 sentences) for each score
- Focus on explicit and implicit learning outcomes in the course
- Consider both content knowledge and skill development
- Be objective and evidence-based in your analysis

Return your response in valid JSON format with the following structure:
{{
    "mappings": [
        {{
            "competency_name": "competency name",
            "relevance_score": score,
            "justification": "brief explanation",
            "evidence_keywords": ["keyword1", "keyword2"]
        }}
    ],
    "overall_assessment": "brief summary of course's primary learning focus"
}}
"""
        return prompt
    
    def _parse_claude_response(self, response_text: str) -> Dict:
        """
        Parse Claude's response and extract mapping information.
        
        Args:
            response_text: Raw response from Claude API
            
        Returns:
            Parsed mapping data
        """
        try:
            # Find JSON content in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON content found in response")
                
            json_content = response_text[start_idx:end_idx]
            return json.loads(json_content)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError(f"Invalid JSON in Claude response: {e}")
    
    def map_course(self, course_id: str, course_title: str, 
                   course_description: str, competencies: List[Dict]) -> CourseMapping:
        """
        Map a single course to learning competencies using Claude API.
        
        Args:
            course_id: Unique identifier for the course
            course_title: Title of the course
            course_description: Detailed description of the course
            competencies: List of institutional learning competencies
            
        Returns:
            CourseMapping object with analysis results
        """
        start_time = time.time()
        
        try:
            # Create prompt for Claude
            prompt = self._create_mapping_prompt(course_description, competencies)
            
            # Call Claude API
            self.logger.info(f"Analyzing course: {course_id} - {course_title}")
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Parse response
            response_data = self._parse_claude_response(message.content[0].text)
            
            # Calculate overall confidence score
            scores = [mapping['relevance_score'] for mapping in response_data['mappings']]
            overall_confidence = sum(scores) / len(scores) if scores else 0
            
            processing_time = time.time() - start_time
            
            # Create mapping object
            mapping = CourseMapping(
                course_id=course_id,
                course_title=course_title,
                course_description=course_description,
                mappings=response_data['mappings'],
                overall_confidence=overall_confidence,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )
            
            self.logger.info(f"Completed analysis for {course_id} in {processing_time:.2f}s")
            return mapping
            
        except Exception as e:
            self.logger.error(f"Error mapping course {course_id}: {e}")
            raise
    
    def map_courses_batch(self, courses: List[Dict], competencies: List[Dict],
                         delay_between_calls: float = 1.0) -> List[CourseMapping]:
        """
        Map multiple courses in batch with rate limiting.
        
        Args:
            courses: List of course dictionaries with id, title, description
            competencies: List of institutional learning competencies
            delay_between_calls: Delay between API calls to respect rate limits
            
        Returns:
            List of CourseMapping objects
        """
        mappings = []
        total_courses = len(courses)
        
        self.logger.info(f"Starting batch processing of {total_courses} courses")
        
        for i, course in enumerate(courses, 1):
            try:
                mapping = self.map_course(
                    course_id=course['id'],
                    course_title=course['title'],
                    course_description=course['description'],
                    competencies=competencies
                )
                mappings.append(mapping)
                
                self.logger.info(f"Progress: {i}/{total_courses} courses completed")
                
                # Rate limiting
                if i < total_courses:
                    time.sleep(delay_between_calls)
                    
            except Exception as e:
                self.logger.error(f"Failed to process course {course.get('id', 'unknown')}: {e}")
                continue
        
        self.logger.info(f"Batch processing completed. {len(mappings)} successful mappings")
        return mappings
    
    def export_to_csv(self, mappings: List[CourseMapping], output_path: str):
        """
        Export mapping results to CSV format.
        
        Args:
            mappings: List of CourseMapping objects
            output_path: Path for the output CSV file
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'course_id', 'course_title', 'competency_name', 'relevance_score',
                'justification', 'evidence_keywords', 'overall_confidence',
                'processing_time', 'timestamp'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for mapping in mappings:
                for comp_mapping in mapping.mappings:
                    writer.writerow({
                        'course_id': mapping.course_id,
                        'course_title': mapping.course_title,
                        'competency_name': comp_mapping['competency_name'],
                        'relevance_score': comp_mapping['relevance_score'],
                        'justification': comp_mapping['justification'],
                        'evidence_keywords': ', '.join(comp_mapping.get('evidence_keywords', [])),
                        'overall_confidence': mapping.overall_confidence,
                        'processing_time': mapping.processing_time,
                        'timestamp': mapping.timestamp
                    })
        
        self.logger.info(f"Exported {len(mappings)} mappings to {output_path}")
    
    def export_to_json(self, mappings: List[CourseMapping], output_path: str):
        """
        Export mapping results to JSON format.
        
        Args:
            mappings: List of CourseMapping objects
            output_path: Path for the output JSON file
        """
        data = {
            'metadata': {
                'total_courses': len(mappings),
                'export_timestamp': datetime.now().isoformat(),
                'model_used': self.model
            },
            'mappings': []
        }
        
        for mapping in mappings:
            data['mappings'].append({
                'course_id': mapping.course_id,
                'course_title': mapping.course_title,
                'course_description': mapping.course_description,
                'competency_mappings': mapping.mappings,
                'overall_confidence': mapping.overall_confidence,
                'processing_time': mapping.processing_time,
                'timestamp': mapping.timestamp
            })
        
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Exported {len(mappings)} mappings to {output_path}")
    
    def generate_summary_statistics(self, mappings: List[CourseMapping]) -> Dict:
        """
        Generate summary statistics for the mapping results.
        
        Args:
            mappings: List of CourseMapping objects
            
        Returns:
            Dictionary with summary statistics
        """
        if not mappings:
            return {}
        
        # Collect all relevance scores
        all_scores = []
        competency_scores = {}
        
        for mapping in mappings:
            for comp_mapping in mapping.mappings:
                score = comp_mapping['relevance_score']
                comp_name = comp_mapping['competency_name']
                
                all_scores.append(score)
                
                if comp_name not in competency_scores:
                    competency_scores[comp_name] = []
                competency_scores[comp_name].append(score)
        
        # Calculate statistics
        avg_score = sum(all_scores) / len(all_scores)
        avg_confidence = sum(m.overall_confidence for m in mappings) / len(mappings)
        total_processing_time = sum(m.processing_time for m in mappings)
        
        # Competency-level statistics
        competency_stats = {}
        for comp_name, scores in competency_scores.items():
            competency_stats[comp_name] = {
                'average_score': sum(scores) / len(scores),
                'max_score': max(scores),
                'min_score': min(scores),
                'course_count': len(scores)
            }
        
        return {
            'total_courses': len(mappings),
            'average_relevance_score': avg_score,
            'average_confidence': avg_confidence,
            'total_processing_time': total_processing_time,
            'competency_statistics': competency_stats,
            'high_confidence_mappings': len([m for m in mappings if m.overall_confidence >= 7.0]),
            'low_confidence_mappings': len([m for m in mappings if m.overall_confidence < 5.0])
        }


if __name__ == "__main__":
    # Example usage
    import os
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Please set ANTHROPIC_API_KEY environment variable")
        exit(1)
    
    mapper = CurriculumMapper(api_key)
    
    # Example course and competencies
    sample_courses = [
        {
            'id': 'CS101',
            'title': 'Introduction to Computer Science',
            'description': 'Fundamental concepts in computer science including programming, algorithms, and data structures.'
        }
    ]
    
    sample_competencies = [
        {
            'name': 'Critical Thinking',
            'description': 'Ability to analyze problems systematically and develop logical solutions'
        },
        {
            'name': 'Technical Skills',
            'description': 'Proficiency in relevant tools, technologies, and methodologies'
        }
    ]
    
    # Process courses
    results = mapper.map_courses_batch(sample_courses, sample_competencies)
    
    # Export results
    mapper.export_to_json(results, 'outputs/sample_mapping.json')
    mapper.export_to_csv(results, 'outputs/sample_mapping.csv')
    
    # Print summary
    stats = mapper.generate_summary_statistics(results)
    print(f"Processed {stats['total_courses']} courses")
    print(f"Average confidence: {stats['average_confidence']:.2f}")