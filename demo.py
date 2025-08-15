#!/usr/bin/env python3
"""
Comprehensive Demo Script for AI-Assisted Curriculum Mapping Tool

This script demonstrates the complete workflow of the curriculum mapping tool,
including data loading, processing, AI analysis, and report generation.
It showcases all major features and provides a working example for users.

Usage:
    python demo.py [--api-key YOUR_API_KEY] [--sample-only]

Features Demonstrated:
- Data loading and validation
- Data cleaning and preprocessing
- AI-powered curriculum mapping
- Statistical analysis and insights
- Report generation with visualizations
- Error handling and logging
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.curriculum_mapper import CurriculumMapper
    from src.data_processor import DataProcessor
    from src.report_generator import ReportGenerator
    from src.utils import ConfigManager, ProgressTracker, PerformanceMonitor
except ImportError:
    print("Error: Could not import required modules. Please ensure all source files are present.")
    sys.exit(1)


class CurriculumMappingDemo:
    """
    Comprehensive demonstration of the curriculum mapping tool.
    
    This class orchestrates the complete workflow from data loading
    through report generation, providing a complete example of how
    to use the curriculum mapping system.
    """
    
    def __init__(self, api_key: str = None, sample_only: bool = False):
        """
        Initialize the demo.
        
        Args:
            api_key: Anthropic API key (optional if using environment variable)
            sample_only: If True, only run with sample data without API calls
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.sample_only = sample_only
        self.config = ConfigManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Set up logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directories exist
        self.ensure_directories()
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.report_generator = ReportGenerator()
        
        if not self.sample_only:
            if not self.api_key:
                self.logger.error("API key required for full demo. Use --sample-only or set ANTHROPIC_API_KEY")
                raise ValueError("API key required")
            self.curriculum_mapper = CurriculumMapper(self.api_key)
    
    def setup_logging(self):
        """Set up comprehensive logging for the demo."""
        log_dir = self.config.get_path('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(log_dir, 'demo.log'))
            ]
        )
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        directories = ['outputs', 'logs', 'data', 'templates']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def print_banner(self):
        """Print welcome banner for the demo."""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║               🎓 AI-Assisted Curriculum Mapping Tool Demo 🎓                  ║
║                                                                              ║
║   Demonstrating sophisticated AI-powered curriculum analysis capabilities    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        print(f"Mode: {'Sample Data Only' if self.sample_only else 'Full AI Analysis'}")
        print(f"Configuration: {self.config.config_path}")
        print("=" * 80)
    
    def load_sample_data(self) -> tuple:
        """
        Load sample course and competency data.
        
        Returns:
            Tuple of (courses_data, competencies_data)
        """
        self.logger.info("Loading sample data...")
        
        try:
            # Load course data
            courses_df = self.data_processor.load_data('data/sample_courses.csv')
            self.logger.info(f"Loaded {len(courses_df)} courses")
            
            # Load competencies
            with open('data/learning_outcomes.json', 'r', encoding='utf-8') as f:
                competencies_data = json.load(f)
            competencies = competencies_data['institutional_learning_competencies']
            self.logger.info(f"Loaded {len(competencies)} learning competencies")
            
            return courses_df, competencies
            
        except Exception as e:
            self.logger.error(f"Error loading sample data: {e}")
            raise
    
    def demonstrate_data_processing(self, courses_df) -> list:
        """
        Demonstrate data processing and validation capabilities.
        
        Args:
            courses_df: Raw course data DataFrame
            
        Returns:
            List of processed course dictionaries
        """
        print("\n📊 STEP 1: Data Processing and Validation")
        print("-" * 50)
        
        self.performance_monitor.start_timer('data_processing')
        
        # Validate data structure
        required_columns = ['id', 'title', 'description']
        is_valid, issues = self.data_processor.validate_data_structure(courses_df, required_columns)
        
        if not is_valid:
            self.logger.error(f"Data validation failed: {issues}")
            raise ValueError(f"Data validation errors: {issues}")
        
        print("✅ Data structure validation passed")
        
        # Clean the data
        cleaned_df = self.data_processor.clean_course_data(courses_df)
        print(f"✅ Data cleaned: {len(cleaned_df)} courses processed")
        
        # Assess data quality
        quality_report = self.data_processor.assess_data_quality(cleaned_df)
        print(f"✅ Data quality assessment: {quality_report.quality_score:.1f}/100")
        
        if quality_report.issues:
            print("⚠️  Quality issues found:")
            for issue in quality_report.issues:
                print(f"   - {issue}")
        
        # Export quality report
        quality_report_path = "outputs/data_quality_report.json"
        self.data_processor.export_quality_report(quality_report, quality_report_path)
        print(f"✅ Quality report saved: {quality_report_path}")
        
        # Prepare for AI processing
        processed_courses, stats = self.data_processor.prepare_for_ai_processing(cleaned_df)
        print(f"✅ Prepared {len(processed_courses)} courses for AI analysis")
        
        processing_time = self.performance_monitor.end_timer('data_processing')
        print(f"⏱️  Processing completed in {processing_time:.2f} seconds")
        
        return processed_courses
    
    def demonstrate_ai_mapping(self, courses: list, competencies: list) -> list:
        """
        Demonstrate AI-powered curriculum mapping.
        
        Args:
            courses: List of processed course dictionaries
            competencies: List of learning competencies
            
        Returns:
            List of mapping results
        """
        if self.sample_only:
            return self.create_sample_mappings(courses, competencies)
        
        print("\n🤖 STEP 2: AI-Powered Curriculum Mapping")
        print("-" * 50)
        
        self.performance_monitor.start_timer('ai_mapping')
        
        # Limit to first 5 courses for demo (to manage API costs)
        demo_courses = courses[:5]
        print(f"📚 Analyzing {len(demo_courses)} courses using Claude AI...")
        
        # Set up progress tracking
        progress = ProgressTracker(len(demo_courses), "AI Mapping Progress")
        
        try:
            # Process courses in batch
            mappings = []
            for i, course in enumerate(demo_courses):
                self.logger.info(f"Processing course: {course['id']} - {course['title']}")
                
                mapping = self.curriculum_mapper.map_course(
                    course_id=course['id'],
                    course_title=course['title'],
                    course_description=course['description'],
                    competencies=competencies
                )
                mappings.append(mapping)
                progress.update()
                
                # Show sample result
                if i == 0:
                    print(f"\n📋 Sample Mapping for {course['id']}:")
                    for comp_mapping in mapping.mappings[:3]:
                        print(f"   {comp_mapping['competency_name']}: {comp_mapping['relevance_score']}/10")
                        print(f"      → {comp_mapping['justification'][:100]}...")
            
            progress.finish()
            
            # Export results
            self.curriculum_mapper.export_to_json(mappings, "outputs/mapping_results.json")
            self.curriculum_mapper.export_to_csv(mappings, "outputs/mapping_results.csv")
            print("✅ Mapping results exported")
            
            # Generate statistics
            stats = self.curriculum_mapper.generate_summary_statistics(mappings)
            print(f"✅ Generated statistics: {stats['total_courses']} courses, "
                  f"avg confidence: {stats['average_confidence']:.2f}")
            
            mapping_time = self.performance_monitor.end_timer('ai_mapping')
            print(f"⏱️  AI mapping completed in {mapping_time:.2f} seconds")
            
            # Convert to format expected by report generator
            return self.convert_mappings_for_reporting(mappings)
            
        except Exception as e:
            self.logger.error(f"Error in AI mapping: {e}")
            print(f"❌ AI mapping failed: {e}")
            # Fall back to sample data
            print("🔄 Falling back to sample data...")
            return self.create_sample_mappings(demo_courses, competencies)
    
    def convert_mappings_for_reporting(self, mappings) -> list:
        """Convert mapping objects to dictionaries for report generation."""
        converted_mappings = []
        for mapping in mappings:
            converted_mappings.append({
                'course_id': mapping.course_id,
                'course_title': mapping.course_title,
                'course_description': mapping.course_description,
                'competency_mappings': mapping.mappings,
                'overall_confidence': mapping.overall_confidence,
                'processing_time': mapping.processing_time,
                'timestamp': mapping.timestamp
            })
        return converted_mappings
    
    def create_sample_mappings(self, courses: list, competencies: list) -> list:
        """
        Create sample mapping data when API is not available.
        
        Args:
            courses: List of course dictionaries
            competencies: List of competencies
            
        Returns:
            List of sample mapping dictionaries
        """
        print("\n📋 Creating sample mapping data...")
        
        import random
        random.seed(42)  # For reproducible results
        
        sample_mappings = []
        
        for course in courses[:10]:  # Limit to 10 courses for demo
            competency_mappings = []
            
            for competency in competencies:
                # Generate realistic scores based on course content
                base_score = random.uniform(3, 8)
                
                # Adjust scores based on course title keywords
                title_lower = course['title'].lower()
                desc_lower = course['description'].lower()
                
                # Boost technical skills for CS courses
                if competency['name'] == 'Technical and Professional Skills':
                    if any(keyword in title_lower for keyword in ['computer', 'programming', 'software']):
                        base_score += random.uniform(1, 2)
                
                # Boost critical thinking for analysis courses
                if competency['name'] == 'Critical Thinking and Problem Solving':
                    if any(keyword in desc_lower for keyword in ['analysis', 'problem', 'design']):
                        base_score += random.uniform(0.5, 1.5)
                
                # Boost quantitative reasoning for math/stats courses
                if competency['name'] == 'Quantitative and Analytical Reasoning':
                    if any(keyword in title_lower for keyword in ['math', 'statistics', 'calculus']):
                        base_score += random.uniform(1, 2)
                
                # Cap the score at 10
                score = min(10, base_score)
                
                competency_mappings.append({
                    'competency_name': competency['name'],
                    'relevance_score': round(score, 1),
                    'justification': f"Sample analysis indicates {score:.1f}/10 relevance based on course content and learning objectives.",
                    'evidence_keywords': ['sample', 'demo', 'analysis']
                })
            
            # Calculate overall confidence
            scores = [cm['relevance_score'] for cm in competency_mappings]
            overall_confidence = sum(scores) / len(scores)
            
            sample_mappings.append({
                'course_id': course['id'],
                'course_title': course['title'],
                'course_description': course['description'],
                'competency_mappings': competency_mappings,
                'overall_confidence': overall_confidence,
                'processing_time': random.uniform(0.5, 2.0),
                'timestamp': '2024-01-15T10:30:00'
            })
        
        print(f"✅ Created sample mappings for {len(sample_mappings)} courses")
        return sample_mappings
    
    def demonstrate_report_generation(self, mappings: list):
        """
        Demonstrate comprehensive report generation.
        
        Args:
            mappings: List of mapping result dictionaries
        """
        print("\n📊 STEP 3: Report Generation and Analysis")
        print("-" * 50)
        
        self.performance_monitor.start_timer('report_generation')
        
        # Generate summary statistics
        stats = self.report_generator.create_summary_statistics(mappings)
        print("✅ Summary statistics generated")
        
        # Generate insights
        insights = self.report_generator.generate_insights(stats)
        print(f"✅ Generated {len(insights)} analytical insights")
        
        # Create visualizations
        print("📈 Creating visualizations...")
        
        heatmap_path = "outputs/competency_heatmap.png"
        self.report_generator.create_competency_heatmap(mappings, heatmap_path)
        print(f"   ✅ Competency heatmap: {heatmap_path}")
        
        distribution_path = "outputs/score_distribution.png"
        self.report_generator.create_score_distribution_plot(mappings, distribution_path)
        print(f"   ✅ Score distribution: {distribution_path}")
        
        confidence_path = "outputs/confidence_analysis.png"
        self.report_generator.create_confidence_analysis_plot(mappings, confidence_path)
        print(f"   ✅ Confidence analysis: {confidence_path}")
        
        # Generate comprehensive HTML report
        html_report_path = "outputs/curriculum_mapping_report.html"
        self.report_generator.generate_html_report(mappings, html_report_path)
        print(f"✅ HTML report generated: {html_report_path}")
        
        # Export detailed CSV
        csv_path = "outputs/detailed_mapping_results.csv"
        self.report_generator.export_detailed_csv(mappings, csv_path)
        print(f"✅ Detailed CSV exported: {csv_path}")
        
        report_time = self.performance_monitor.end_timer('report_generation')
        print(f"⏱️  Report generation completed in {report_time:.2f} seconds")
        
        # Display key insights
        print("\n💡 Key Insights:")
        for i, insight in enumerate(insights[:5], 1):
            print(f"   {i}. {insight}")
    
    def display_performance_summary(self):
        """Display performance summary of the demo."""
        print("\n⚡ Performance Summary")
        print("-" * 50)
        
        summary = self.performance_monitor.get_summary()
        for operation, stats in summary.items():
            print(f"{operation.replace('_', ' ').title()}:")
            print(f"   Total Time: {stats['total_time']:.2f}s")
            print(f"   Average Time: {stats['average_time']:.2f}s")
            print(f"   Operations: {stats['count']}")
    
    def display_output_summary(self):
        """Display summary of generated outputs."""
        print("\n📁 Generated Outputs")
        print("-" * 50)
        
        output_files = [
            ("outputs/data_quality_report.json", "Data Quality Report"),
            ("outputs/mapping_results.json", "Mapping Results (JSON)"),
            ("outputs/mapping_results.csv", "Mapping Results (CSV)"),
            ("outputs/curriculum_mapping_report.html", "Comprehensive HTML Report"),
            ("outputs/detailed_mapping_results.csv", "Detailed Analysis CSV"),
            ("outputs/competency_heatmap.png", "Competency Heatmap"),
            ("outputs/score_distribution.png", "Score Distribution Chart"),
            ("outputs/confidence_analysis.png", "Confidence Analysis Chart")
        ]
        
        for file_path, description in output_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"✅ {description}: {file_path} ({size} bytes)")
            else:
                print(f"❌ {description}: {file_path} (not found)")
    
    def run_complete_demo(self):
        """Execute the complete demonstration workflow."""
        try:
            # Print banner
            self.print_banner()
            
            # Load sample data
            courses_df, competencies = self.load_sample_data()
            
            # Step 1: Data Processing
            processed_courses = self.demonstrate_data_processing(courses_df)
            
            # Step 2: AI Mapping
            mappings = self.demonstrate_ai_mapping(processed_courses, competencies)
            
            # Step 3: Report Generation
            self.demonstrate_report_generation(mappings)
            
            # Show performance summary
            self.display_performance_summary()
            
            # Show output summary
            self.display_output_summary()
            
            # Final message
            print("\n🎉 Demo completed successfully!")
            print("\n📖 Next Steps:")
            print("   1. Open outputs/curriculum_mapping_report.html in your browser")
            print("   2. Review the detailed CSV files for further analysis")
            print("   3. Examine the generated visualizations")
            print("   4. Customize the configuration in data/config.yaml")
            print("   5. Add your own course data to data/sample_courses.csv")
            print("\n💡 To run with real AI analysis, set ANTHROPIC_API_KEY and remove --sample-only")
            
        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
            print(f"\n❌ Demo failed: {e}")
            print("Check the logs for detailed error information.")
            raise


def main():
    """Main function to run the demo with command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-Assisted Curriculum Mapping Tool Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo.py --sample-only                    # Run with sample data only
    python demo.py --api-key YOUR_KEY               # Run with real AI analysis
    python demo.py                                  # Use ANTHROPIC_API_KEY environment variable
        """
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='Anthropic API key for Claude access (or set ANTHROPIC_API_KEY env var)'
    )
    
    parser.add_argument(
        '--sample-only',
        action='store_true',
        help='Run demo with sample data only (no API calls)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create and run demo
        demo = CurriculumMappingDemo(
            api_key=args.api_key,
            sample_only=args.sample_only
        )
        
        demo.run_complete_demo()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()