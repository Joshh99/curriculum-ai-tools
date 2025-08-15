"""
Automated Report Generation for Curriculum Mapping

This module provides comprehensive reporting functionality for curriculum mapping
results, including visualizations, statistical analysis, and formatted output
in multiple formats (HTML, PDF).

Key Features:
- Statistical analysis and insights
- Interactive visualizations using matplotlib/seaborn
- HTML report generation with templates
- PDF export capabilities
- Customizable report themes and layouts
- Data export for further analysis
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template, Environment, FileSystemLoader
import base64
from io import BytesIO


class ReportGenerator:
    """
    Automated report generation system for curriculum mapping analysis.
    
    This class creates comprehensive reports with visualizations, statistics,
    and insights from curriculum mapping results.
    """
    
    def __init__(self, output_dir: str = "outputs", template_dir: str = "templates"):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory for output files
            template_dir: Directory containing report templates
        """
        self.output_dir = output_dir
        self.template_dir = template_dir
        self.logger = self._setup_logging()
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Configure matplotlib for better-looking plots
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 10
    
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
    
    def load_mapping_results(self, file_path: str) -> List[Dict]:
        """
        Load mapping results from JSON file.
        
        Args:
            file_path: Path to the mapping results JSON file
            
        Returns:
            List of mapping result dictionaries
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if 'mappings' in data:
                return data['mappings']
            else:
                return data
                
        except Exception as e:
            self.logger.error(f"Error loading mapping results: {e}")
            raise
    
    def create_summary_statistics(self, mappings: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics.
        
        Args:
            mappings: List of mapping result dictionaries
            
        Returns:
            Dictionary with summary statistics
        """
        if not mappings:
            return {}
        
        # Flatten data for analysis
        all_scores = []
        competency_data = {}
        course_data = {}
        
        for mapping in mappings:
            course_id = mapping['course_id']
            course_title = mapping['course_title']
            overall_confidence = mapping.get('overall_confidence', 0)
            
            course_data[course_id] = {
                'title': course_title,
                'confidence': overall_confidence,
                'competency_count': len(mapping['competency_mappings'])
            }
            
            for comp_mapping in mapping['competency_mappings']:
                comp_name = comp_mapping['competency_name']
                score = comp_mapping['relevance_score']
                
                all_scores.append(score)
                
                if comp_name not in competency_data:
                    competency_data[comp_name] = []
                competency_data[comp_name].append(score)
        
        # Calculate overall statistics
        total_courses = len(mappings)
        total_mappings = len(all_scores)
        avg_score = np.mean(all_scores) if all_scores else 0
        score_std = np.std(all_scores) if all_scores else 0
        
        # Competency-level analysis
        competency_stats = {}
        for comp_name, scores in competency_data.items():
            competency_stats[comp_name] = {
                'mean_score': np.mean(scores),
                'median_score': np.median(scores),
                'std_score': np.std(scores),
                'max_score': np.max(scores),
                'min_score': np.min(scores),
                'course_count': len(scores),
                'high_relevance_count': len([s for s in scores if s >= 7]),
                'medium_relevance_count': len([s for s in scores if 4 <= s < 7]),
                'low_relevance_count': len([s for s in scores if s < 4])
            }
        
        # Course-level analysis
        confidence_scores = [data['confidence'] for data in course_data.values()]
        
        # Distribution analysis
        score_distribution = {
            'high_scores': len([s for s in all_scores if s >= 7]),
            'medium_scores': len([s for s in all_scores if 4 <= s < 7]),
            'low_scores': len([s for s in all_scores if s < 4])
        }
        
        return {
            'overview': {
                'total_courses': total_courses,
                'total_mappings': total_mappings,
                'unique_competencies': len(competency_data),
                'average_score': avg_score,
                'score_standard_deviation': score_std,
                'average_confidence': np.mean(confidence_scores) if confidence_scores else 0
            },
            'competency_statistics': competency_stats,
            'course_statistics': course_data,
            'score_distribution': score_distribution,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def create_competency_heatmap(self, mappings: List[Dict], save_path: Optional[str] = None) -> str:
        """
        Create a heatmap showing competency relevance across courses.
        
        Args:
            mappings: List of mapping result dictionaries
            save_path: Optional path to save the plot
            
        Returns:
            Base64 encoded image string or file path
        """
        # Prepare data for heatmap
        course_comp_matrix = {}
        all_competencies = set()
        
        for mapping in mappings:
            course_id = mapping['course_id']
            course_comp_matrix[course_id] = {}
            
            for comp_mapping in mapping['competency_mappings']:
                comp_name = comp_mapping['competency_name']
                score = comp_mapping['relevance_score']
                
                course_comp_matrix[course_id][comp_name] = score
                all_competencies.add(comp_name)
        
        # Create DataFrame for heatmap
        df_data = []
        for course_id, competencies in course_comp_matrix.items():
            row = [competencies.get(comp, 0) for comp in sorted(all_competencies)]
            df_data.append(row)
        
        df = pd.DataFrame(
            df_data,
            index=list(course_comp_matrix.keys()),
            columns=sorted(all_competencies)
        )
        
        # Create heatmap
        plt.figure(figsize=(14, max(8, len(df) * 0.5)))
        
        sns.heatmap(
            df,
            annot=True,
            cmap='RdYlBu_r',
            center=5,
            vmin=0,
            vmax=10,
            cbar_kws={'label': 'Relevance Score'},
            fmt='.1f'
        )
        
        plt.title('Curriculum Mapping: Competency Relevance Heatmap', fontsize=16, pad=20)
        plt.xlabel('Learning Competencies', fontsize=12)
        plt.ylabel('Courses', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            # Return base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return image_base64
    
    def create_score_distribution_plot(self, mappings: List[Dict], save_path: Optional[str] = None) -> str:
        """
        Create visualizations for score distributions.
        
        Args:
            mappings: List of mapping result dictionaries
            save_path: Optional path to save the plot
            
        Returns:
            Base64 encoded image string or file path
        """
        # Extract all scores
        all_scores = []
        competency_scores = {}
        
        for mapping in mappings:
            for comp_mapping in mapping['competency_mappings']:
                score = comp_mapping['relevance_score']
                comp_name = comp_mapping['competency_name']
                
                all_scores.append(score)
                
                if comp_name not in competency_scores:
                    competency_scores[comp_name] = []
                competency_scores[comp_name].append(score)
        
        # Create subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Overall score distribution
        ax1.hist(all_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(all_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_scores):.2f}')
        ax1.set_xlabel('Relevance Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Overall Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot by competency
        comp_names = list(competency_scores.keys())
        comp_score_lists = [competency_scores[comp] for comp in comp_names]
        
        box_plot = ax2.boxplot(comp_score_lists, labels=comp_names, patch_artist=True)
        for patch in box_plot['boxes']:
            patch.set_facecolor('lightblue')
        
        ax2.set_xlabel('Learning Competencies')
        ax2.set_ylabel('Relevance Score')
        ax2.set_title('Score Distribution by Competency')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Score categories pie chart
        high_scores = len([s for s in all_scores if s >= 7])
        medium_scores = len([s for s in all_scores if 4 <= s < 7])
        low_scores = len([s for s in all_scores if s < 4])
        
        categories = ['High (7-10)', 'Medium (4-6)', 'Low (0-3)']
        sizes = [high_scores, medium_scores, low_scores]
        colors = ['green', 'orange', 'red']
        
        ax3.pie(sizes, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Score Category Distribution')
        
        # 4. Competency average scores bar chart
        comp_averages = {comp: np.mean(scores) for comp, scores in competency_scores.items()}
        sorted_comps = sorted(comp_averages.items(), key=lambda x: x[1], reverse=True)
        
        comp_names_sorted = [item[0] for item in sorted_comps]
        comp_avg_scores = [item[1] for item in sorted_comps]
        
        bars = ax4.bar(range(len(comp_names_sorted)), comp_avg_scores, 
                      color='lightcoral', alpha=0.7)
        ax4.set_xlabel('Learning Competencies')
        ax4.set_ylabel('Average Relevance Score')
        ax4.set_title('Average Relevance by Competency')
        ax4.set_xticks(range(len(comp_names_sorted)))
        ax4.set_xticklabels(comp_names_sorted, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return image_base64
    
    def create_confidence_analysis_plot(self, mappings: List[Dict], save_path: Optional[str] = None) -> str:
        """
        Create visualizations for confidence analysis.
        
        Args:
            mappings: List of mapping result dictionaries
            save_path: Optional path to save the plot
            
        Returns:
            Base64 encoded image string or file path
        """
        # Extract confidence data
        confidence_scores = [mapping.get('overall_confidence', 0) for mapping in mappings]
        course_ids = [mapping['course_id'] for mapping in mappings]
        
        # Create subplot layout
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Confidence distribution histogram
        ax1.hist(confidence_scores, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        ax1.axvline(np.mean(confidence_scores), color='red', linestyle='--',
                   label=f'Mean: {np.mean(confidence_scores):.2f}')
        ax1.set_xlabel('Overall Confidence Score')
        ax1.set_ylabel('Number of Courses')
        ax1.set_title('Distribution of Mapping Confidence Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Confidence by course
        sorted_data = sorted(zip(course_ids, confidence_scores), key=lambda x: x[1], reverse=True)
        sorted_courses, sorted_confidence = zip(*sorted_data)
        
        colors = ['green' if score >= 7 else 'orange' if score >= 5 else 'red' 
                 for score in sorted_confidence]
        
        bars = ax2.bar(range(len(sorted_courses)), sorted_confidence, color=colors, alpha=0.7)
        ax2.set_xlabel('Courses (sorted by confidence)')
        ax2.set_ylabel('Overall Confidence Score')
        ax2.set_title('Mapping Confidence by Course')
        ax2.set_xticks(range(len(sorted_courses)))
        ax2.set_xticklabels(sorted_courses, rotation=90)
        ax2.grid(True, alpha=0.3)
        
        # Add confidence threshold lines
        ax2.axhline(y=7, color='green', linestyle=':', alpha=0.7, label='High Confidence (≥7)')
        ax2.axhline(y=5, color='orange', linestyle=':', alpha=0.7, label='Medium Confidence (≥5)')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return image_base64
    
    def generate_insights(self, stats: Dict[str, Any]) -> List[str]:
        """
        Generate analytical insights from the statistics.
        
        Args:
            stats: Summary statistics dictionary
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Overview insights
        overview = stats.get('overview', {})
        avg_score = overview.get('average_score', 0)
        total_courses = overview.get('total_courses', 0)
        
        if avg_score >= 6:
            insights.append(f"Strong overall alignment: Average relevance score of {avg_score:.1f} indicates good curriculum-competency alignment.")
        elif avg_score >= 4:
            insights.append(f"Moderate alignment: Average relevance score of {avg_score:.1f} suggests room for improvement in curriculum design.")
        else:
            insights.append(f"Weak alignment: Low average relevance score of {avg_score:.1f} indicates significant gaps in curriculum-competency mapping.")
        
        # Competency insights
        comp_stats = stats.get('competency_statistics', {})
        if comp_stats:
            # Find strongest and weakest competencies
            comp_means = {comp: data['mean_score'] for comp, data in comp_stats.items()}
            strongest_comp = max(comp_means, key=comp_means.get)
            weakest_comp = min(comp_means, key=comp_means.get)
            
            insights.append(f"Strongest competency coverage: '{strongest_comp}' with average score of {comp_means[strongest_comp]:.1f}.")
            insights.append(f"Weakest competency coverage: '{weakest_comp}' with average score of {comp_means[weakest_comp]:.1f}.")
            
            # Coverage analysis
            well_covered = len([comp for comp, score in comp_means.items() if score >= 6])
            poorly_covered = len([comp for comp, score in comp_means.items() if score < 4])
            
            if well_covered / len(comp_means) >= 0.7:
                insights.append(f"Excellent competency coverage: {well_covered}/{len(comp_means)} competencies well-covered.")
            elif poorly_covered / len(comp_means) >= 0.3:
                insights.append(f"Poor competency coverage: {poorly_covered}/{len(comp_means)} competencies under-represented.")
        
        # Score distribution insights
        score_dist = stats.get('score_distribution', {})
        total_mappings = sum(score_dist.values()) if score_dist else 0
        
        if total_mappings > 0:
            high_percentage = (score_dist.get('high_scores', 0) / total_mappings) * 100
            low_percentage = (score_dist.get('low_scores', 0) / total_mappings) * 100
            
            if high_percentage >= 40:
                insights.append(f"High relevance dominance: {high_percentage:.1f}% of mappings show high relevance (7-10).")
            elif low_percentage >= 40:
                insights.append(f"Low relevance concern: {low_percentage:.1f}% of mappings show low relevance (0-3).")
        
        return insights
    
    def generate_html_report(self, mappings: List[Dict], output_path: str, 
                           template_name: str = "report_template.html"):
        """
        Generate comprehensive HTML report.
        
        Args:
            mappings: List of mapping result dictionaries
            output_path: Path for the output HTML file
            template_name: Name of the HTML template file
        """
        # Generate statistics and visualizations
        stats = self.create_summary_statistics(mappings)
        insights = self.generate_insights(stats)
        
        # Create visualizations as base64 images
        heatmap_img = self.create_competency_heatmap(mappings)
        distribution_img = self.create_score_distribution_plot(mappings)
        confidence_img = self.create_confidence_analysis_plot(mappings)
        
        # Prepare template data
        template_data = {
            'title': 'Curriculum Mapping Analysis Report',
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'statistics': stats,
            'insights': insights,
            'mappings': mappings,
            'heatmap_image': heatmap_img,
            'distribution_image': distribution_img,
            'confidence_image': confidence_img,
            'total_courses': len(mappings),
            'unique_competencies': len(stats.get('competency_statistics', {}))
        }
        
        # Load and render template
        try:
            env = Environment(loader=FileSystemLoader(self.template_dir))
            template = env.get_template(template_name)
            html_content = template.render(**template_data)
            
        except Exception as e:
            self.logger.warning(f"Template loading failed, using simple HTML: {e}")
            html_content = self._generate_simple_html_report(template_data)
        
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {output_path}")
    
    def _generate_simple_html_report(self, data: Dict) -> str:
        """Generate a simple HTML report when template is not available."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{data['title']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }}
                .section {{ margin: 30px 0; }}
                .insight {{ background: #f0f8ff; padding: 10px; margin: 10px 0; border-left: 4px solid #4CAF50; }}
                .stats {{ background: #f9f9f9; padding: 15px; border-radius: 5px; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{data['title']}</h1>
                <p>Generated on: {data['generation_date']}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="stats">
                    <p><strong>Total Courses Analyzed:</strong> {data['total_courses']}</p>
                    <p><strong>Learning Competencies:</strong> {data['unique_competencies']}</p>
                    <p><strong>Average Relevance Score:</strong> {data['statistics']['overview']['average_score']:.2f}/10</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Key Insights</h2>
                {''.join([f'<div class="insight">{insight}</div>' for insight in data['insights']])}
            </div>
            
            <div class="section">
                <h2>Competency Heatmap</h2>
                <img src="data:image/png;base64,{data['heatmap_image']}" alt="Competency Heatmap">
            </div>
            
            <div class="section">
                <h2>Score Distribution Analysis</h2>
                <img src="data:image/png;base64,{data['distribution_image']}" alt="Score Distribution">
            </div>
            
            <div class="section">
                <h2>Confidence Analysis</h2>
                <img src="data:image/png;base64,{data['confidence_image']}" alt="Confidence Analysis">
            </div>
        </body>
        </html>
        """
        return html
    
    def export_detailed_csv(self, mappings: List[Dict], output_path: str):
        """
        Export detailed mapping results to CSV for further analysis.
        
        Args:
            mappings: List of mapping result dictionaries
            output_path: Path for the output CSV file
        """
        rows = []
        
        for mapping in mappings:
            base_data = {
                'course_id': mapping['course_id'],
                'course_title': mapping['course_title'],
                'overall_confidence': mapping.get('overall_confidence', 0),
                'processing_time': mapping.get('processing_time', 0),
                'timestamp': mapping.get('timestamp', '')
            }
            
            for comp_mapping in mapping['competency_mappings']:
                row = base_data.copy()
                row.update({
                    'competency_name': comp_mapping['competency_name'],
                    'relevance_score': comp_mapping['relevance_score'],
                    'justification': comp_mapping['justification'],
                    'evidence_keywords': ', '.join(comp_mapping.get('evidence_keywords', []))
                })
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        self.logger.info(f"Detailed CSV exported: {output_path}")


if __name__ == "__main__":
    # Example usage
    generator = ReportGenerator()
    
    # Sample mapping data for testing
    sample_mappings = [
        {
            'course_id': 'CS101',
            'course_title': 'Introduction to Programming',
            'overall_confidence': 8.5,
            'competency_mappings': [
                {
                    'competency_name': 'Technical Skills',
                    'relevance_score': 9,
                    'justification': 'Strong programming focus',
                    'evidence_keywords': ['programming', 'coding', 'algorithms']
                },
                {
                    'competency_name': 'Critical Thinking',
                    'relevance_score': 7,
                    'justification': 'Problem-solving emphasis',
                    'evidence_keywords': ['problem-solving', 'logic']
                }
            ]
        }
    ]
    
    # Generate statistics and insights
    stats = generator.create_summary_statistics(sample_mappings)
    insights = generator.generate_insights(stats)
    
    print("Generated insights:")
    for insight in insights:
        print(f"- {insight}")
    
    # Generate visualizations
    heatmap_path = "outputs/sample_heatmap.png"
    generator.create_competency_heatmap(sample_mappings, heatmap_path)
    print(f"Heatmap saved to: {heatmap_path}")
    
    # Generate HTML report
    html_path = "outputs/sample_report.html"
    generator.generate_html_report(sample_mappings, html_path)
    print(f"HTML report saved to: {html_path}")