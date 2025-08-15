# 🎓 AI-Assisted Curriculum Mapping Tool

A sophisticated Python-based system that leverages Anthropic's Claude AI to analyze course descriptions and map them to institutional learning competencies. This tool demonstrates advanced AI integration, data processing, and educational assessment capabilities.

## 🌟 Features

### 🤖 AI-Powered Analysis
- **Claude API Integration**: Uses Anthropic's Claude for intelligent curriculum analysis
- **Confidence Scoring**: Provides reliability metrics for each mapping
- **Batch Processing**: Efficiently handles multiple courses with rate limiting
- **Natural Language Understanding**: Sophisticated analysis of course descriptions

### 📊 Data Processing Pipeline
- **Data Validation**: Comprehensive quality assessment and reporting
- **Text Cleaning**: Standardization and normalization of course descriptions
- **Format Support**: CSV, JSON, Excel file input formats
- **Quality Metrics**: Detailed data quality scoring and recommendations

### 📈 Advanced Reporting
- **Interactive Visualizations**: Heatmaps, distribution plots, confidence analysis
- **HTML Reports**: Professional, responsive report templates
- **Multiple Export Formats**: CSV, JSON, HTML, PNG outputs
- **Statistical Analysis**: Comprehensive insights and recommendations

### 🛠️ Professional Features
- **Configuration Management**: YAML-based configuration system
- **Error Handling**: Robust error handling and recovery
- **Logging**: Comprehensive logging with multiple levels
- **Performance Monitoring**: Built-in performance tracking and optimization

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Anthropic API key (for full functionality)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Joshh99/curriculum-ai-tools.git
   cd curriculum-ai-tools
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key** (optional for demo):
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

### Running the Demo

**Option 1: Sample Data Only (No API Key Required)**
```bash
python demo.py --sample-only
```

**Option 2: Full AI Analysis**
```bash
python demo.py --api-key your-api-key
# or if you set the environment variable:
python demo.py
```

### Demo Output
The demo will generate:
- Data quality reports
- AI mapping results
- Comprehensive HTML report
- Visualization charts
- Statistical analysis

## 📁 Project Structure

```
curriculum-ai-tools/
├── src/                          # Source code
│   ├── curriculum_mapper.py      # Main AI mapping functionality
│   ├── data_processor.py         # Data cleaning and validation
│   ├── report_generator.py       # Report generation and visualization
│   └── utils.py                  # Utility functions and configuration
├── data/                         # Sample data and configuration
│   ├── sample_courses.csv        # Carnegie Mellon course descriptions
│   ├── learning_outcomes.json    # Institutional learning competencies
│   └── config.yaml              # System configuration
├── templates/                    # HTML report templates
│   └── report_template.html     # Professional report template
├── outputs/                     # Generated reports and results
├── demo.py                      # Comprehensive demonstration script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔧 Core Components

### 1. CurriculumMapper (`src/curriculum_mapper.py`)
The main AI integration module that:
- Interfaces with Claude API for course analysis
- Generates confidence-scored mappings
- Handles batch processing with rate limiting
- Exports results in multiple formats

**Key Methods**:
```python
mapper = CurriculumMapper(api_key)
mapping = mapper.map_course(course_id, title, description, competencies)
mappings = mapper.map_courses_batch(courses, competencies)
```

### 2. DataProcessor (`src/data_processor.py`)
Comprehensive data processing pipeline that:
- Validates and cleans course data
- Performs quality assessment
- Standardizes text content
- Prepares data for AI processing

**Key Methods**:
```python
processor = DataProcessor()
quality_report = processor.assess_data_quality(df)
processed_courses = processor.prepare_for_ai_processing(df)
```

### 3. ReportGenerator (`src/report_generator.py`)
Advanced reporting system that:
- Creates interactive visualizations
- Generates professional HTML reports
- Provides statistical analysis
- Exports data for further analysis

**Key Methods**:
```python
generator = ReportGenerator()
generator.create_competency_heatmap(mappings)
generator.generate_html_report(mappings, output_path)
```

## 📊 Sample Data

### Course Data
The project includes 20 real Carnegie Mellon University course descriptions covering:
- Computer Science (Programming, Systems, AI/ML)
- Mathematics (Calculus, Statistics, Analysis)
- Engineering (Mechanical, Electrical)
- Liberal Arts (History, Psychology, English)
- Business (Corporate Finance)
- Physics (Experimental, Theoretical)

### Learning Competencies
10 comprehensive institutional learning competencies:
1. **Critical Thinking and Problem Solving**
2. **Technical and Professional Skills**
3. **Communication and Collaboration**
4. **Research and Information Literacy**
5. **Quantitative and Analytical Reasoning**
6. **Creativity and Innovation**
7. **Ethical Reasoning and Social Responsibility**
8. **Global and Cultural Awareness**
9. **Leadership and Initiative**
10. **Adaptability and Lifelong Learning**

## 🎯 Educational Assessment Concepts

### Curriculum Mapping
The process of documenting relationships between curriculum standards, taught curriculum, and tested curriculum. This tool automates the analysis of course content against institutional learning outcomes.

### Competency-Based Education
An educational approach that focuses on students demonstrating specific competencies rather than spending time in classes. Our tool helps institutions assess how well their courses develop these competencies.

### Learning Outcomes Assessment
The systematic collection and analysis of information to improve student learning. This tool provides quantitative analysis of curriculum alignment with learning objectives.

## 🔍 AI Integration Strategies

### Prompt Engineering
The tool uses sophisticated prompts that:
- Provide clear instructions for competency mapping
- Include scoring rubrics and evidence requirements
- Request structured JSON responses for consistency
- Incorporate educational assessment best practices

### Confidence Scoring
Each mapping includes confidence metrics based on:
- Relevance score distributions
- Consistency of evidence keywords
- Alignment with educational assessment criteria
- Statistical analysis of mapping patterns

### Quality Assurance
Multiple validation layers ensure mapping quality:
- Input data validation and cleaning
- AI response parsing and validation
- Statistical outlier detection
- Human-readable justifications for review

## 📈 Output Examples

### Statistical Summary
```
Total Courses Analyzed: 20
Average Relevance Score: 6.8/10
Average Confidence: 7.2/10
High-Quality Mappings: 85%
```

### Sample Insights
- "Strong overall alignment: Average relevance score of 6.8 indicates good curriculum-competency alignment"
- "Strongest competency coverage: 'Technical and Professional Skills' with average score of 8.2"
- "High relevance dominance: 45.2% of mappings show high relevance (7-10)"

## ⚙️ Configuration

### System Configuration (`data/config.yaml`)
```yaml
api:
  anthropic:
    model: "claude-3-sonnet-20240229"
    max_tokens: 4000
    temperature: 0.1

data_processing:
  min_description_length: 50
  max_description_length: 5000
  quality_threshold: 70.0

reporting:
  output_formats: ["html", "csv", "json"]
  visualizations: ["competency_heatmap", "score_distribution"]
```

### Environment Variables
```bash
ANTHROPIC_API_KEY=your-api-key-here
```

## 🧪 Advanced Usage

### Custom Data Processing
```python
from src.data_processor import DataProcessor
from src.curriculum_mapper import CurriculumMapper

# Load and process custom data
processor = DataProcessor()
courses_df = processor.load_data("your_courses.csv")
processed_courses = processor.prepare_for_ai_processing(courses_df)

# Run AI analysis
mapper = CurriculumMapper(api_key)
mappings = mapper.map_courses_batch(processed_courses, competencies)
```

### Custom Report Generation
```python
from src.report_generator import ReportGenerator

generator = ReportGenerator()
stats = generator.create_summary_statistics(mappings)
insights = generator.generate_insights(stats)
generator.generate_html_report(mappings, "custom_report.html")
```

### Performance Monitoring
```python
from src.utils import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_timer("operation")
# ... perform operation
duration = monitor.end_timer("operation")
stats = monitor.get_statistics("operation")
```

## 🔬 Technical Details

### AI Model Specifications
- **Model**: Claude-3-Sonnet (claude-3-sonnet-20240229)
- **Context Window**: 200K tokens
- **Temperature**: 0.1 (for consistent, analytical responses)
- **Max Tokens**: 4000 per response

### Performance Characteristics
- **Processing Speed**: ~2-3 seconds per course analysis
- **Batch Processing**: Configurable rate limiting
- **Memory Usage**: Optimized for large datasets
- **Error Handling**: Comprehensive retry mechanisms

### Data Quality Metrics
- **Completeness**: Percentage of non-missing data
- **Validity**: Adherence to data format requirements
- **Consistency**: Standardization across records
- **Accuracy**: Alignment with expected patterns

## 🛡️ Security and Privacy

### Data Protection
- No persistent storage of sensitive data
- API keys masked in logs
- Input sanitization and validation
- Secure file handling practices

### API Usage
- Rate limiting to respect service limits
- Error handling for API failures
- Efficient token usage optimization
- Secure credential management

## 🤝 Contributing

### Development Setup
1. Clone the repository
2. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   # For development tools:
   pip install black flake8 pytest
   ```
3. Run tests:
   ```bash
   pytest tests/
   ```

### Code Quality
- Follow PEP 8 style guidelines
- Use type hints for function parameters
- Include comprehensive docstrings
- Write unit tests for new features

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Carnegie Mellon University** for inspiring this educational technology
- **Anthropic** for the Claude AI platform
- **Python Community** for the excellent libraries and tools
- **Educational Assessment Community** for research and best practices

## 📞 Support

For questions, issues, or contributions:
1. Check the documentation in this README
2. Review the code comments and docstrings
3. Run the demo script to understand functionality
4. Open an issue for bugs or feature requests

## 🚀 Future Enhancements

### Planned Features
- **Web Interface**: Flask/FastAPI-based web application
- **Database Integration**: PostgreSQL/SQLite support
- **Advanced NLP**: Spacy integration for text analysis
- **Interactive Dashboards**: Plotly/Bokeh visualizations
- **API Development**: RESTful API for external integrations

### Research Opportunities
- **Multi-language Support**: Course analysis in multiple languages
- **Temporal Analysis**: Tracking curriculum changes over time
- **Benchmarking**: Comparison with industry standards
- **Predictive Analytics**: Student outcome prediction based on curriculum mapping

---

*This project demonstrates sophisticated AI integration for educational applications, showcasing advanced software engineering practices and educational assessment expertise.*
