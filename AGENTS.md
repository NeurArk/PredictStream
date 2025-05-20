# AGENTS.md - Guidance for Codex

## IMPORTANT: READ FIRST

Before doing any development work, you MUST read and understand these two critical documents:

1. **REQUIREMENTS.md** - Contains detailed specifications and requirements for the application
2. **TODO.md** - Outlines the development plan with tasks organized by PR milestones

These documents provide essential context for the project. You MUST maintain and update the TODO.md file as you complete tasks, using it to track progress and determine what to work on next. Always work on one PR milestone at a time, and include appropriate tests for each feature you implement.

## Project Overview

PredictStream is a Streamlit-based dashboard application for data exploration and predictive analytics. Your task is to help develop and maintain this app that allows users to upload data, perform exploratory data analysis, and build machine learning models through an intuitive interface.

This is a demonstration project intended to showcase high-quality code and polished user experience.

## Repository Structure

The project follows a simple modular structure with these key components:
- Main application file (app.py) serving as the entry point
- Pages directory for multi-page Streamlit functionality
- Utils directory for helper functions and modules
- Data directory for sample datasets
- Static directory for images and other static resources

## Development Environment

### Setup
The project requires Python 3.11+. To set up the development environment:
1. Install all dependencies listed in requirements.txt
2. Run the application using the Streamlit CLI

### Dependencies
The project relies on common data science and visualization libraries including Streamlit, Pandas, NumPy, scikit-learn, Matplotlib, Plotly, and Seaborn. All dependencies are documented in the requirements.txt file.

## Testing

The project uses pytest for testing. Run all tests with the pytest command from the project root. Write test functions with descriptive names that clearly indicate what's being tested.

Every PR must include appropriate tests to verify the functionality being added. Ensure that test coverage is maintained or improved with each PR.

## Coding Guidelines

### General Practices
- Follow PEP 8 style guidelines for Python code
- Use meaningful variable and function names
- Keep functions small and focused on a single responsibility
- Add docstrings to all functions and classes
- Avoid deep nesting of conditional statements
- Prefer explicit over implicit code

### Streamlit-Specific Guidelines
- Place interactive elements (widgets) at the top of the page
- Use caching decorators for performance optimization
- Implement consistent layout spacing
- Provide clear user instructions for interactive elements
- Use session state to maintain data between page reloads

## Important Design Patterns

### Data Flow Architecture
1. User uploads data or selects sample dataset
2. Data is processed and validated using utility functions
3. Exploratory analysis is performed on the processed data
4. Visualizations are generated based on user selections
5. Machine learning models are built, trained and evaluated

### Session State Management
Use Streamlit's session state to maintain state between page navigations. Important state variables include the current dataset, trained models, and evaluation results.

## Pull Request Workflow

For each PR milestone in TODO.md:

1. Complete all tasks listed in the milestone
2. Write tests for all new functionality
3. Update the TODO.md file to mark completed tasks
4. Create a pull request with a clear description of what was implemented
5. Wait for review and approval before moving to the next milestone

## Error Handling

Implement robust error handling throughout the application. Validate user inputs before processing and provide clear, helpful error messages. Log errors for debugging purposes.

This guidance document is designed to help you understand the structure, conventions, and best practices for the PredictStream project. Refer to it when you need direction on how to approach development tasks.