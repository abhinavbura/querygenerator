# NLP-to-SQL Query Generator

Welcome to the NLP-to-SQL Query Generator project! This project converts natural language input into SQL queries. It utilizes Natural Language Processing (NLP) techniques to interpret user requests and translate them into executable SQL statements.

## Overview

This project involves:
1. **Intent Identification**: Detects the type of SQL query needed (e.g., SELECT, UPDATE, DELETE) using TF-IDF Vectorization and Logistic Regression.
2. **Condition and Column Extraction**: Uses spaCy's Matcher to identify relevant columns and conditions from the natural language input.
3. **Query Construction**: Constructs the appropriate SQL query based on the identified intent, columns, and conditions.

## Getting Started

To get started with the project, follow these steps:

### Prerequisites

Make sure you have Python installed on your machine. The project uses Python 3.x.


## installation

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/nlp-to-sql-query-generator.git
cd nlp-to-sql-query-generator
```

2. **Set Up the Virtual Environment**
Install the required dependencies using requirements.txt:


```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```
## Usage

1. **Prepare Your Data**

   Ensure you have your dataset ready for training and testing. The dataset should include examples of different types of queries with associated intents. 

2. **Run the Query Generator**

   Execute the main script to start the query generation:

   ```bash
   python querygenerator.py
   ```
   The script will prompt you to enter a natural language query. It will then output the corresponding SQL query.

   ## Project Structure

- `querygenerator.py`: The main script for running the NLP-to-SQL query generation.
- `requirements.txt`: A file listing all necessary Python packages.
- `model/`: Directory containing trained models and other resources (if applicable).
- `data/`: Directory for datasets used in training and testing (if applicable).

## Example

Here's an example of how the system processes a natural language request:

**Input**: "Update grade to 12 for student with id 15"

**Output**: `UPDATE students SET class=12 WHERE id=15;`

## Contributing

Feel free to fork the repository and submit pull requests if you have improvements or bug fixes. Please follow the standard contribution guidelines for this repository.

## Acknowledgments

- **spaCy**: For providing powerful NLP tools.
- **scikit-learn**: For machine learning and data processing.

## Future Updates

I plan to enhance this project by adding the following features:

- **Speech-to-Text Module**: Integration to convert spoken language into text before processing.
- **UI with Flask**: A user interface using Flask to make it easier to interact with the query generator.

Stay tuned for updates!
   
