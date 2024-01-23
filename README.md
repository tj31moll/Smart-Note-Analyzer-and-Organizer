# Smart Note Analyzer and Organizer

This Flask application leverages OpenAI's GPT for auto-summarizing and smart tagging of notes, and a machine learning model for task prioritization. It provides a user-friendly interface for note analysis, organization, and exportation.

## Features

- **Auto-Summarize Notes**: Summarizes notes using OpenAI's GPT.
- **Smart Tagging**: Generates relevant tags for each note.
- **Task Prioritization**: Prioritizes notes using a machine learning model.
- **Database Integration**: Stores and manages notes in a SQLite database.
- **Export Functionality**: Allows exporting notes to a CSV file.

## Requirements

- Python 3
- Flask
- OpenAI API key
- Pandas
- NumPy
- scikit-learn
- SQLite3

## Installation

1. Clone or download the application.
2. Install the required Python packages:
   ```bash
   pip install Flask pandas numpy scikit-learn openai
   ```
3. Set your OpenAI API key in the script.

## Database Setup

Create a SQLite database named `notes.db` with a table structure suitable for storing notes. Example SQL:
```sql
CREATE TABLE notes (
    id INTEGER PRIMARY KEY,
    title TEXT,
    text TEXT,
    labels TEXT
);
```

## Usage

1. Start the Flask application:
   ```bash
   python app.py
   ```
2. Access the web interface at `http://localhost:5000`.
3. Input notes for analysis, tagging, and prioritization.
4. View and download the organized notes.

## API Routes

- `GET /`: Home page.
- `POST /analyze`: Processes and organizes the input notes.
- `GET /download`: Downloads the notes as a CSV file.

## Configuration

- Set `openai.api_key` to your OpenAI API key.
- Customize the machine learning model and data processing as needed.

## Example Use-Cases

- Efficient note-taking and summarization for meetings and lectures.
- Organizing and prioritizing tasks and to-do lists.
- Exporting notes for use in other applications like Google Keep.

## Note

This is a demonstration application. Modify and use it according to your requirements and OpenAI usage policies.
