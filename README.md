<p align="center">
  <img src="https://cont-edu.technion.ac.il/wp-content/uploads/2024/07/smallpicAI-156x156.jpg" alt="Logo" width="120" height="120">
</p>

<h1 align="center">TechFlow AI Recruiter</h1>

<p align="center">
  <b>Final Project for the GenAI & LLM Course at the Technion</b><br>
  A multi-agent recruitment assistant that screens candidates, answers FAQs using RAG, and schedules interviews.<br>
  <br>
  <a href="#usage">How to Run</a>
  ·
  <a href="#contact">Contact</a>
</p>

---

## Table of Contents

- [About The Project](#about-the-project)
- [Architecture & Features](#architecture--features)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Database Setup](#database-setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Evaluation](#evaluation)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## About The Project

> **Course:** Generative AI & Large Language Models (Technion)<br>
> **Goal:** Create an intelligent conversational agent capable of handling complex recruitment flows.

This project is an **AI-powered HR Recruitment Bot** designed to automate the initial stages of the hiring process. Unlike simple chatbots, this system uses an **Agent Orchestration** architecture to handle different aspects of the conversation dynamically.

**Key capabilities:**
* **Screens candidates** based on their responses
* **Answers technical questions** about the job description using a specialist RAG agent
* **Detects disinterest** or rejection using a fine-tuned LLM
* **Schedules interviews** by checking real-time slot availability

**Technologies:** Python, LangChain, OpenAI (GPT-4 & Fine-tuned models), ChromaDB (Vector Store), SQLite/MongoDB (Scheduling), Streamlit (UI)

---

## Architecture & Features

This bot operates on a "Manager-Expert" hierarchy:

- [x] **Main Orchestrator Agent:** Controls the flow of conversation and delegates tasks (CONTINUE / SCHEDULE / END)
- [x] **Info Advisor (RAG):** A specialized agent that uses **ChromaDB** to retrieve specific details from the Job Description
- [x] **Exit Advisor (Fine-Tuned):** Uses a custom fine-tuned model to accurately detect when a candidate wants to end the conversation
- [x] **Scheduling Advisor:** Connects to database using **Function Calling** to fetch available interview slots and book meetings
- [x] **Streamlit Interface:** A user-friendly web UI for candidates to interact with the bot

### Prompting Strategies

| Strategy | Implementation |
|----------|---------------|
| **Role-Based Prompts** | Each agent has a defined persona and role |
| **Few-Shot Learning** | 10+ curated examples per agent |
| **API Parameters** | Optimized temperature and max_tokens |
| **Instruction Prompts** | Detailed behavioral guidelines |

---

## Getting Started

Follow these steps to set up the project locally.

### Prerequisites

- Python >= 3.9
- OpenAI API Key
- MongoDB Atlas Account (optional, SQLite fallback available)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Orlavan/TechFlow-AI-Recruiter.git
   cd TechFlow-AI-Recruiter
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # Mac/Linux:
   source venv/bin/activate
   # Windows:
   .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

Create a `.env` file in the root directory:

```bash
# OpenAI API Key (required)
OPENAI_API_KEY=sk-your_key_here

# MongoDB Connection String (optional)
MONGO_URI=mongodb+srv://<username>:<password>@cluster0.mongodb.net/

# Fine-tuned Exit Model ID (optional, after training)
FINETUNED_EXIT_MODEL=ft:gpt-4.1-...
```

---

## Database Setup

### SQLite (Default)
The project includes a pre-configured SQLite database (`tech.db`) with interview slots.

### MongoDB (Optional)
1. Create a Database named: `Project`
2. Create a Collection named: `Availability`
3. Import the `availability.json` file into this collection

### ChromaDB (Job Knowledge)
To enable the bot to answer questions about the job position:

```bash
python -c "from app.modules.info_agent.ingest import EmbeddingsManager; EmbeddingsManager()"
```

This creates the `chroma_db` folder and indexes the Job Description.

---

## Training the Exit Agent

This project utilizes a fine-tuned OpenAI model for exit detection.

1. **Prepare Training Data:**
   ```bash
   python -m app.modules.fine_tuning --prepare
   ```

2. **Start Fine-Tuning Job:**
   ```bash
   python -m app.modules.fine_tuning --start
   ```

3. **Check Status:**
   ```bash
   python -m app.modules.fine_tuning --status <job_id>
   ```

4. **Update Model ID:** Once complete, add the new `ft:gpt...` ID to your `.env` file.

---

## Project Structure

```
TechFlow-AI-Recruiter/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── modules/
│       ├── agents.py              # Main Orchestrator
│       ├── fine_tuning.py         # Fine-tuning pipeline
│       ├── exit_agent/            # Exit Advisor (Fine-tuned)
│       ├── info_agent/            # RAG Logic (ChromaDB)
│       │   ├── chroma_db/         # Vector Database
│       │   ├── ingest.py
│       │   └── info_agent.py
│       └── schedule/              # Scheduling (DB connection)
├── streamlit_app/
│   └── streamlit_main.py          # Web Interface
├── tests/
│   ├── test_main.py               # Unit Tests
│   └── test_evals.ipynb           # Evaluation Notebook
├── sms_conversations.json         # Labeled training data
├── job_description.txt            # Job description for RAG
├── availability.json              # Interview slots data
├── requirements.txt
└── README.md
```

---

## Usage

### Web Interface (Streamlit)
The recommended way to test the full chat experience:

```bash
streamlit run streamlit_app/streamlit_main.py
```

### Terminal / CLI
For quick debugging of the conversation flow:

```bash
python main.py
```

### Run Tests
```bash
python -m pytest tests/test_main.py -v
```

---

## Evaluation

The project includes comprehensive evaluation using labeled conversation data.

### Metrics
- **Accuracy:** Overall correct predictions
- **Confusion Matrix:** Detailed prediction breakdown
- **Classification Report:** Precision, Recall, F1 per class

### Running Evaluation
Open `tests/test_evals.ipynb` in Jupyter and run all cells to generate:
- `confusion_matrix.png`
- `evaluation_results.csv`

---

## Contact

Or Lavan - [orlavanbs@gmail.com](mailto:orlavanbs@gmail.com)

Project Link: [https://github.com/Orlavan/TechFlow-AI-Recruiter](https://github.com/Orlavan/TechFlow-AI-Recruiter)

---

## Acknowledgments

* Technion GenAI & LLM Course Staff - [Amir Ben-Haim](https://cont-edu.technion.ac.il/staff/lecturers/%D7%90%D7%9E%D7%99%D7%A8-%D7%91%D7%9F-%D7%97%D7%99%D7%99%D7%9D/)
* [LangChain](https://python.langchain.com/)
* [Streamlit](https://docs.streamlit.io/)
* [ChromaDB](https://docs.trychroma.com/)
* [MongoDB](https://www.mongodb.com/)
* [OpenAI](https://platform.openai.com/)
