# Contextual-Robust-Question-Answering-Bot

## Project Overview

This project implements a **state-of-the-art extractive Question Answering (QA)** system built upon the powerful capabilities of the **Hugging Face Transformers** library.

It utilizes a **pre-trained RoBERTa model fine-tuned on the SQuAD 2.0 dataset**, chosen for its ability to handle both standard factoid questions and *unanswerable* questions — a major advancement over older benchmarks like SQuAD 1.1.

The core technical enhancement, inspired by the **CoQA dataset**, is the implementation of a **QAEngine** that maintains conversational history, allowing the model to answer **contextual follow-up questions** without requiring the user to restate the full context each time.

---

## Key Features

- Robust QA (SQuAD 2.0):  
  Uses the `deepset/roberta-base-squad2` model, fine-tuned on SQuAD 2.0, enabling the system to detect when *no answer exists* within the given context based on a confidence score threshold.

- Conversational Flow (CoQA Inspired):
  The `qa_engine.py` script manages conversational state and refines follow-up questions, enabling **multi-turn, contextual QA** that mimics natural dialogue.

- Efficient Inference:
  Leverages Hugging Face’s optimized `pipeline` API for **fast and reliable** predictions.

- Organized Structure:
  Clear modular separation between:
  - Core QA logic (`qa_engine.py`)
  - Conversational demonstration (`conversational_demo.py`)

---

## Project Structure

```bash
.
├── README.md
├── requirements.txt
├── qa_engine.py          # Core QA logic and model interface
└── conversational_demo.py # Script to run a sample conversation flow
```

## Setup and Installation

### Clone the Repository
```bash
gh clone lvarshini3096/Contextual-Robust-Question-Answering-Bot
cd Contextual-Robust-Question-Answering-Bot
```

### Create and Activate Environment
We recommend using a Python virtual environment:

```bash
# Create the environment
python -m venv venv

# Activate the environment
source venv/bin/activate      # On Linux/macOS
.\venv\Scripts\activate       # On Windows (or "venv\Scripts\activate" in cmd)
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Usage
To test the full conversational and robustness features of the bot, run the main demonstration script:

```bash
python conversational_demo.py
```

### Expected Demonstration Output
Below is a sample demonstration showing the bot handling multiple question types, demonstrating robustness (Turn 3) and contextual memory (Turn 2):


--- Contextual & Robust Question Answering Demo ---

Loading QA model: deepset/roberta-base-squad2...
Model loaded successfully.

--- New Context Set ---
The first modern computer was the Z1, created by Konrad Zuse in 1938. It was a mechanical calculator...
----------------------

[Turn 1] Direct Question
Q: Who built the Z1?
A: Konrad Zuse (Confidence: 99.85%)

[Turn 2] Contextual Follow-up
Q: When was it rebuilt?
A: 1980s (Confidence: 98.71%)

[Turn 3] Unanswerable Question Test
Q: What material was used for the ENIAC's casing?
A: I cannot find a relevant answer in the provided context (Low confidence).

[Turn 4] Answerable Question on a different fact
Q: How much power did the ENIAC use?
A: 150 kW (Confidence: 99.55%)

--- Demo Complete ---

## Technologies Used
Python 3.8+
Hugging Face Transformers
PyTorch
SQuAD 2.0 Dataset
CoQA-Inspired Conversational Logic

## References
Hugging Face Transformers
SQuAD 2.0 Dataset
CoQA Dataset
