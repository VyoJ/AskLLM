# ASK Evaluation API

## Overview

The ASK Evaluation API is a FastAPI-based backend service that helps entrepreneurs refine and improve their "asks" when collaborating with stakeholders. Using Groq's Llama3.1-70b language model, it analyzes and enhances requests based on effectual reasoning principles and proven communication strategies.

## Features

- Evaluates and improves entrepreneurial asks using AI
- Configurable system prompts and examples
- Based on five key principles of effectual reasoning:
  1. Bird in Hand (Clarify Your Goals)
  2. Affordable Loss
  3. Crazy Quilt (Partnership-Oriented)
  4. Lemonade Principle (Embrace Surprises)
  5. Pilot-in-the-Plane (Control Your Future)

## Prerequisites

- Python 3.8+
- Groq API key
- FastAPI
- Uvicorn

## Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd askllm
```

2. Install dependencies:

```bash
pip install fastapi uvicorn groq pydantic
```

3. Set up your environment variables:

```bash
export GROQ_API_KEY="your-api-key-here"
```

## Running the API

Start the server with:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Evaluate Ask

```http
POST /evaluate-ask
```

Evaluates and improves an entrepreneurial ask.

**Request Body:**

```json
{
  "about_me": "string",
  "about_stakeholder": "string",
  "ask": "string"
}
```

**Response:**

```json
{
  "evaluation": "string",
  "status": "success"
}
```

### 2. Update System Prompt

```http
PUT /update-system-prompt
```

Updates the system prompt used for evaluation.

**Request Body:**

```json
{
  "new_prompt": "string"
}
```

### 3. Get System Prompt

```http
GET /get-system-prompt
```

Returns the current system prompt.

### 4. Update Few-Shot Examples

```http
PUT /update-few-shot-examples
```

Updates the few-shot examples used in the evaluation.

**Request Body:**

```json
{
  "examples": [
    {
      "poor_ask": "string",
      "better_ask": "string"
    }
  ]
}
```

### 5. Get Few-Shot Examples

```http
GET /get-few-shot-examples
```

Returns the current few-shot examples.

### 6. Reset Defaults

```http
POST /reset-defaults
```

Resets system prompt and few-shot examples to default values.

## Example Usage

### Evaluating an Ask

```python
import requests

url = "http://localhost:8000/evaluate-ask"
data = {
    "about_me": "I'm the founder of a SaaS startup that provides workflow automation tools for small businesses",
    "about_stakeholder": "The person is the head of a community of tech innovators with a strong background in scaling startups",
    "ask": "Can you introduce me to potential beta customers?"
}

response = requests.post(url, json=data)
print(response.json())
```

### Updating System Prompt

```python
import requests

url = "http://localhost:8000/update-system-prompt"
data = {
    "new_prompt": "Your new system prompt here..."
}

response = requests.put(url, json=data)
print(response.json())
```

## Response Format

The API evaluates asks and provides structured feedback including:

1. Analysis of the current ask using effectual reasoning principles
2. Improved version of the ask
3. Specific suggestions for improvement

## Best Practices

1. Provide detailed context in the `about_me` and `about_stakeholder` fields
2. Keep asks concise but informative
3. Include relevant metrics or achievements when possible
4. Consider updating few-shot examples based on your specific industry or use case

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
