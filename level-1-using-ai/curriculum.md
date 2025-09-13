# 2025-09-13 23:45:00

# Level 1: Using AI - Foundations of AI Engineering

## Course Overview
This module introduces undergraduate students to the fundamental concepts and practical skills needed to begin working with AI systems. Students will learn how to effectively interact with AI models, understand their capabilities and limitations, and build their first AI-powered applications.

## Learning Objectives
By the end of this module, students will be able to:
1. Master prompt engineering techniques for optimal AI interactions
2. Integrate AI APIs into applications
3. Understand and manipulate key model parameters
4. Build practical AI-powered solutions
5. Evaluate AI outputs critically

## Module 1: Introduction to Prompt Engineering (Week 1-2)

### 1.1 Understanding Prompts
- What is a prompt?
- The role of prompts in AI systems
- How language models process prompts
- Token concepts and context windows

### 1.2 Zero-Shot Prompting
Zero-shot prompting involves asking the model to perform a task without providing examples.

**Basic Structure:**
```
Task: [Clear description of what you want]
Input: [The data to process]
Output: [Expected format]
```

**Example:**
```
Task: Classify the sentiment of this review
Input: "The product arrived late and was damaged"
Output: Sentiment classification (positive/negative/neutral)
```

### 1.3 Few-Shot Prompting
Few-shot prompting provides examples to guide the model's behavior.

**Structure:**
```
Example 1:
Input: [example input]
Output: [example output]

Example 2:
Input: [example input]
Output: [example output]

Now process:
Input: [actual input]
Output:
```

### 1.4 Chain-of-Thought (CoT) Prompting
CoT prompting encourages step-by-step reasoning.

**Standard CoT:**
```
Question: [Problem]
Let's think step by step:
```

**Advanced Techniques (2024):**
- **Tree of Thoughts (ToT)**: Non-linear reasoning paths
- **Graph of Thoughts (GoT)**: Complex interconnected reasoning
- **Thread of Thought (ThoT)**: Systematic context segmentation
- **Self-Consistency**: Generate multiple reasoning chains and select the most consistent

### 1.5 Advanced Prompting Patterns

**Instance-Adaptive Prompting:**
Tailoring prompts to specific instance characteristics for improved performance.

**Structured Guidance Templates:**
```python
template = """
Context: {context}
Task: {task_description}
Constraints: {constraints}
Expected Output Format: {format}

Input: {input_data}
Output:
"""
```

## Module 2: Working with AI APIs (Week 3-4)

### 2.1 Understanding AI Service Providers

**Major Providers (2025):**

| Provider | Key Models | Context Window | Unique Features |
|----------|------------|----------------|-----------------|
| OpenAI | GPT-4o, o1 | 128k-200k | Multimodal, SORA video |
| Anthropic | Claude 3.5 | 200k | Safety-focused, Visual PDF |
| Cohere | Command R7B | Variable | Enterprise RAG, Multilingual |
| Hugging Face | Open models | Variable | Open-source hub |

### 2.2 API Integration Fundamentals

**Basic API Call Structure:**
```python
import openai
from anthropic import Anthropic
import cohere

# OpenAI Example
openai_client = openai.OpenAI(api_key="your-key")
response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Explain quantum computing"}
    ],
    temperature=0.7,
    max_tokens=500
)

# Anthropic Example
anthropic_client = Anthropic(api_key="your-key")
response = anthropic_client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ],
    max_tokens=500
)

# Cohere Example
co = cohere.Client('your-key')
response = co.generate(
    model='command',
    prompt='Explain quantum computing',
    max_tokens=500,
    temperature=0.7
)
```

### 2.3 Understanding Tokens and Context Windows

**Token Basics:**
- Tokens are chunks of text (roughly 4 characters or 0.75 words)
- Context window = maximum tokens the model can process
- Input tokens + Output tokens ≤ Context window

**Token Counting:**
```python
import tiktoken

def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Example usage
text = "Hello, how are you today?"
token_count = count_tokens(text)
print(f"Token count: {token_count}")
```

### 2.4 Model Parameters

**Temperature (0.0 - 2.0):**
- Lower (0.0-0.3): Deterministic, focused responses
- Medium (0.4-0.7): Balanced creativity and coherence
- Higher (0.8-2.0): Creative, diverse outputs

**Top-p (Nucleus Sampling):**
- Controls cumulative probability of token selection
- 0.1: Very focused (top 10% of probability mass)
- 0.9: More diverse (top 90% of probability mass)

**Example Parameter Tuning:**
```python
# Creative writing task
creative_response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a short story"}],
    temperature=1.2,
    top_p=0.95
)

# Technical documentation task
technical_response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Document this API"}],
    temperature=0.2,
    top_p=0.1
)
```

## Module 3: Practical Applications (Week 5-6)

### 3.1 Building Your First AI Application

**Project: Intelligent Document Summarizer**

```python
import openai
from typing import List, Dict
import asyncio

class DocumentSummarizer:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    def chunk_document(self, text: str, chunk_size: int = 2000) -> List[str]:
        """Split document into manageable chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)
        return chunks

    def summarize_chunk(self, chunk: str) -> str:
        """Summarize a single chunk"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Summarize the following text concisely"},
                {"role": "user", "content": chunk}
            ],
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content

    def create_final_summary(self, summaries: List[str]) -> str:
        """Combine chunk summaries into final summary"""
        combined = "\n".join(summaries)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Create a coherent summary from these partial summaries"},
                {"role": "user", "content": combined}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content

    def summarize(self, document: str) -> str:
        """Main summarization pipeline"""
        chunks = self.chunk_document(document)
        summaries = [self.summarize_chunk(chunk) for chunk in chunks]
        return self.create_final_summary(summaries)
```

### 3.2 Error Handling and Rate Limiting

```python
import time
from typing import Optional
import backoff

class RateLimitedAPIClient:
    def __init__(self, api_key: str, requests_per_minute: int = 60):
        self.client = openai.OpenAI(api_key=api_key)
        self.requests_per_minute = requests_per_minute
        self.request_times = []

    def _wait_if_needed(self):
        """Implement rate limiting"""
        now = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]

        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0]) + 0.1
            time.sleep(sleep_time)

    @backoff.on_exception(
        backoff.expo,
        openai.RateLimitError,
        max_tries=5
    )
    def complete(self, prompt: str, **kwargs) -> Optional[str]:
        """Make API call with rate limiting and retry logic"""
        self._wait_if_needed()

        try:
            response = self.client.chat.completions.create(
                model=kwargs.get('model', 'gpt-4o'),
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            self.request_times.append(time.time())
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            return None
```

### 3.3 Cost Optimization Strategies

**1. Model Selection:**
```python
def select_model_by_task(task_complexity: str) -> str:
    """Choose appropriate model based on task complexity"""
    model_map = {
        "simple": "gpt-3.5-turbo",  # Classification, simple Q&A
        "moderate": "gpt-4o-mini",   # Summarization, basic analysis
        "complex": "gpt-4o",         # Complex reasoning, creative tasks
        "specialized": "o1-preview"  # Advanced reasoning, math
    }
    return model_map.get(task_complexity, "gpt-4o-mini")
```

**2. Caching Responses:**
```python
import hashlib
import json
from typing import Dict, Optional

class ResponseCache:
    def __init__(self):
        self.cache: Dict[str, str] = {}

    def _get_key(self, prompt: str, params: dict) -> str:
        """Generate cache key from prompt and parameters"""
        cache_str = json.dumps({"prompt": prompt, **params}, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    def get(self, prompt: str, params: dict) -> Optional[str]:
        """Retrieve cached response"""
        key = self._get_key(prompt, params)
        return self.cache.get(key)

    def set(self, prompt: str, params: dict, response: str):
        """Cache a response"""
        key = self._get_key(prompt, params)
        self.cache[key] = response
```

## Module 4: Best Practices and Evaluation (Week 7-8)

### 4.1 Prompt Engineering Best Practices

**1. Be Specific and Clear:**
```python
# Poor prompt
"Write about AI"

# Better prompt
"Write a 500-word technical article explaining how transformer neural networks work,
targeting undergraduate computer science students. Include a simple analogy and
avoid mathematical notation."
```

**2. Use Structured Formats:**
```python
structured_prompt = """
Task: Generate a product review analysis
Format: JSON

Input: {review_text}

Output should include:
- sentiment: positive/negative/neutral
- key_points: list of main points
- rating_prediction: 1-5 stars
- improvement_suggestions: list of suggestions
"""
```

**3. Implement Verification:**
```python
def verify_json_output(response: str) -> bool:
    """Verify that AI output is valid JSON"""
    try:
        json.loads(response)
        return True
    except json.JSONDecodeError:
        return False

def get_verified_json_response(client, prompt: str, max_retries: int = 3):
    """Get JSON response with verification"""
    for attempt in range(max_retries):
        response = client.complete(
            prompt + "\nReturn only valid JSON.",
            temperature=0.2
        )
        if verify_json_output(response):
            return json.loads(response)
    return None
```

### 4.2 Evaluation Metrics

**1. Response Quality Assessment:**
```python
def evaluate_response_quality(response: str, criteria: Dict[str, float]) -> float:
    """
    Evaluate response based on multiple criteria
    criteria = {
        "relevance": 0.3,
        "accuracy": 0.3,
        "completeness": 0.2,
        "clarity": 0.2
    }
    """
    # This would typically involve another AI call or human evaluation
    # For demonstration, we'll show the structure
    scores = {}
    total_score = 0

    for criterion, weight in criteria.items():
        # Get score for this criterion (0-1)
        score = evaluate_criterion(response, criterion)
        scores[criterion] = score
        total_score += score * weight

    return total_score
```

**2. A/B Testing Framework:**
```python
class ABTestFramework:
    def __init__(self):
        self.results = {"A": [], "B": []}

    def test_prompts(self, prompt_a: str, prompt_b: str,
                     test_inputs: List[str], client):
        """Compare two different prompts"""
        for input_text in test_inputs:
            # Test prompt A
            response_a = client.complete(prompt_a.format(input=input_text))
            score_a = self.evaluate_response(response_a)
            self.results["A"].append(score_a)

            # Test prompt B
            response_b = client.complete(prompt_b.format(input=input_text))
            score_b = self.evaluate_response(response_b)
            self.results["B"].append(score_b)

    def get_winner(self) -> str:
        """Determine which prompt performs better"""
        avg_a = sum(self.results["A"]) / len(self.results["A"])
        avg_b = sum(self.results["B"]) / len(self.results["B"])
        return "A" if avg_a > avg_b else "B"
```

## Practical Exercises

### Exercise 1: Prompt Engineering Challenge
Create prompts for the following scenarios and compare their effectiveness:
1. Generate a technical tutorial
2. Solve a coding problem
3. Analyze sentiment in customer reviews
4. Create marketing copy

### Exercise 2: API Integration Project
Build a complete application that:
1. Accepts user input
2. Processes it through multiple AI calls
3. Implements proper error handling
4. Includes rate limiting
5. Caches responses for efficiency

### Exercise 3: Parameter Tuning Lab
Experiment with different parameter settings for:
1. Creative writing tasks
2. Technical documentation
3. Code generation
4. Data analysis

Document how each parameter affects output quality.

### Exercise 4: Cost Analysis Project
1. Calculate the cost of different approaches to the same problem
2. Implement caching and measure cost savings
3. Compare different models for cost-effectiveness
4. Create a cost optimization strategy

## Assessment Criteria

### Knowledge Assessment (30%)
- Understanding of prompt engineering techniques
- Knowledge of API capabilities and limitations
- Understanding of token economics and context windows

### Practical Skills (40%)
- Ability to write effective prompts
- Successful API integration
- Proper error handling implementation
- Cost-conscious development

### Project Work (30%)
- Completeness of implementation
- Code quality and documentation
- Innovation and problem-solving
- Performance optimization

## Resources and References

### Documentation
- OpenAI API Documentation: https://platform.openai.com/docs
- Anthropic Claude Documentation: https://docs.anthropic.com
- Cohere API Reference: https://docs.cohere.com

### Research Papers (2024)
- "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
- "Chain-of-Table: Evolving Tables in the Reasoning Chain"
- "Thread of Thought: Unraveling Chaotic Contexts"

### Tools and Libraries
- tiktoken: Token counting for OpenAI models
- langchain: Framework for LLM applications
- promptfoo: Prompt testing and evaluation

## Summary
Level 1 provides students with foundational skills in AI engineering, focusing on practical application rather than theoretical knowledge. Students learn to effectively interact with AI systems, integrate them into applications, and evaluate their outputs critically. This foundation prepares them for more advanced topics in subsequent levels.

# 2025-09-13 23:45:00