# 2025-09-13 23:50:00

# Level 2: Integrating AI - Building Intelligent Systems

## Course Overview
This module advances students from basic AI usage to building integrated AI systems. Students will learn to create sophisticated applications using Retrieval Augmented Generation (RAG), vector databases, embeddings, and intelligent agents. This level focuses on production-ready implementations that solve real-world problems.

## Learning Objectives
By the end of this module, students will be able to:
1. Design and implement RAG systems for enhanced AI capabilities
2. Work with vector databases and embedding models
3. Build intelligent agents with tool use capabilities
4. Optimize AI systems for cost and performance
5. Create production-ready AI applications

## Module 1: Retrieval Augmented Generation (RAG) Fundamentals (Week 1-2)

### 1.1 Understanding RAG Architecture

RAG combines the power of retrieval systems with generative AI to provide accurate, contextual responses grounded in specific knowledge bases.

**Core Components:**
```
1. Document Ingestion → 2. Chunking → 3. Embedding → 4. Vector Storage
                                                            ↓
7. Generation ← 6. Context Formation ← 5. Retrieval
```

### 1.2 Document Processing Pipeline

```python
from typing import List, Dict
import hashlib
from dataclasses import dataclass

@dataclass
class Document:
    content: str
    metadata: Dict
    doc_id: str = None

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: Document) -> List[Document]:
        """Split document into overlapping chunks"""
        text = document.content
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Find natural break point (end of sentence)
            if end < len(text):
                # Look for sentence endings
                for char in ['. ', '? ', '! ', '\n\n']:
                    last_period = text.rfind(char, start, end)
                    if last_period != -1:
                        end = last_period + 1
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = Document(
                    content=chunk_text,
                    metadata={
                        **document.metadata,
                        'chunk_index': len(chunks),
                        'parent_doc_id': document.doc_id,
                        'start_char': start,
                        'end_char': end
                    }
                )
                chunks.append(chunk)

            start = end - self.chunk_overlap if end < len(text) else end

        return chunks

    def prepare_documents(self, documents: List[Document]) -> List[Document]:
        """Process multiple documents for RAG"""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks
```

### 1.3 Advanced Chunking Strategies

**Semantic Chunking:**
```python
class SemanticChunker:
    def __init__(self, embedding_model, similarity_threshold: float = 0.8):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

    def chunk_by_similarity(self, text: str) -> List[str]:
        """Create chunks based on semantic similarity"""
        sentences = self._split_into_sentences(text)
        embeddings = [self.embedding_model.encode(s) for s in sentences]

        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]

        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(current_embedding, embeddings[i])

            if similarity < self.similarity_threshold:
                # Start new chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
                current_embedding = embeddings[i]
            else:
                # Add to current chunk
                current_chunk.append(sentences[i])
                # Update embedding (average)
                current_embedding = (current_embedding + embeddings[i]) / 2

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between vectors"""
        import numpy as np
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
```

## Module 2: Vector Databases and Embeddings (Week 3-4)

### 2.1 Understanding Embeddings

Embeddings convert text into high-dimensional vectors that capture semantic meaning.

**Embedding Models Comparison (2024):**

| Model | Dimensions | Performance | Use Case |
|-------|------------|-------------|----------|
| OpenAI text-embedding-3-large | 3072 | Highest quality | Production systems |
| OpenAI text-embedding-3-small | 1536 | Good quality, faster | Cost-sensitive apps |
| Cohere embed-v3 | 1024 | Multilingual strong | Global applications |
| BAAI/bge-large-en-v1.5 | 1024 | Best open-source | Self-hosted systems |

### 2.2 Implementing Embedding Pipeline

```python
from typing import List, Union
import numpy as np
from openai import OpenAI
import cohere

class EmbeddingService:
    def __init__(self, provider: str = "openai", api_key: str = None):
        self.provider = provider
        if provider == "openai":
            self.client = OpenAI(api_key=api_key)
            self.model = "text-embedding-3-large"
        elif provider == "cohere":
            self.client = cohere.Client(api_key)
            self.model = "embed-english-v3.0"

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text"""
        if isinstance(text, str):
            text = [text]

        if self.provider == "openai":
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embeddings = [e.embedding for e in response.data]

        elif self.provider == "cohere":
            response = self.client.embed(
                texts=text,
                model=self.model,
                input_type="search_document"
            )
            embeddings = response.embeddings

        return np.array(embeddings)

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Embed large batches of text efficiently"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embed_text(batch)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)
```

### 2.3 Vector Database Implementation

**Working with Popular Vector Databases:**

```python
# Pinecone Implementation
import pinecone
from pinecone import Pinecone, ServerlessSpec

class PineconeVectorStore:
    def __init__(self, api_key: str, environment: str):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = "rag-index"

    def create_index(self, dimension: int = 1536):
        """Create a new Pinecone index"""
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        self.index = self.pc.Index(self.index_name)

    def upsert_vectors(self, vectors: List[tuple]):
        """Insert or update vectors"""
        # vectors = [(id, embedding, metadata), ...]
        self.index.upsert(vectors=vectors)

    def search(self, query_vector: List[float], top_k: int = 5):
        """Search for similar vectors"""
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        return results.matches

# Weaviate Implementation
import weaviate
from weaviate.embedded import EmbeddedOptions

class WeaviateVectorStore:
    def __init__(self):
        self.client = weaviate.Client(
            embedded_options=EmbeddedOptions()
        )
        self.class_name = "Document"

    def create_schema(self):
        """Create Weaviate schema"""
        schema = {
            "class": self.class_name,
            "vectorizer": "none",  # We'll provide our own embeddings
            "properties": [
                {"name": "content", "dataType": ["text"]},
                {"name": "metadata", "dataType": ["text"]},
                {"name": "doc_id", "dataType": ["string"]}
            ]
        }
        self.client.schema.create_class(schema)

    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        """Add documents with embeddings"""
        with self.client.batch as batch:
            for doc, embedding in zip(documents, embeddings):
                batch.add_data_object(
                    data_object=doc,
                    class_name=self.class_name,
                    vector=embedding
                )

    def search(self, query_vector: List[float], limit: int = 5):
        """Perform vector search"""
        result = self.client.query.get(
            self.class_name,
            ["content", "metadata", "doc_id"]
        ).with_near_vector({
            "vector": query_vector
        }).with_limit(limit).do()

        return result["data"]["Get"][self.class_name]

# FAISS Implementation (Local)
import faiss
import pickle

class FAISSVectorStore:
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

    def add_vectors(self, embeddings: np.ndarray, documents: List[Dict]):
        """Add vectors to FAISS index"""
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(documents)

    def search(self, query_vector: np.ndarray, k: int = 5):
        """Search for similar vectors"""
        query_vector = query_vector.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_vector, k)

        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'distance': float(dist),
                    'similarity': 1 / (1 + float(dist))  # Convert distance to similarity
                })

        return results

    def save(self, path: str):
        """Save index to disk"""
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.docs", 'wb') as f:
            pickle.dump(self.documents, f)

    def load(self, path: str):
        """Load index from disk"""
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.docs", 'rb') as f:
            self.documents = pickle.load(f)
```

### 2.4 Hybrid Search Implementation

Combining vector search with keyword search for improved retrieval:

```python
class HybridSearchSystem:
    def __init__(self, vector_store, keyword_index):
        self.vector_store = vector_store
        self.keyword_index = keyword_index

    def search(self, query: str, alpha: float = 0.7, top_k: int = 10):
        """
        Perform hybrid search
        alpha: weight for vector search (1-alpha for keyword search)
        """
        # Vector search
        query_embedding = self.embed_query(query)
        vector_results = self.vector_store.search(query_embedding, top_k * 2)

        # Keyword search
        keyword_results = self.keyword_index.search(query, top_k * 2)

        # Reciprocal Rank Fusion (RRF)
        combined_scores = {}
        k = 60  # RRF constant

        # Process vector results
        for rank, result in enumerate(vector_results):
            doc_id = result['doc_id']
            score = alpha * (1 / (k + rank + 1))
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + score

        # Process keyword results
        for rank, result in enumerate(keyword_results):
            doc_id = result['doc_id']
            score = (1 - alpha) * (1 / (k + rank + 1))
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + score

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return sorted_results
```

## Module 3: Building Intelligent Agents (Week 5-6)

### 3.1 Agent Architecture

```python
from typing import List, Dict, Any, Callable
from enum import Enum
import json

class ToolType(Enum):
    RETRIEVAL = "retrieval"
    CALCULATION = "calculation"
    WEB_SEARCH = "web_search"
    DATABASE = "database"
    API_CALL = "api_call"

class Tool:
    def __init__(self, name: str, description: str,
                 function: Callable, parameters: Dict):
        self.name = name
        self.description = description
        self.function = function
        self.parameters = parameters

    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters"""
        return self.function(**kwargs)

class Agent:
    def __init__(self, llm_client, tools: List[Tool]):
        self.llm_client = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.conversation_history = []

    def plan(self, task: str) -> List[Dict]:
        """Create an execution plan for the task"""
        planning_prompt = f"""
        Task: {task}

        Available tools:
        {self._format_tools()}

        Create a step-by-step plan to accomplish this task.
        Return a JSON array of steps, each with:
        - step_number: int
        - description: string
        - tool: string (tool name to use)
        - parameters: object (parameters for the tool)
        """

        response = self.llm_client.complete(
            planning_prompt,
            temperature=0.3
        )

        return json.loads(response)

    def execute_plan(self, plan: List[Dict]) -> str:
        """Execute a plan step by step"""
        results = []

        for step in plan:
            tool_name = step['tool']
            parameters = step['parameters']

            if tool_name in self.tools:
                try:
                    result = self.tools[tool_name].execute(**parameters)
                    results.append({
                        'step': step['step_number'],
                        'result': result,
                        'status': 'success'
                    })
                except Exception as e:
                    results.append({
                        'step': step['step_number'],
                        'error': str(e),
                        'status': 'failed'
                    })
            else:
                results.append({
                    'step': step['step_number'],
                    'error': f"Tool {tool_name} not found",
                    'status': 'failed'
                })

        return self._synthesize_results(results)

    def _format_tools(self) -> str:
        """Format tools for LLM understanding"""
        tool_descriptions = []
        for name, tool in self.tools.items():
            tool_descriptions.append(
                f"- {name}: {tool.description}\n"
                f"  Parameters: {json.dumps(tool.parameters)}"
            )
        return "\n".join(tool_descriptions)

    def _synthesize_results(self, results: List[Dict]) -> str:
        """Synthesize execution results into final response"""
        synthesis_prompt = f"""
        Synthesize these execution results into a coherent response:
        {json.dumps(results, indent=2)}

        Provide a clear, comprehensive answer based on the results.
        """

        return self.llm_client.complete(synthesis_prompt)
```

### 3.2 Implementing RAG Agent

```python
class RAGAgent(Agent):
    def __init__(self, llm_client, vector_store, embedding_service):
        self.vector_store = vector_store
        self.embedding_service = embedding_service

        # Define RAG-specific tools
        tools = [
            Tool(
                name="search_knowledge_base",
                description="Search the knowledge base for relevant information",
                function=self.search_knowledge_base,
                parameters={"query": "string", "top_k": "integer"}
            ),
            Tool(
                name="extract_information",
                description="Extract specific information from documents",
                function=self.extract_information,
                parameters={"documents": "array", "question": "string"}
            ),
            Tool(
                name="summarize_documents",
                description="Summarize multiple documents",
                function=self.summarize_documents,
                parameters={"documents": "array", "max_length": "integer"}
            )
        ]

        super().__init__(llm_client, tools)

    def search_knowledge_base(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search vector store for relevant documents"""
        query_embedding = self.embedding_service.embed_text(query)
        results = self.vector_store.search(query_embedding[0], top_k)
        return [{'content': r['document']['content'],
                'metadata': r['document']['metadata']} for r in results]

    def extract_information(self, documents: List[Dict], question: str) -> str:
        """Extract specific information from documents"""
        context = "\n\n".join([doc['content'] for doc in documents])
        prompt = f"""
        Context: {context}

        Question: {question}

        Extract and provide the specific information requested.
        If the information is not available, say so clearly.
        """
        return self.llm_client.complete(prompt)

    def summarize_documents(self, documents: List[Dict], max_length: int = 500) -> str:
        """Summarize multiple documents"""
        combined_content = "\n\n".join([doc['content'] for doc in documents])
        prompt = f"""
        Summarize the following documents in {max_length} words or less:

        {combined_content}

        Provide a comprehensive summary that captures the key points.
        """
        return self.llm_client.complete(prompt)

    def answer_question(self, question: str) -> str:
        """Main RAG pipeline for answering questions"""
        # Step 1: Retrieve relevant documents
        documents = self.search_knowledge_base(question, top_k=5)

        if not documents:
            return "I couldn't find relevant information to answer your question."

        # Step 2: Generate answer with context
        context = "\n\n".join([doc['content'] for doc in documents[:3]])

        prompt = f"""
        Based on the following context, answer the question.
        If the answer is not in the context, say you don't know.

        Context:
        {context}

        Question: {question}

        Answer:
        """

        answer = self.llm_client.complete(prompt, temperature=0.3)

        # Step 3: Add citations
        citations = [doc['metadata'].get('source', 'Unknown') for doc in documents[:3]]
        answer_with_citations = f"{answer}\n\nSources: {', '.join(set(citations))}"

        return answer_with_citations
```

### 3.3 Advanced Agent Patterns

**ReAct (Reasoning and Acting) Agent:**

```python
class ReActAgent:
    def __init__(self, llm_client, tools: Dict[str, Callable]):
        self.llm_client = llm_client
        self.tools = tools
        self.max_iterations = 10

    def run(self, task: str) -> str:
        """Execute task using ReAct pattern"""
        thought_history = []
        action_history = []

        for i in range(self.max_iterations):
            # Generate thought
            thought = self._generate_thought(task, thought_history, action_history)
            thought_history.append(thought)

            # Check if task is complete
            if "FINAL ANSWER:" in thought:
                return thought.split("FINAL ANSWER:")[1].strip()

            # Generate action
            action = self._generate_action(thought)

            if action:
                # Execute action
                result = self._execute_action(action)
                action_history.append({
                    'action': action,
                    'result': result
                })

        return "Maximum iterations reached without finding answer"

    def _generate_thought(self, task: str, thoughts: List[str],
                         actions: List[Dict]) -> str:
        """Generate reasoning thought"""
        prompt = f"""
        Task: {task}

        Previous thoughts: {thoughts}
        Previous actions: {actions}

        What should I think about next?
        If you have the final answer, start with "FINAL ANSWER:"
        """
        return self.llm_client.complete(prompt)

    def _generate_action(self, thought: str) -> Dict:
        """Generate action based on thought"""
        prompt = f"""
        Based on this thought: {thought}

        Available tools: {list(self.tools.keys())}

        What action should I take? Return JSON:
        {{"tool": "tool_name", "parameters": {{}}}}

        Or return null if no action needed.
        """
        response = self.llm_client.complete(prompt)
        return json.loads(response) if response != "null" else None

    def _execute_action(self, action: Dict) -> Any:
        """Execute the specified action"""
        tool_name = action['tool']
        parameters = action['parameters']

        if tool_name in self.tools:
            return self.tools[tool_name](**parameters)
        return f"Error: Tool {tool_name} not found"
```

## Module 4: Optimization Techniques (Week 7-8)

### 4.1 Caching Strategies

```python
import redis
import hashlib
import json
from datetime import timedelta
from functools import wraps

class CacheManager:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )

    def cache_key(self, prefix: str, params: Dict) -> str:
        """Generate cache key from parameters"""
        param_str = json.dumps(params, sort_keys=True)
        hash_val = hashlib.md5(param_str.encode()).hexdigest()
        return f"{prefix}:{hash_val}"

    def cache_embedding(self, text: str, embedding: List[float],
                       ttl: int = 86400):
        """Cache text embedding"""
        key = self.cache_key("embedding", {"text": text})
        self.redis_client.setex(
            key,
            timedelta(seconds=ttl),
            json.dumps(embedding)
        )

    def get_cached_embedding(self, text: str) -> List[float]:
        """Retrieve cached embedding"""
        key = self.cache_key("embedding", {"text": text})
        cached = self.redis_client.get(key)
        return json.loads(cached) if cached else None

    def cache_llm_response(self, prompt: str, response: str,
                          model: str, ttl: int = 3600):
        """Cache LLM response"""
        key = self.cache_key("llm", {"prompt": prompt, "model": model})
        self.redis_client.setex(
            key,
            timedelta(seconds=ttl),
            response
        )

    def get_cached_llm_response(self, prompt: str, model: str) -> str:
        """Retrieve cached LLM response"""
        key = self.cache_key("llm", {"prompt": prompt, "model": model})
        return self.redis_client.get(key)

def with_cache(cache_manager: CacheManager, ttl: int = 3600):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = cache_manager.cache_key(
                func.__name__,
                {"args": str(args), "kwargs": str(kwargs)}
            )

            # Check cache
            cached_result = cache_manager.redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            cache_manager.redis_client.setex(
                cache_key,
                timedelta(seconds=ttl),
                json.dumps(result)
            )

            return result
        return wrapper
    return decorator
```

### 4.2 Batch Processing

```python
import asyncio
from typing import List, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class BatchProcessor:
    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def process_batch_async(self, items: List[Any],
                                 process_func: Callable) -> List[Any]:
        """Process items in batches asynchronously"""
        results = []

        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = await asyncio.gather(
                *[self._process_item_async(item, process_func) for item in batch]
            )
            results.extend(batch_results)

        return results

    async def _process_item_async(self, item: Any,
                                  process_func: Callable) -> Any:
        """Process single item asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, process_func, item)

    def process_embeddings_batch(self, texts: List[str],
                                embedding_service) -> np.ndarray:
        """Efficiently process embeddings in batches"""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # Check cache first
            cached_embeddings = []
            texts_to_embed = []
            indices_to_embed = []

            for j, text in enumerate(batch):
                cached = cache_manager.get_cached_embedding(text)
                if cached:
                    cached_embeddings.append((j, cached))
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(j)

            # Generate new embeddings
            if texts_to_embed:
                new_embeddings = embedding_service.embed_text(texts_to_embed)

                # Cache new embeddings
                for text, embedding in zip(texts_to_embed, new_embeddings):
                    cache_manager.cache_embedding(text, embedding.tolist())

            # Combine cached and new embeddings in correct order
            batch_embeddings = [None] * len(batch)
            for idx, embedding in cached_embeddings:
                batch_embeddings[idx] = embedding

            for idx, embedding in zip(indices_to_embed, new_embeddings):
                batch_embeddings[idx] = embedding

            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)
```

### 4.3 Cost Optimization

```python
class CostOptimizer:
    def __init__(self):
        self.model_costs = {
            "gpt-4o": {"input": 0.005, "output": 0.015},  # per 1K tokens
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
            "text-embedding-3-large": {"input": 0.00013, "output": 0},
            "text-embedding-3-small": {"input": 0.00002, "output": 0}
        }

    def estimate_cost(self, model: str, input_tokens: int,
                     output_tokens: int) -> float:
        """Estimate cost for API call"""
        if model not in self.model_costs:
            return 0

        costs = self.model_costs[model]
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]

        return input_cost + output_cost

    def select_optimal_model(self, task_type: str,
                           quality_threshold: float = 0.8) -> str:
        """Select most cost-effective model for task"""
        model_selection = {
            "simple_classification": "gpt-4o-mini",
            "complex_reasoning": "gpt-4o",
            "creative_writing": "claude-3-5-sonnet",
            "embeddings_high": "text-embedding-3-large",
            "embeddings_standard": "text-embedding-3-small"
        }

        return model_selection.get(task_type, "gpt-4o-mini")

    def optimize_rag_pipeline(self, documents: List[str],
                            query: str) -> Dict:
        """Optimize RAG pipeline for cost"""
        optimization_strategy = {
            "embedding_model": "text-embedding-3-small",  # Lower cost
            "reranking": True,  # Use cheaper model for initial retrieval
            "generation_model": "gpt-4o-mini",  # Start with cheaper model
            "fallback_model": "gpt-4o"  # Use for complex queries
        }

        # Estimate query complexity
        query_complexity = self._estimate_complexity(query)

        if query_complexity > 0.7:
            optimization_strategy["generation_model"] = "gpt-4o"
            optimization_strategy["embedding_model"] = "text-embedding-3-large"

        return optimization_strategy

    def _estimate_complexity(self, query: str) -> float:
        """Estimate query complexity (0-1)"""
        # Simple heuristic based on query characteristics
        complexity_factors = {
            "length": len(query.split()) / 50,
            "technical_terms": self._count_technical_terms(query) / 10,
            "question_words": len([w for w in ["why", "how", "explain", "analyze"]
                                 if w in query.lower()]) / 4
        }

        return min(1.0, sum(complexity_factors.values()) / len(complexity_factors))

    def _count_technical_terms(self, text: str) -> int:
        """Count technical terms in text"""
        technical_terms = ["algorithm", "optimization", "neural", "vector",
                         "embedding", "transformer", "gradient", "backpropagation"]
        return sum(1 for term in technical_terms if term in text.lower())
```

## Practical Projects

### Project 1: Complete RAG System
Build an end-to-end RAG system that:
1. Ingests documents from multiple sources
2. Implements intelligent chunking
3. Uses hybrid search
4. Includes reranking
5. Provides cited responses

### Project 2: Multi-Tool Agent
Create an agent that can:
1. Search the web
2. Query databases
3. Perform calculations
4. Access RAG knowledge base
5. Generate reports

### Project 3: Cost-Optimized Pipeline
Implement a pipeline that:
1. Monitors API costs
2. Caches intelligently
3. Selects models dynamically
4. Batches requests
5. Falls back gracefully

## Assessment Rubric

### Technical Implementation (40%)
- Code quality and organization
- Proper error handling
- Performance optimization
- Documentation

### System Design (30%)
- Architecture decisions
- Scalability considerations
- Cost awareness
- Security implementation

### Innovation (20%)
- Creative problem solving
- Advanced feature implementation
- Optimization strategies

### Testing and Evaluation (10%)
- Test coverage
- Performance benchmarks
- Cost analysis
- User evaluation

## Resources

### Vector Databases
- Pinecone Documentation
- Weaviate Tutorials
- FAISS GitHub Repository
- Milvus Best Practices

### Embedding Models
- OpenAI Embeddings Guide
- Cohere Semantic Search
- Sentence Transformers Library
- MTEB Leaderboard

### Agent Frameworks
- LangChain Documentation
- LlamaIndex Guides
- AutoGPT Repository
- ReAct Paper

## Summary
Level 2 transforms students from AI users to AI system builders. They learn to create sophisticated applications that combine retrieval systems with generative AI, implement intelligent agents, and optimize for production deployment. This foundation enables them to build real-world AI solutions that are accurate, efficient, and cost-effective.

# 2025-09-13 23:50:00