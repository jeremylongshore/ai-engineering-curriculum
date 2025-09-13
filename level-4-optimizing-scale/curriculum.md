# 2025-09-14 00:00:00

# Level 4: Optimizing AI at Scale - Enterprise Deployment and Operations

## Course Overview
This advanced module prepares students to deploy and operate AI systems at enterprise scale. Students will master distributed inference, implement sophisticated optimization strategies, manage compliance and governance, and build systems that balance performance with cost-effectiveness. This level focuses on real-world production challenges and solutions.

## Learning Objectives
By the end of this module, students will be able to:
1. Design and implement distributed AI inference systems
2. Optimize context management and memory utilization
3. Balance cost vs performance tradeoffs effectively
4. Implement comprehensive privacy and compliance measures
5. Deploy and monitor AI systems at enterprise scale

## Module 1: Distributed Inference Architecture (Week 1-2)

### 1.1 Understanding Distributed Inference Frameworks

**Framework Comparison (2024-2025):**

| Framework | Key Features | Best For | Performance |
|-----------|-------------|----------|-------------|
| vLLM | PyTorch ecosystem, FP8 quantization | High throughput | Excellent |
| Ray Serve | Distributed computing, auto-scaling | Complex pipelines | Very Good |
| TGI (Text Generation Inference) | Hugging Face native | Easy deployment | Good |
| Triton Inference Server | Multi-framework support | Heterogeneous models | Excellent |
| Dynamo | Rust-based, engine agnostic | Flexibility | Excellent |

### 1.2 Implementing vLLM for Production

```python
from vllm import LLM, SamplingParams
from vllm.distributed import initialize_distributed_environment
import torch
from typing import List, Dict, Optional
import asyncio

class DistributedInferenceEngine:
    def __init__(self, model_name: str, tensor_parallel_size: int = 2,
                 pipeline_parallel_size: int = 1):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.setup_engine()

    def setup_engine(self):
        """Initialize vLLM with distributed settings"""
        # Configure for distributed inference
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            trust_remote_code=True,
            dtype="float16",  # or "bfloat16" for newer models
            max_model_len=32768,  # Adjust based on model
            gpu_memory_utilization=0.95,  # Maximize GPU usage
            swap_space=4,  # GB of CPU swap space
            enforce_eager=False,  # Use CUDA graphs for speed
            enable_prefix_caching=True,  # Cache common prefixes
            enable_chunked_prefill=True,  # Better handling of long contexts
            max_num_batched_tokens=8192,
            max_num_seqs=256  # Maximum concurrent sequences
        )

        # Default sampling parameters
        self.default_sampling = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
            stop=["</s>", "\n\n"]
        )

    def generate(self, prompts: List[str],
                sampling_params: Optional[SamplingParams] = None) -> List[str]:
        """Generate responses for batch of prompts"""
        if sampling_params is None:
            sampling_params = self.default_sampling

        outputs = self.llm.generate(prompts, sampling_params)

        responses = []
        for output in outputs:
            response = output.outputs[0].text
            responses.append(response)

        return responses

    async def async_generate(self, prompts: List[str]) -> List[str]:
        """Asynchronous generation for better concurrency"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompts)

    def profile_performance(self, test_prompts: List[str]) -> Dict:
        """Profile inference performance"""
        import time
        import numpy as np

        metrics = {
            "throughput": [],
            "latency": [],
            "tokens_per_second": []
        }

        for batch_size in [1, 8, 16, 32, 64]:
            batch = test_prompts[:batch_size]

            start_time = time.time()
            outputs = self.llm.generate(batch, self.default_sampling)
            end_time = time.time()

            total_time = end_time - start_time
            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

            metrics["throughput"].append(batch_size / total_time)
            metrics["latency"].append(total_time / batch_size)
            metrics["tokens_per_second"].append(total_tokens / total_time)

        return {
            "avg_throughput": np.mean(metrics["throughput"]),
            "avg_latency": np.mean(metrics["latency"]),
            "avg_tokens_per_second": np.mean(metrics["tokens_per_second"]),
            "details": metrics
        }
```

### 1.3 Ray Serve Implementation

```python
import ray
from ray import serve
from ray.serve.handle import RayServeHandle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import asyncio

# Initialize Ray
ray.init(address="auto", ignore_reinit_error=True)
serve.start(detached=True)

@serve.deployment(
    num_replicas=4,  # Number of model replicas
    ray_actor_options={
        "num_cpus": 2,
        "num_gpus": 1,
        "memory": 16 * 1024 * 1024 * 1024  # 16GB per replica
    },
    autoscaling_config={
        "min_replicas": 2,
        "max_replicas": 10,
        "target_num_ongoing_requests_per_replica": 5,
        "upscale_delay_s": 10,
        "downscale_delay_s": 60
    },
    health_check_period_s=10,
    health_check_timeout_s=30
)
class ModelServer:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    async def __call__(self, request: Dict) -> Dict:
        """Handle inference request"""
        prompt = request["prompt"]
        params = request.get("params", {})

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=params.get("max_tokens", 256),
                temperature=params.get("temperature", 0.7),
                top_p=params.get("top_p", 0.95),
                do_sample=True
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "response": response,
            "model": self.model.config.name_or_path,
            "tokens_generated": len(outputs[0]) - len(inputs["input_ids"][0])
        }

# Deploy the model
deployment = ModelServer.bind(model_name="meta-llama/Llama-2-7b-chat-hf")
serve.run(deployment, name="llm_service", route_prefix="/generate")

class LoadBalancer:
    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
        self.current_index = 0
        self.health_status = {ep: True for ep in endpoints}
        self.request_counts = {ep: 0 for ep in endpoints}

    async def route_request(self, request: Dict) -> Dict:
        """Route request to healthy endpoint with least load"""
        # Get healthy endpoints sorted by load
        healthy_endpoints = [
            ep for ep in self.endpoints
            if self.health_status[ep]
        ]

        if not healthy_endpoints:
            raise Exception("No healthy endpoints available")

        # Select endpoint with least requests
        selected = min(healthy_endpoints, key=lambda x: self.request_counts[x])
        self.request_counts[selected] += 1

        try:
            # Make request to selected endpoint
            response = await self.make_request(selected, request)
            return response
        except Exception as e:
            # Mark endpoint as unhealthy
            self.health_status[selected] = False
            # Retry with different endpoint
            return await self.route_request(request)
        finally:
            self.request_counts[selected] -= 1

    async def health_check(self):
        """Periodic health check for all endpoints"""
        while True:
            for endpoint in self.endpoints:
                try:
                    # Simple health check
                    await self.make_request(endpoint, {"prompt": "test", "params": {"max_tokens": 1}})
                    self.health_status[endpoint] = True
                except:
                    self.health_status[endpoint] = False

            await asyncio.sleep(30)  # Check every 30 seconds
```

### 1.4 Advanced Batching and Scheduling

```python
import heapq
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time
import threading
from queue import PriorityQueue

@dataclass(order=True)
class InferenceRequest:
    priority: int
    request_id: str = field(compare=False)
    prompt: str = field(compare=False)
    params: Dict = field(compare=False)
    timestamp: float = field(compare=False)
    callback: Any = field(compare=False)

class DynamicBatcher:
    def __init__(self, model, max_batch_size: int = 32,
                 max_wait_time: float = 0.1):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.request_queue = PriorityQueue()
        self.processing = False
        self.start_processing()

    def add_request(self, request: InferenceRequest):
        """Add request to batching queue"""
        request.timestamp = time.time()
        self.request_queue.put(request)

    def start_processing(self):
        """Start batch processing thread"""
        thread = threading.Thread(target=self._process_batches, daemon=True)
        thread.start()

    def _process_batches(self):
        """Process batches continuously"""
        while True:
            batch = self._collect_batch()
            if batch:
                self._process_batch(batch)
            else:
                time.sleep(0.01)  # Small sleep if no requests

    def _collect_batch(self) -> List[InferenceRequest]:
        """Collect requests into a batch"""
        batch = []
        deadline = time.time() + self.max_wait_time

        while len(batch) < self.max_batch_size and time.time() < deadline:
            if not self.request_queue.empty():
                try:
                    request = self.request_queue.get_nowait()
                    batch.append(request)
                except:
                    break
            else:
                # Wait a bit for more requests
                time.sleep(0.001)

        return batch

    def _process_batch(self, batch: List[InferenceRequest]):
        """Process a batch of requests"""
        # Sort by sequence length for efficient padding
        batch.sort(key=lambda x: len(x.prompt))

        # Extract prompts and parameters
        prompts = [r.prompt for r in batch]

        # Batch inference
        responses = self.model.generate_batch(prompts)

        # Return results via callbacks
        for request, response in zip(batch, responses):
            if request.callback:
                request.callback(response)

class ContinuousBatching:
    """
    Implement continuous batching for optimal throughput
    """
    def __init__(self, model, max_batch_size: int = 128):
        self.model = model
        self.max_batch_size = max_batch_size
        self.active_sequences = {}
        self.kv_cache = {}

    def add_sequence(self, seq_id: str, prompt: str):
        """Add new sequence to active batch"""
        self.active_sequences[seq_id] = {
            "prompt": prompt,
            "generated": [],
            "done": False,
            "position": 0
        }

    def step(self) -> Dict[str, str]:
        """Perform one generation step for all active sequences"""
        if not self.active_sequences:
            return {}

        # Prepare batch
        batch_inputs = []
        seq_ids = []

        for seq_id, seq_data in self.active_sequences.items():
            if not seq_data["done"]:
                batch_inputs.append(seq_data)
                seq_ids.append(seq_id)

        # Generate next tokens
        next_tokens = self.model.generate_next_tokens(
            batch_inputs,
            kv_cache=self.kv_cache
        )

        # Update sequences
        results = {}
        for seq_id, next_token in zip(seq_ids, next_tokens):
            self.active_sequences[seq_id]["generated"].append(next_token)

            # Check if sequence is done
            if next_token in ["</s>", "<|endoftext|>"]:
                self.active_sequences[seq_id]["done"] = True
                results[seq_id] = "".join(self.active_sequences[seq_id]["generated"])

        # Remove completed sequences
        for seq_id in results:
            del self.active_sequences[seq_id]
            if seq_id in self.kv_cache:
                del self.kv_cache[seq_id]

        return results
```

## Module 2: Context Management and Memory Optimization (Week 3-4)

### 2.1 Advanced Context Window Management

```python
class ContextWindowManager:
    def __init__(self, max_context_length: int = 32768):
        self.max_context_length = max_context_length
        self.compression_strategies = {
            "summarization": self.compress_by_summarization,
            "selective": self.selective_context_retention,
            "hierarchical": self.hierarchical_compression,
            "sliding": self.sliding_window_compression
        }

    def manage_context(self, context: str, new_input: str,
                      strategy: str = "hierarchical") -> str:
        """Manage context to fit within window"""
        total_length = len(context) + len(new_input)

        if total_length <= self.max_context_length:
            return context + "\n" + new_input

        # Apply compression strategy
        compression_func = self.compression_strategies[strategy]
        compressed_context = compression_func(context, new_input)

        return compressed_context

    def compress_by_summarization(self, context: str, new_input: str) -> str:
        """Compress context by summarizing older parts"""
        # Split context into chunks
        chunks = self.split_into_chunks(context, chunk_size=1000)

        # Keep recent chunks as-is, summarize older ones
        cutoff = len(chunks) // 3
        summarized = []

        for i, chunk in enumerate(chunks):
            if i < cutoff:
                # Summarize older chunks more aggressively
                summary = self.summarize_chunk(chunk, max_length=100)
                summarized.append(f"[Summarized]: {summary}")
            else:
                # Keep recent chunks
                summarized.append(chunk)

        compressed = "\n".join(summarized) + "\n" + new_input
        return self.truncate_to_fit(compressed)

    def selective_context_retention(self, context: str, new_input: str) -> str:
        """Keep only relevant parts of context"""
        # Extract key information
        entities = self.extract_entities(context)
        facts = self.extract_facts(context)
        recent_exchanges = self.get_recent_exchanges(context, n=3)

        # Reconstruct compressed context
        compressed_parts = [
            f"Key entities: {', '.join(entities)}",
            f"Important facts: {' '.join(facts)}",
            "Recent context:",
            "\n".join(recent_exchanges)
        ]

        compressed = "\n".join(compressed_parts) + "\n" + new_input
        return self.truncate_to_fit(compressed)

    def hierarchical_compression(self, context: str, new_input: str) -> str:
        """Multi-level compression based on importance"""
        # Parse context into segments with importance scores
        segments = self.segment_and_score(context)

        # Sort by importance
        segments.sort(key=lambda x: x["importance"], reverse=True)

        # Build compressed context within limit
        compressed_parts = []
        current_length = len(new_input)

        for segment in segments:
            segment_text = segment["text"]

            # Apply compression based on importance
            if segment["importance"] > 0.8:
                # Keep as-is
                compressed_text = segment_text
            elif segment["importance"] > 0.5:
                # Light compression
                compressed_text = self.light_compress(segment_text)
            else:
                # Heavy compression
                compressed_text = self.heavy_compress(segment_text)

            if current_length + len(compressed_text) < self.max_context_length:
                compressed_parts.append(compressed_text)
                current_length += len(compressed_text)
            else:
                break

        return "\n".join(compressed_parts) + "\n" + new_input

    def sliding_window_compression(self, context: str, new_input: str) -> str:
        """Maintain sliding window with decay"""
        window_size = self.max_context_length - len(new_input) - 100  # Buffer

        if len(context) <= window_size:
            return context + "\n" + new_input

        # Keep most recent content
        return context[-window_size:] + "\n" + new_input

    def segment_and_score(self, context: str) -> List[Dict]:
        """Segment context and assign importance scores"""
        segments = []

        # Simple heuristic-based scoring
        lines = context.split("\n")
        for line in lines:
            importance = 0.5  # Base score

            # Adjust based on content
            if any(keyword in line.lower() for keyword in ["important", "key", "critical"]):
                importance += 0.3
            if any(keyword in line.lower() for keyword in ["user:", "assistant:"]):
                importance += 0.2
            if len(line) > 200:  # Longer content might be more detailed
                importance += 0.1

            segments.append({
                "text": line,
                "importance": min(1.0, importance)
            })

        return segments
```

### 2.2 Memory-Efficient Model Loading

```python
class MemoryEfficientLoader:
    def __init__(self):
        self.loaded_models = {}
        self.memory_monitor = MemoryMonitor()

    def load_model_sharded(self, model_name: str, num_shards: int = 4):
        """Load model in shards across multiple GPUs"""
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch

        # Initialize model with empty weights
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                AutoConfig.from_pretrained(model_name)
            )

        # Load and dispatch across devices
        model = load_checkpoint_and_dispatch(
            model,
            model_name,
            device_map="auto",
            max_memory={
                0: "10GiB",
                1: "10GiB",
                2: "10GiB",
                3: "10GiB",
                "cpu": "30GiB"
            },
            no_split_module_classes=["Block"],
            dtype=torch.float16
        )

        return model

    def quantize_model(self, model, quantization_config: Dict):
        """Apply quantization for memory reduction"""
        from transformers import BitsAndBytesConfig
        import bitsandbytes as bnb

        if quantization_config["method"] == "int8":
            # 8-bit quantization
            model = bnb.nn.Linear8bitLt(model)
        elif quantization_config["method"] == "int4":
            # 4-bit quantization
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = model.quantize(config)
        elif quantization_config["method"] == "fp8":
            # FP8 quantization (for newest GPUs)
            model = self.apply_fp8_quantization(model)

        return model

    def offload_to_cpu(self, model, offload_config: Dict):
        """Offload model layers to CPU when not in use"""
        from accelerate import cpu_offload_with_hook

        # Offload specific layers
        offload_layers = offload_config.get("layers", [])
        hooks = []

        for layer_name in offload_layers:
            layer = getattr(model, layer_name)
            hook = cpu_offload_with_hook(layer, execution_device="cuda:0")
            hooks.append(hook)

        return model, hooks

    def implement_gradient_checkpointing(self, model):
        """Enable gradient checkpointing to save memory during training"""
        model.gradient_checkpointing_enable()

        # Custom checkpointing for specific layers
        def checkpoint_forward(module, *args, **kwargs):
            return torch.utils.checkpoint.checkpoint(
                module._forward,
                *args,
                use_reentrant=False,
                **kwargs
            )

        # Apply to transformer layers
        for layer in model.transformer.h:
            layer.forward = lambda *args, **kwargs: checkpoint_forward(layer, *args, **kwargs)

        return model

class MemoryMonitor:
    def __init__(self):
        self.gpu_memory_reserved = {}
        self.gpu_memory_allocated = {}

    def get_memory_stats(self) -> Dict:
        """Get current memory statistics"""
        stats = {}

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                stats[f"gpu_{i}"] = {
                    "allocated": torch.cuda.memory_allocated(i) / 1024**3,  # GB
                    "reserved": torch.cuda.memory_reserved(i) / 1024**3,    # GB
                    "free": (torch.cuda.get_device_properties(i).total_memory -
                            torch.cuda.memory_allocated(i)) / 1024**3       # GB
                }

        # CPU memory
        import psutil
        mem = psutil.virtual_memory()
        stats["cpu"] = {
            "used": mem.used / 1024**3,
            "available": mem.available / 1024**3,
            "percent": mem.percent
        }

        return stats

    def optimize_memory(self):
        """Optimize memory usage"""
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Garbage collection
        import gc
        gc.collect()

        # Clear unused variables
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                if obj.device.type == 'cuda' and not obj.is_pinned():
                    del obj
```

## Module 3: Cost Optimization Strategies (Week 5-6)

### 3.1 Dynamic Model Selection

```python
class CostOptimizer:
    def __init__(self):
        self.model_registry = self.initialize_model_registry()
        self.cost_tracker = CostTracker()
        self.performance_monitor = PerformanceMonitor()

    def initialize_model_registry(self) -> Dict:
        """Registry of available models with costs and capabilities"""
        return {
            "gpt-4o": {
                "cost_per_1k_input": 0.005,
                "cost_per_1k_output": 0.015,
                "quality_score": 0.95,
                "latency": 2.5,
                "capabilities": ["complex_reasoning", "creative", "analysis"]
            },
            "gpt-4o-mini": {
                "cost_per_1k_input": 0.00015,
                "cost_per_1k_output": 0.0006,
                "quality_score": 0.80,
                "latency": 1.0,
                "capabilities": ["general", "simple_tasks"]
            },
            "claude-3-5-sonnet": {
                "cost_per_1k_input": 0.003,
                "cost_per_1k_output": 0.015,
                "quality_score": 0.93,
                "latency": 2.0,
                "capabilities": ["analysis", "coding", "long_context"]
            },
            "llama-3-70b": {
                "cost_per_1k_input": 0.0008,
                "cost_per_1k_output": 0.0008,
                "quality_score": 0.85,
                "latency": 1.5,
                "capabilities": ["general", "multilingual"]
            },
            "mixtral-8x7b": {
                "cost_per_1k_input": 0.0005,
                "cost_per_1k_output": 0.0005,
                "quality_score": 0.78,
                "latency": 1.2,
                "capabilities": ["general", "fast"]
            }
        }

    def select_optimal_model(self, task: Dict, constraints: Dict) -> str:
        """Select most cost-effective model for task"""
        required_quality = constraints.get("min_quality", 0.7)
        max_latency = constraints.get("max_latency", 5.0)
        max_cost = constraints.get("max_cost", float('inf'))
        required_capabilities = set(task.get("capabilities", []))

        candidates = []

        for model_name, model_info in self.model_registry.items():
            # Check constraints
            if model_info["quality_score"] < required_quality:
                continue
            if model_info["latency"] > max_latency:
                continue

            # Check capabilities
            model_capabilities = set(model_info["capabilities"])
            if required_capabilities and not required_capabilities.issubset(model_capabilities):
                continue

            # Estimate cost for this task
            estimated_cost = self.estimate_task_cost(task, model_info)
            if estimated_cost > max_cost:
                continue

            # Calculate efficiency score
            efficiency = model_info["quality_score"] / estimated_cost
            candidates.append({
                "model": model_name,
                "cost": estimated_cost,
                "quality": model_info["quality_score"],
                "efficiency": efficiency
            })

        if not candidates:
            # Fallback to cheapest model if no candidates meet criteria
            return "mixtral-8x7b"

        # Select model with best efficiency
        best = max(candidates, key=lambda x: x["efficiency"])
        return best["model"]

    def estimate_task_cost(self, task: Dict, model_info: Dict) -> float:
        """Estimate cost for specific task"""
        # Estimate token counts
        input_tokens = task.get("estimated_input_tokens", 500)
        output_tokens = task.get("estimated_output_tokens", 500)

        input_cost = (input_tokens / 1000) * model_info["cost_per_1k_input"]
        output_cost = (output_tokens / 1000) * model_info["cost_per_1k_output"]

        return input_cost + output_cost

    def implement_cascading_strategy(self, task: str) -> Dict:
        """Implement cost-efficient cascading strategy"""
        # Start with cheapest model
        models_tried = []
        total_cost = 0

        for model_name in sorted(self.model_registry.keys(),
                                key=lambda x: self.model_registry[x]["cost_per_1k_input"]):
            model_info = self.model_registry[model_name]

            # Try current model
            result = self.try_model(model_name, task)
            cost = self.calculate_actual_cost(result, model_info)
            total_cost += cost

            models_tried.append({
                "model": model_name,
                "cost": cost,
                "quality": result.get("quality_score", 0)
            })

            # Check if quality is sufficient
            if result.get("quality_score", 0) >= 0.8:
                return {
                    "final_model": model_name,
                    "result": result,
                    "total_cost": total_cost,
                    "models_tried": models_tried
                }

        # Return best result if quality threshold not met
        best_result = max(models_tried, key=lambda x: x["quality"])
        return {
            "final_model": best_result["model"],
            "total_cost": total_cost,
            "models_tried": models_tried,
            "quality_warning": "Quality threshold not met"
        }

class CostTracker:
    def __init__(self):
        self.usage_history = []
        self.cost_by_model = {}
        self.cost_by_user = {}
        self.daily_budget = 1000  # $1000 per day

    def track_usage(self, model: str, input_tokens: int,
                   output_tokens: int, user_id: str):
        """Track API usage and costs"""
        timestamp = time.time()

        # Calculate cost
        model_costs = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            # Add more models
        }

        costs = model_costs.get(model, {"input": 0.001, "output": 0.001})
        total_cost = (input_tokens / 1000) * costs["input"] + \
                    (output_tokens / 1000) * costs["output"]

        # Record usage
        usage = {
            "timestamp": timestamp,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": total_cost,
            "user_id": user_id
        }

        self.usage_history.append(usage)

        # Update aggregates
        self.cost_by_model[model] = self.cost_by_model.get(model, 0) + total_cost
        self.cost_by_user[user_id] = self.cost_by_user.get(user_id, 0) + total_cost

        # Check budget
        self.check_budget_alerts()

    def check_budget_alerts(self):
        """Monitor and alert on budget usage"""
        # Calculate today's spending
        today_start = time.time() - (time.time() % 86400)
        today_cost = sum(
            u["cost"] for u in self.usage_history
            if u["timestamp"] >= today_start
        )

        if today_cost > self.daily_budget * 0.8:
            self.send_alert(f"Daily budget 80% consumed: ${today_cost:.2f}")

        if today_cost > self.daily_budget:
            self.send_alert(f"Daily budget exceeded: ${today_cost:.2f}")
            self.implement_throttling()

    def generate_cost_report(self, period: str = "daily") -> Dict:
        """Generate cost analysis report"""
        report = {
            "period": period,
            "total_cost": sum(u["cost"] for u in self.usage_history),
            "by_model": self.cost_by_model,
            "by_user": self.cost_by_user,
            "top_users": sorted(
                self.cost_by_user.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "optimization_opportunities": self.identify_optimizations()
        }

        return report

    def identify_optimizations(self) -> List[Dict]:
        """Identify cost optimization opportunities"""
        optimizations = []

        # Analyze model usage patterns
        model_usage = {}
        for usage in self.usage_history:
            model = usage["model"]
            if model not in model_usage:
                model_usage[model] = {"count": 0, "total_cost": 0}
            model_usage[model]["count"] += 1
            model_usage[model]["total_cost"] += usage["cost"]

        # Suggest model downgrades for simple tasks
        for model, stats in model_usage.items():
            if model == "gpt-4o" and stats["count"] > 100:
                avg_tokens = sum(u["output_tokens"] for u in self.usage_history
                               if u["model"] == model) / stats["count"]
                if avg_tokens < 200:
                    optimizations.append({
                        "type": "model_downgrade",
                        "current_model": model,
                        "suggested_model": "gpt-4o-mini",
                        "potential_savings": stats["total_cost"] * 0.8
                    })

        return optimizations
```

## Module 4: Privacy, Compliance, and Governance (Week 7-8)

### 4.1 Privacy-Preserving AI Implementation

```python
class PrivacyPreservingAI:
    def __init__(self):
        self.setup_privacy_tools()
        self.compliance_standards = ["GDPR", "HIPAA", "SOC2", "CCPA"]

    def setup_privacy_tools(self):
        """Initialize privacy protection tools"""
        self.pii_detector = PIIDetector()
        self.data_anonymizer = DataAnonymizer()
        self.differential_privacy = DifferentialPrivacy()
        self.audit_logger = AuditLogger()

    def process_with_privacy(self, data: str, user_consent: Dict) -> str:
        """Process data with privacy protection"""
        # Log access
        self.audit_logger.log_access(data, user_consent)

        # Detect and handle PII
        pii_scan = self.pii_detector.scan(data)
        if pii_scan["contains_pii"]:
            if not user_consent.get("allow_pii_processing", False):
                # Anonymize PII
                data = self.data_anonymizer.anonymize(data, pii_scan["entities"])

        # Apply differential privacy for aggregations
        if user_consent.get("require_differential_privacy", False):
            data = self.differential_privacy.add_noise(data)

        return data

class PIIDetector:
    def __init__(self):
        import spacy
        self.nlp = spacy.load("en_core_web_sm")
        self.patterns = {
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b',
            "ip_address": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            "medical_record": r'\b[A-Z]{2}\d{6,8}\b',
            "passport": r'\b[A-Z][0-9]{8}\b'
        }

    def scan(self, text: str) -> Dict:
        """Scan text for PII"""
        results = {
            "contains_pii": False,
            "entities": [],
            "risk_level": "low"
        }

        # Named entity recognition
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "DATE"]:
                results["entities"].append({
                    "text": ent.text,
                    "type": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })

        # Pattern matching
        import re
        for pii_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                results["entities"].append({
                    "text": match.group(),
                    "type": pii_type,
                    "start": match.start(),
                    "end": match.end()
                })

        if results["entities"]:
            results["contains_pii"] = True
            results["risk_level"] = self.assess_risk_level(results["entities"])

        return results

    def assess_risk_level(self, entities: List[Dict]) -> str:
        """Assess privacy risk level"""
        high_risk_types = ["ssn", "credit_card", "medical_record", "passport"]
        medium_risk_types = ["email", "phone", "ip_address"]

        for entity in entities:
            if entity["type"] in high_risk_types:
                return "high"

        for entity in entities:
            if entity["type"] in medium_risk_types:
                return "medium"

        return "low"

class DataAnonymizer:
    def __init__(self):
        self.replacement_strategies = {
            "PERSON": self.anonymize_person,
            "email": self.anonymize_email,
            "phone": self.anonymize_phone,
            "ssn": lambda x: "[SSN_REDACTED]",
            "credit_card": lambda x: "[CC_REDACTED]"
        }

    def anonymize(self, text: str, entities: List[Dict]) -> str:
        """Anonymize PII in text"""
        # Sort entities by position (reverse order for replacement)
        entities.sort(key=lambda x: x["start"], reverse=True)

        anonymized_text = text
        for entity in entities:
            entity_type = entity["type"]

            if entity_type in self.replacement_strategies:
                replacement = self.replacement_strategies[entity_type](entity["text"])
            else:
                replacement = f"[{entity_type}_REDACTED]"

            anonymized_text = (
                anonymized_text[:entity["start"]] +
                replacement +
                anonymized_text[entity["end"]:]
            )

        return anonymized_text

    def anonymize_person(self, name: str) -> str:
        """Generate consistent anonymous name"""
        import hashlib

        # Generate consistent hash for name
        name_hash = hashlib.md5(name.encode()).hexdigest()

        # Use hash to select from anonymous names
        first_names = ["Alex", "Jordan", "Casey", "Morgan", "Riley"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones"]

        first_idx = int(name_hash[:8], 16) % len(first_names)
        last_idx = int(name_hash[8:16], 16) % len(last_names)

        return f"{first_names[first_idx]} {last_names[last_idx]}"

    def anonymize_email(self, email: str) -> str:
        """Anonymize email while preserving domain"""
        parts = email.split("@")
        if len(parts) == 2:
            return f"user_xxxx@{parts[1]}"
        return "[EMAIL_REDACTED]"

    def anonymize_phone(self, phone: str) -> str:
        """Anonymize phone number keeping area code"""
        import re

        # Extract area code if present
        match = re.match(r'(\+?\d{1,3}[-.]?)?\(?(\d{3})\)?', phone)
        if match:
            area_code = match.group(2)
            return f"({area_code}) XXX-XXXX"
        return "[PHONE_REDACTED]"

class ComplianceManager:
    def __init__(self):
        self.compliance_checks = {
            "GDPR": self.check_gdpr_compliance,
            "HIPAA": self.check_hipaa_compliance,
            "SOC2": self.check_soc2_compliance,
            "CCPA": self.check_ccpa_compliance
        }
        self.data_retention_policies = self.load_retention_policies()

    def check_compliance(self, data_processing: Dict) -> Dict:
        """Check compliance with regulations"""
        results = {
            "compliant": True,
            "violations": [],
            "warnings": [],
            "recommendations": []
        }

        for regulation, check_func in self.compliance_checks.items():
            check_result = check_func(data_processing)

            if not check_result["compliant"]:
                results["compliant"] = False
                results["violations"].extend(check_result.get("violations", []))

            results["warnings"].extend(check_result.get("warnings", []))
            results["recommendations"].extend(check_result.get("recommendations", []))

        return results

    def check_gdpr_compliance(self, data_processing: Dict) -> Dict:
        """Check GDPR compliance"""
        result = {"compliant": True, "violations": [], "warnings": []}

        # Check for lawful basis
        if not data_processing.get("lawful_basis"):
            result["violations"].append("No lawful basis for processing")
            result["compliant"] = False

        # Check for user consent
        if data_processing.get("personal_data") and not data_processing.get("user_consent"):
            result["violations"].append("Processing personal data without consent")
            result["compliant"] = False

        # Check data minimization
        if data_processing.get("data_collected_unnecessarily"):
            result["warnings"].append("Potential violation of data minimization principle")

        # Check right to erasure
        if not data_processing.get("deletion_mechanism"):
            result["warnings"].append("No mechanism for right to erasure")

        return result

    def implement_data_retention(self, data: Dict, policy: str):
        """Implement data retention policies"""
        retention_period = self.data_retention_policies.get(policy, 90)  # days

        # Schedule deletion
        deletion_date = time.time() + (retention_period * 86400)

        return {
            "data_id": data.get("id"),
            "retention_period": retention_period,
            "deletion_scheduled": deletion_date,
            "policy": policy
        }

class AuditLogger:
    def __init__(self, log_path: str = "/var/log/ai_audit.log"):
        self.log_path = log_path
        self.setup_logger()

    def setup_logger(self):
        """Setup secure audit logging"""
        import logging
        from logging.handlers import RotatingFileHandler

        self.logger = logging.getLogger("ai_audit")
        self.logger.setLevel(logging.INFO)

        # Rotating file handler with encryption
        handler = RotatingFileHandler(
            self.log_path,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10
        )

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_access(self, data: Any, user_context: Dict):
        """Log data access for audit trail"""
        import json
        import hashlib

        log_entry = {
            "timestamp": time.time(),
            "user_id": user_context.get("user_id"),
            "action": user_context.get("action", "access"),
            "data_hash": hashlib.sha256(str(data).encode()).hexdigest(),
            "ip_address": user_context.get("ip_address"),
            "session_id": user_context.get("session_id"),
            "compliance_flags": user_context.get("compliance_flags", [])
        }

        self.logger.info(json.dumps(log_entry))

    def generate_audit_report(self, start_date: str, end_date: str) -> Dict:
        """Generate audit report for compliance"""
        # Parse logs and generate report
        report = {
            "period": f"{start_date} to {end_date}",
            "total_accesses": 0,
            "unique_users": set(),
            "actions": {},
            "compliance_violations": []
        }

        # Read and parse logs
        with open(self.log_path, 'r') as f:
            for line in f:
                try:
                    # Parse log entry
                    entry = json.loads(line.split(" - ")[-1])

                    # Update statistics
                    report["total_accesses"] += 1
                    report["unique_users"].add(entry.get("user_id"))

                    action = entry.get("action")
                    report["actions"][action] = report["actions"].get(action, 0) + 1

                    # Check for violations
                    if entry.get("compliance_flags"):
                        report["compliance_violations"].append(entry)

                except Exception as e:
                    continue

        report["unique_users"] = len(report["unique_users"])
        return report
```

## Practical Exercises

### Exercise 1: Distributed Inference System
Build a complete distributed inference system that:
1. Implements model sharding across GPUs
2. Handles dynamic batching
3. Provides load balancing
4. Monitors performance metrics

### Exercise 2: Cost Optimization Pipeline
Create a cost optimization system that:
1. Tracks API usage and costs
2. Implements model cascading
3. Provides budget alerts
4. Generates optimization recommendations

### Exercise 3: Privacy-Preserving System
Implement a privacy-preserving AI system that:
1. Detects and anonymizes PII
2. Implements differential privacy
3. Maintains audit logs
4. Ensures regulatory compliance

### Exercise 4: Production Monitoring
Build a monitoring system that:
1. Tracks system performance
2. Monitors model drift
3. Alerts on anomalies
4. Generates operational reports

## Assessment Criteria

### System Design (30%)
- Architecture scalability
- Performance optimization
- Cost efficiency
- Security implementation

### Implementation Quality (30%)
- Code organization
- Error handling
- Documentation
- Testing coverage

### Operational Excellence (25%)
- Monitoring completeness
- Incident response
- Compliance adherence
- Audit trail quality

### Innovation (15%)
- Novel optimization techniques
- Creative problem solving
- Advanced features
- Research integration

## Summary
Level 4 represents the pinnacle of AI engineering education, preparing students to deploy and operate AI systems at enterprise scale. Students master distributed computing, advanced optimization, compliance requirements, and operational excellence. This level transforms students into AI architects capable of building and maintaining production systems that serve millions of users while balancing performance, cost, and compliance requirements.

# 2025-09-14 00:00:00