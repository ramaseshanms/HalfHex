#!/usr/bin/env python3
"""Generate INT8 calibration dataset for QNN quantization.

QNN quantization needs representative inputs — do NOT skip this.
"""

import numpy as np
import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./models/qwen3-1.7b-hf")

# Diverse calibration prompts — cover your actual use cases
CALIBRATION_PROMPTS = [
    "Explain quantum entanglement simply.",
    "Write a Python function to sort a list.",
    "What is the capital of France and why is it significant?",
    "Summarize the French Revolution in 3 sentences.",
    "How does photosynthesis work at the molecular level?",
    "Translate 'Hello world' to 5 different languages.",
    "What are the main differences between TCP and UDP?",
    "Describe the process of making bread from scratch.",
    "How do vaccines work in the human immune system?",
    "Explain recursion with a simple example.",
    "What is the difference between machine learning and deep learning?",
    "How does a compiler work step by step?",
    "Explain the concept of blockchain in simple terms.",
    "What are the SOLID principles in software engineering?",
    "How does DNS resolution work on the internet?",
    "Describe the water cycle in detail.",
    "What is the theory of general relativity?",
    "How do neural networks learn from data?",
    "Explain the difference between REST and GraphQL APIs.",
    "What is the role of mitochondria in cellular respiration?",
    "How does public key cryptography ensure secure communication?",
    "Describe the process of protein synthesis in cells.",
    "What are design patterns and why are they useful?",
    "How does garbage collection work in modern programming languages?",
    "Explain the concept of containerization and Docker.",
    "What is the significance of the Turing test?",
    "How do operating systems manage memory?",
    "Describe the architecture of a modern web browser.",
    "What are the key differences between SQL and NoSQL databases?",
    "How does image recognition work in computer vision?",
    "Explain the concept of eventual consistency in distributed systems.",
    "What is the CAP theorem and why does it matter?",
    "How do transformers work in natural language processing?",
    "Describe the process of TCP three-way handshake.",
    "What are microservices and when should you use them?",
    "How does a GPU differ from a CPU in computation?",
    "Explain the concept of attention mechanism in deep learning.",
    "What is the difference between supervised and unsupervised learning?",
    "How does HTTP/2 improve upon HTTP/1.1?",
    "Describe the principles behind quantum computing.",
    "What is federated learning and what are its applications?",
    "How do recommendation systems work?",
    "Explain the concept of zero-knowledge proofs.",
    "What are the key challenges in edge AI deployment?",
    "How does model quantization affect inference quality?",
    "Describe the architecture of a transformer-based language model.",
    "What is the role of batch normalization in deep learning?",
    "How do convolutional neural networks process images?",
    "Explain the concept of transfer learning.",
    "What are the trade-offs between model size and inference speed?",
]

os.makedirs("./calibration_data", exist_ok=True)
for i, prompt in enumerate(CALIBRATION_PROMPTS):
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length",
                       max_length=128, truncation=True)
    np.save(f"./calibration_data/input_ids_{i}.npy", inputs["input_ids"])
    np.save(f"./calibration_data/attention_mask_{i}.npy", inputs["attention_mask"])

print(f"[PROFILE] Generated {len(CALIBRATION_PROMPTS)} calibration samples")
