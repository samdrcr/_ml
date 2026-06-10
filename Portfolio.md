# Machine Learning Course — Homework Portfolio

**Student:** 范權榮 &nbsp;|&nbsp; **Student ID:** 111210557 &nbsp;|&nbsp; **Course:** Machine Learning (ccc114b/\_ml)

---

## Portfolio Overview

| #       | Assignment                                     | Topic                       | Key Concept                       | Repo                                                                                     |
| ------- | ---------------------------------------------- | --------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------- |
| HW1     | Traveling Salesperson Problem                  | Local Search                | Hill Climbing + 2-opt             | [Link](https://github.com/samdrcr/_ml/tree/d65b2aeb4e79068b9e8b9b5fe59e7ff5c8279593/HW1) |
| HW2     | Backpropagation & Computational Graphs         | Neural Network Fundamentals | Chain Rule, Forward/Backward Pass | [Link](https://github.com/samdrcr/_ml/tree/d65b2aeb4e79068b9e8b9b5fe59e7ff5c8279593/HW2) |
| HW3     | Neural Network Training Interface              | Full-Stack ML Visualization | FastAPI + React + SSE Streaming   | [Link](https://github.com/samdrcr/_ml/tree/d65b2aeb4e79068b9e8b9b5fe59e7ff5c8279593/HW3) |
| HW4     | microGPT                                       | Transformer Architecture    | GPT Built from Scratch            | [Link](https://github.com/samdrcr/_ml/tree/d65b2aeb4e79068b9e8b9b5fe59e7ff5c8279593/HW4) |
| HW5     | Secure Agent Wrapper                           | AI Agent Safety             | Sandboxing + Human-in-the-Loop    | [Link](https://github.com/samdrcr/_ml/tree/d65b2aeb4e79068b9e8b9b5fe59e7ff5c8279593/HW5) |
| HW6     | Markov Chain Text Generator                    | Probabilistic NLP           | 2nd-Order Markov Chain            | [Link](https://github.com/samdrcr/_ml/tree/d65b2aeb4e79068b9e8b9b5fe59e7ff5c8279593/HW6) |
| Midterm | Automated Code Generation & Evaluation Harness | LLM Agents                  | Multi-Stage Refinement Loop       | [Link](https://github.com/samdrcr/my-harness-agent)                                      |

---

## Learning Progression

```
HW1              HW2                HW3               HW4
Hill Climbing → Backpropagation → NN Training UI → GPT from Scratch
(Search)        (Gradients)        (Full-Stack)      (Transformer)

    HW5               HW6                  Midterm
AI Sandboxing → Markov Chain NLP → Automated Code Harness + LLM Agent
(Safety)         (Probabilistic)    (Agent Loop + Dynamic Evaluation)
```

---

## Homework 1 — Traveling Salesperson Problem

**Repository:** [HW1](https://github.com/samdrcr/_ml/tree/d65b2aeb4e79068b9e8b9b5fe59e7ff5c8279593/HW1)

### Overview

This assignment implements a **Hill Climbing** algorithm to solve the classic Traveling Salesperson Problem (TSP) — finding the shortest possible route that visits every city exactly once and returns to the origin. Rather than exhaustively searching all permutations, the algorithm climbs toward better solutions using local neighborhood exploration.

### Algorithm Design

The solver represents a solution as a permutation of city indices. The height function is defined as the negative total distance, converting the minimization problem into a maximization one suitable for hill climbing. Neighbors are generated using a **2-opt swap**, which reverses a segment of the current route to eliminate crossing paths.

```
Height Function:   h(solution) = -total_distance(solution)

Neighbor Strategy: pick indices i, j at random
                   reverse path[i:j+1]  ← uncrosses edges
```

The algorithm terminates when no neighbor improves the current solution, meaning a local optimum has been reached.

```
def hill_climbing(initial_solution):
    current = initial_solution
    while True:
        neighbor = current.neighbor()
        if neighbor.height() > current.height():
            current = neighbor
        else:
            break
    return current
```

### Key Components

| Component            | Description                                |
| -------------------- | ------------------------------------------ |
| State Representation | Permutation of city indices `[0 .. n-1]`   |
| Objective            | Minimize total Euclidean distance          |
| Neighbor Function    | 2-opt segment reversal                     |
| Reproducibility      | `random.seed(0)` for deterministic results |

### How to Run

```bash
.\.venv\Scripts\Activate.ps1
python HillClimbing.py
```

### Key Takeaways

Local search is powerful for combinatorial optimization — 2-opt swaps efficiently uncross route segments without requiring global knowledge. The main limitation is susceptibility to local optima, a foundational tension that motivates the more sophisticated learning methods explored in later assignments.

---

## Homework 2 — Backpropagation & Computational Graphs

**Repository:** [HW2](https://github.com/samdrcr/_ml/tree/d65b2aeb4e79068b9e8b9b5fe59e7ff5c8279593/HW2)

### Overview

This assignment explores **Backpropagation** — the algorithm that makes neural network training possible. Rather than using a library, gradients are computed manually by constructing explicit computational graphs and applying the chain rule, mirroring exactly what frameworks like PyTorch do internally.

### Theoretical Foundation

A computational graph maps data flow through a sequence of differentiable operations. Each node stores an operation; each edge carries a tensor value. During the **forward pass**, values are computed from inputs to output. During the **backward pass**, gradients flow in reverse, multiplied at each step by the local derivative.

**Chain Rule:**

$$\frac{df}{dx} = \frac{df}{dh} \cdot \frac{dh}{dx}$$

### Function 1: $f(x, y, z) = (x \cdot y) + z$ &nbsp; with $x=2,\ y=3,\ z=4$

| Node    | Forward Value | Gradient |
| ------- | ------------- | -------- |
| x       | 2             | **3**    |
| y       | 3             | **2**    |
| z       | 4             | **1**    |
| p = x·y | 6             | 1        |
| f       | 10            | 1        |

### Function 2: $f(x, y, z, t) = ((x \cdot y) + z) \cdot t$ &nbsp; with $x=2,\ y=3,\ z=4,\ t=5$

| Node    | Forward Value | Gradient |
| ------- | ------------- | -------- |
| x       | 2             | **15**   |
| y       | 3             | **10**   |
| z       | 4             | **5**    |
| t       | 5             | **10**   |
| p = x·y | 6             | 5        |
| q = p+z | 10            | 5        |
| f       | 50            | 1        |

**Gradient derivation for x:**

$$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial q} \cdot \frac{\partial q}{\partial p} \cdot \frac{\partial p}{\partial x} = t \cdot 1 \cdot y = 5 \cdot 1 \cdot 3 = 15$$

### Key Takeaways

Building the computational graph by hand reveals the exact mechanism that enables deep learning. Every `loss.backward()` call in PyTorch is performing this same chain-rule traversal across millions of nodes. Understanding it at this level makes all subsequent work with neural networks far more concrete.

---

## Homework 3 — Neural Network Training Interface

**Repository:** [HW3](https://github.com/samdrcr/_ml/tree/d65b2aeb4e79068b9e8b9b5fe59e7ff5c8279593/HW3)

### Overview

This assignment builds a full-stack **neural network training visualization interface**, framed as a power grid optimization simulator. Training metrics stream in real time from a FastAPI backend to a React frontend via **Server-Sent Events (SSE)**, allowing users to watch a neural network learn live.

### System Architecture

```
React Frontend
├── GridMap          — Neural network visualized as power grid nodes
├── TrendMonitor     — Real-time loss/accuracy chart
├── MetricsPanel     — Live training statistics
├── ControlPanel     — Start / stop / reset controls
└── OptimizerToggle  — Switch between SGD, Adam, RMSProp

         │  SSE Stream (/train/stream)
         ▼

FastAPI Backend
├── POST /train/start   — Begin training session
├── GET  /train/stream  — SSE endpoint (yields metrics per epoch)
└── POST /train/reset   — Reset model weights

         │
         ▼

nn0.py — Custom Autograd Engine
├── Value class (full computational graph)
├── Linear layers
└── SGD / Adam / RMSProp optimizers
```

### Visual Metaphor

The interface maps neural network internals onto a power grid theme, making abstract concepts tangible:

| Neural Network Element | Grid Representation                       |
| ---------------------- | ----------------------------------------- |
| Input neurons          | Sensor stations (Load, Temperature, Cost) |
| Hidden neurons         | Distribution hubs                         |
| Output neurons         | Power dispatch center                     |
| Weights                | Connection opacity (flow capacity)        |
| Activations            | Current load level                        |
| Gradients              | Stress indicators (color intensity)       |

### How to Run

```bash
# Backend
pip install -r requirements.txt
python main.py           # → http://localhost:8000

# Frontend
cd client
npm install
npm run dev              # → http://localhost:5173
```

### Key Takeaways

SSE is an elegant pattern for ML training interfaces — it keeps the HTTP connection open and pushes each epoch's metrics without polling. Running training in an isolated thread prevents the API from blocking. This assignment bridged the gap between understanding an algorithm and making it visually meaningful.

---

## Homework 4 — microGPT

**Repository:** [HW4](https://github.com/samdrcr/_ml/tree/d65b2aeb4e79068b9e8b9b5fe59e7ff5c8279593/HW4)

### Overview

This assignment implements a minimal **GPT (Generative Pre-trained Transformer)** entirely from scratch using only Python's standard library — no PyTorch, no NumPy, no external ML dependencies. Every component, from attention to the Adam optimizer, is built by hand.

### Architecture

```
Input Tokens
    → Token Embeddings + Positional Embeddings
    → [Transformer Block] × N
        RMSNorm
        → Multi-Head Causal Self-Attention
        → Feed-Forward MLP (GELU, 4× expansion)
    → RMSNorm
    → Linear projection
    → Vocabulary logits (softmax for sampling)
```

### Components

| Component       | Implementation Detail                         |
| --------------- | --------------------------------------------- |
| Autograd Engine | `Value` class with full computational graph   |
| Normalization   | RMSNorm (no mean centering, more efficient)   |
| Attention       | Causal masking + scaled dot-product           |
| Activation      | GELU (smoother than ReLU for language models) |
| Optimizer       | Adam from scratch (momentum + adaptive LR)    |
| Tokenization    | Character-level                               |

### Default Hyperparameters

| Parameter       | Value | Description             |
| --------------- | ----- | ----------------------- |
| `d_model`       | 128   | Embedding dimension     |
| `num_heads`     | 4     | Attention heads         |
| `num_layers`    | 4     | Transformer blocks      |
| `d_ff`          | 512   | Feed-forward hidden dim |
| `block_size`    | 64    | Context window length   |
| `batch_size`    | 16    | Training batch size     |
| `learning_rate` | 0.001 | Adam LR                 |
| `epochs`        | 10    | Training epochs         |

### How to Run

```bash
python Micro_gpt.py
# Auto-downloads dataset → trains → generates sample text
```

### Key Takeaways

Implementing every component manually — including RMSNorm, causal masking, and Adam — makes the Transformer architecture fully transparent. The model is small but conceptually identical to production GPT systems. Building attention from scalar operations upward clarifies why it is so effective: it is a learned, content-based routing mechanism over a context window.

**References:** _Attention Is All You Need_ (Vaswani et al., 2017) · _GPT-2_ (Radford et al.) · _Adam_ (Kingma & Ba, 2014)

---

## Homework 5 — Secure Agent Wrapper

**Repository:** [HW5](https://github.com/samdrcr/_ml/tree/d65b2aeb4e79068b9e8b9b5fe59e7ff5c8279593/HW5)

### Overview

This assignment builds a **security wrapper for AI coding agents**, implementing three independent defense layers to prevent unsafe autonomous file operations. It addresses a core challenge in deploying LLM agents: the model cannot be trusted to self-restrict, so external enforcement is required.

### Three-Layer Security Architecture

```
Agent Action Request
        │
        ▼
┌───────────────────────┐
│  Layer 1: Path        │  Restrict all file I/O to BASE_DIR sandbox.
│  Validator            │  Any path traversal → immediately blocked.
└───────────┬───────────┘
            │ valid path
            ▼
┌───────────────────────┐
│  Layer 2: LLM         │  Secondary model audits every action plan.
│  Security Reviewer    │  Returns structured XML verdict:
│                       │  <response>safe/unsafe</response>
└───────────┬───────────┘
            │ safe
            ▼
┌───────────────────────┐
│  Layer 3: Human-in-   │  Flagged actions require explicit y/n
│  the-Loop (HITL)      │  approval before execution proceeds.
└───────────┬───────────┘
            │ approved
            ▼
     Execute Action
```

### Usage

```python
from v3_agent_secure import SecureAgentWrapper

wrapper = SecureAgentWrapper(
    base_dir="./my_project",
    llm_client=llm_client,
    llm_model="gpt-4",
    hitl_auto_approve_patterns=[r"^logs/", r"\.tmp$"]
)

content = wrapper.read_file("src/main.py")
wrapper.write_file("output.txt", "Hello World")
```

```bash
pip install openai   # only external dependency
```

### Key Takeaways

Defense-in-depth is more robust than a single security mechanism — each layer catches a different class of failure. Path sandboxing is cheap and should always be present. LLM-based review catches semantic risks that rule-based filters miss. HITL approval remains essential for high-stakes operations where the cost of a mistake exceeds the cost of a human decision.

---

## Homework 6 — Second-Order Markov Chain Text Generator

**Repository:** [HW6](https://github.com/samdrcr/_ml/tree/d65b2aeb4e79068b9e8b9b5fe59e7ff5c8279593/HW6)

### Overview

This assignment implements a **2nd-Order Markov Chain** for probabilistic text generation — demonstrating how coherent sequences can emerge from statistical patterns, with no neural network required. It also serves as a historical anchor: Markov chains are the direct precursor to RNNs, LSTMs, and eventually Transformers.

### The Markov Property

> _"The next state depends only on the recent past — not the full history."_

| Model         | Memory           | Probability Form              |
| ------------- | ---------------- | ----------------------------- |
| 1st-Order     | 1 word back      | `P(next \| "cat")`            |
| **2nd-Order** | **2 words back** | **`P(next \| "the", "cat")`** |
| Nth-Order     | N words back     | `P(next \| w₁ … wₙ)`          |

### How It Works

**Training — Build the transition table from corpus text:**

```
Corpus: "the cat sat on the cat mat"

("the", "cat") → ["sat", "mat"]
("cat", "sat") → ["on"]
("sat", "on")  → ["the"]
("on",  "the") → ["cat"]
```

Duplicates are stored intentionally. `random.choice()` over the list naturally implements the empirical probability distribution without needing to track explicit counts.

**Generation — Inference loop:**

```
1. Pick seed bigram:     ("the", "cat")
2. Look up successors:   ["sat", "mat"]
3. Sample one:           "sat"
4. Slide the window:     ("cat", "sat")
5. Repeat until target length is reached
```

### Configuration

| Parameter      | Default  | Description                                 |
| -------------- | -------- | ------------------------------------------- |
| `CORPUS_FILE`  | `tw.txt` | Training text                               |
| `GENERATE_LEN` | 150      | Tokens to generate                          |
| `TOKEN_MODE`   | `"auto"` | `"word"`, `"char"`, or `"auto"` (CJK-aware) |
| `RANDOM_SEED`  | `None`   | `None` = random; integer = reproducible     |

### How to Run

```bash
python generate.py
# No pip install required — standard library only
```

### Where This Fits

```
Markov Chain (this project)
        ↓
N-gram Language Models
        ↓
RNN / LSTM
        ↓
Transformer (Attention)
        ↓
GPT / Modern LLMs
```

### Key Takeaways

Storing duplicates in a list elegantly encodes probability without explicit frequency tracking. The assignment also highlights the core limitation of fixed-order Markov models — a context window of 2 is useful but cannot capture long-range dependencies, which is precisely the problem the Transformer's attention mechanism was designed to solve.

---

## Midterm Project — Automated Code Generation & Evaluation Harness

**Repository:** [my-harness-agent](https://github.com/samdrcr/my-harness-agent)

### Overview

This midterm project builds an **automated code generation and evaluation harness** powered by LLM agents. The system takes a natural language task description, generates Python code using a language model, executes it inside a safe dynamic sandbox, and feeds errors back into the agent loop for iterative refinement — demonstrating how safety mechanisms and automated feedback can meaningfully improve the reliability of AI-generated code.

The project is structured as three sequential versions, each adding a new capability layer on top of the previous one.

```
v1_basic_agent/          — Core generation loop
v2_harness_integration/  — Dynamic execution + automated test harness
v3_secure_debug/         — Safety sandboxing + error-feedback refinement
```

### Evolution: V1 → V2 → V3

```
V1 Basic Agent
    Generate code from prompt → save to file → display output
    Limitation: never verifies if the code actually runs

         ↓

V2 Harness Integration
    Generate → execute dynamically → capture stdout/stderr
    Automated test harness evaluates correctness
    Limitation: reports failures but cannot repair them

         ↓

V3 Secure Debug
    Generate → sandbox execute → on failure:
        capture traceback → feed back to LLM → regenerate → retry
    Safety sandboxing restricts dangerous operations
    Multi-stage refinement loop improves execution success rate
```

| Capability                         | V1  | V2  | V3  |
| ---------------------------------- | --- | --- | --- |
| Generate code from prompt          | Yes | Yes | Yes |
| Execute generated code dynamically | No  | Yes | Yes |
| Capture stdout / stderr            | No  | Yes | Yes |
| Automated test harness             | No  | Yes | Yes |
| Error feedback to LLM              | No  | No  | Yes |
| Iterative refinement loop          | No  | No  | Yes |
| Safety sandboxing                  | No  | No  | Yes |

### Architecture (V3)

```
User Prompt
    │
    ▼
LLM Agent  ←─────────────────────────┐
    │                                 │
    │ generated code                  │ error traceback
    ▼                                 │
Dynamic Execution Sandbox             │
    │                                 │
    ├── Success → return output       │
    │                                 │
    └── Failure ──────────────────────┘
              (up to N retry attempts)
```

The key insight is that the error message itself becomes a structured prompt input. Rather than surfacing failures to the user, the harness packages the traceback alongside the original task and failed code, and asks the LLM to diagnose the root cause before rewriting — a lightweight form of chain-of-thought repair.

### Technical Details

The dynamic execution sandbox evaluates Python code strings in an isolated environment, restricting access to dangerous built-ins and limiting side effects. This prevents generated code from performing unintended file system operations or network calls during evaluation.

AI assistance (Gemini) was used to design the architectural split between the agent execution loop and the testing harness, and to scaffold the Python dynamic execution sandboxing. The automated testing harness and multi-stage refinement loop were implemented from scratch based on the conceptual framework of autonomous loop agents from `ccc114b/cccocw/_code/nn/agent/`.

### How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run V1 (basic generation)
python v1_basic_agent/agent.py

# Run V2 (with harness)
python v2_harness_integration/agent.py

# Run V3 (secure + self-correcting)
python v3_secure_debug/agent.py
```

### Key Takeaways

Closing the loop between generation and execution is the essential step that transforms a code generator into an agent. Automated error feedback substantially increases reliability compared to single-shot generation — the model performs better when it can see its own mistakes. Safety sandboxing is not optional: unreviewed code execution requires hard boundaries regardless of how confident the model appears.

---

_Portfolio compiled for Machine Learning course — 范權榮, 111210557_
