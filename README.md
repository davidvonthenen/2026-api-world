# API World * AI TechWorld 2026: Vertical APIs for Agentic AI: Routing the Right Context to the Right Expert

All resources (slides, code, etc) for API World 2026: Vertical APIs for Agentic AI: Routing the Right Context to the Right Expert

## Demo Prerequisites

Participants should ensure they have the minimum requirements:

- A Linux or Mac Developer's Laptop with enough memory (16GB minimum) to run 2 databases containers plus a [Quantized 7B Small Language Model](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF).
  - No GPU is required. Will strictly be using CPU only.
  - Windows Users should use a VM or Cloud Instance
    - If you opt for this, you must provide your own instances
- Tested On Python 3.12 Only (Should work on 3.10+)
  - (HIGHLY Recommended) Use [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview), [uv](https://docs.astral.sh/uv/getting-started/installation/), or [venv](https://docs.python.org/3/library/venv.html) virtual development environment. (I prefer miniconda and will provide instructions usig it)
- (HIGHLY Recommended) [Huggingface developer token](https://huggingface.co/settings/tokens) saved to `HF_TOKEN` environment variable.
- Using OpenSearch on [Instaclustr](https://www.instaclustr.com/). Try for 30 days for free! [SIGN UP HERE!](https://bit.ly/44gYn7J)

Software Downloads:
- [Qwen2.5-7B-Instruct-1M](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M)
  - Running on CPU? Use [Q5_K_M version](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF/blob/main/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf)
  - [(Apple Silicon?) Download this instead](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-1M-4bit)
  - Drop the file into your ~/models folder. (You might need to create this.)
- [Nemotron-Orchestrator-8B](https://huggingface.co/nvidia/Nemotron-Orchestrator-8B)
  - Running on CPU? Use [Q4_K_M version](https://huggingface.co/Mungert/Nemotron-Orchestrator-8B-GGUF/blob/main/Nemotron-Orchestrator-8B-q4_k_m.gguf)
  - [(Apple Silicon?) Download this instead](https://huggingface.co/mlx-community/Orchestrator-8B-4bit)
  - Drop the file into your ~/models folder. (You might need to create this.)

Install Python dependencies from any folder:

```bash
pip install -r requirements.txt
```

## Start the Demo

Ingest the data:

```bash
cd news_agent
make ingest

cd financials_agent
make ingest
```

Start the services:

```bash
# new console
cd slm_service
python slm_service.py

# new console
cd orch_service
python orch_service.py

# new console
cd news_agent
make mcp

# new console
cd news_agent
make agent

# new console
cd financials_agent
make mcp

# new console
cd financials_agent
make agent

# new console
cd orchestrator_agent
make agent

# new console
cd orchestrator_agent
make client
```

Ask the following question in the client:

```text
Can you tell me what NVIDIA is doing in the Artificial Intelligence space in news articles? How that work has affected their current stock price (ticker symbol: NVDA)?
```
