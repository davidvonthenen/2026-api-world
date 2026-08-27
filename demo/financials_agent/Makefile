PYTHON ?= python
QUESTION ?= What is NVDA current stock price?
# QUESTION ?= Tell me something about NVIDIA past several quarters' financial performance.

.PHONY: install ingest mcp server agent client query audit test clean

install:
	$(PYTHON) -m pip install -r requirements.txt

ingest:
	$(PYTHON) -m src.ingest

mcp:
	$(PYTHON) -m src.financials_agent.finnhub_server

server: agent
agent:
	$(PYTHON) -m src.financials_agent

client: query
query:
	$(PYTHON) -m src.query -q "$(QUESTION)"

audit:
	$(PYTHON) -m src.financials_agent.audit ./var/financials-audit.jsonl

test:
	 $(PYTHON) -m pytest -q
	
clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
