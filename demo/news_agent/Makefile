PYTHON ?= python
QUESTION ?= What is Google currently working on in Artificial Intelligence (AI)?

.PHONY: install ingest mcp server agent client query test check clean

install:
	$(PYTHON) -m pip install -r requirements.txt

ingest:
	$(PYTHON) -m src.ingest

mcp:
	$(PYTHON) -m src.tavily_mcp_server

server: agent
agent:
	$(PYTHON) -m src.news_agent

client: query
query:
	$(PYTHON) -m src.query --question "$(QUESTION)"

test:
	$(PYTHON) -m pytest -q

check:
	$(PYTHON) -m compileall -q src tests
	$(PYTHON) -m pytest -q

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
