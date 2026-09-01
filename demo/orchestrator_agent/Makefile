.PHONY: server agent client test check audit

server: agent
agent:
	python -m src.host_agent

client:
	python client.py

test:
	pytest -q

check:
	python -m compileall -q client.py src tests
	pytest -q

audit:
	python -m src.host_agent.audit
