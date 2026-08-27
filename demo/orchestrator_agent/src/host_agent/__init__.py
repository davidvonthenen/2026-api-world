"""Vertical API Orchestrator Agent package."""

from .config import Settings, load_settings


def create_routing_agent(settings: Settings | None = None):
    from .routing_agent import create_routing_agent as _create

    return _create(settings)


__all__ = ["Settings", "create_routing_agent", "load_settings"]
