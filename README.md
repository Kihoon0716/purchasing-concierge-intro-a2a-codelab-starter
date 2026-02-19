# Purchasing Concierge A2A Demo

> **⚠️ DISCLAIMER: THIS IS NOT AN OFFICIALLY SUPPORTED GOOGLE PRODUCT. THIS PROJECT IS INTENDED FOR DEMONSTRATION PURPOSES ONLY. IT IS NOT INTENDED FOR USE IN A PRODUCTION ENVIRONMENT.**

This demo shows A2A (Agent2Agent) communication between a purchasing concierge orchestrator and remote pizza/burger seller agents using the A2A Python SDK. All agents use LangGraph + ChatOpenAI.

## Prerequisites

- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

    ```shell
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.12
    ```

## How to Run (Local)

Run all services locally: two remote seller agents + one local concierge UI.

### 1) Run the Burger Agent

1. Copy `remote_seller_agents/burger_agent/.env.example` to `remote_seller_agents/burger_agent/.env`.
2. Fill env vars:

    ```bash
    OPENAI_API_KEY={your-openai-api-key}
    OPENAI_BASE_URL={optional-openai-compatible-base-url}
    OPENAI_MODEL=gpt-4o-mini
    HOST_OVERRIDE=http://localhost:10001
    ```

3. Start the server:

    ```bash
    cd remote_seller_agents/burger_agent
    uv sync --frozen
    uv run . --host 0.0.0.0 --port 10001
    ```

### 2) Run the Pizza Agent

1. Copy `remote_seller_agents/pizza_agent/.env.example` to `remote_seller_agents/pizza_agent/.env`.
2. Fill env vars:

    ```bash
    OPENAI_API_KEY={your-openai-api-key}
    OPENAI_BASE_URL={optional-openai-compatible-base-url}
    OPENAI_MODEL=gpt-4o-mini
    HOST_OVERRIDE=http://localhost:10000
    ```

3. Start the server:

    ```bash
    cd remote_seller_agents/pizza_agent
    uv sync --frozen
    uv run . --host 0.0.0.0 --port 10000
    ```

### 3) Run the Purchasing Concierge UI (Local Orchestrator)

1. From the repo root, copy `purchasing_concierge/.env.example` to `purchasing_concierge/.env`.
2. Fill env vars:

    ```bash
    PIZZA_SELLER_AGENT_URL=http://localhost:10000
    BURGER_SELLER_AGENT_URL=http://localhost:10001
    OPENAI_API_KEY={your-openai-api-key}
    OPENAI_BASE_URL={optional-openai-compatible-base-url}
    OPENAI_MODEL=gpt-4o-mini
    ```

3. Start UI:

    ```bash
    uv sync --frozen
    uv run purchasing_concierge_ui.py
    ```

4. Open `http://localhost:8080`.
