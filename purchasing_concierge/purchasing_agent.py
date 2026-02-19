"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import json
import os
import uuid
from typing import Any

import httpx
from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from .remote_agent_connection import RemoteAgentConnections


class PurchasingAgent:
    """Purchasing agent that delegates work to remote A2A seller agents."""

    ROOT_INSTRUCTION = """You are an expert purchasing delegator that can delegate the user product inquiry and purchase request to the
appropriate seller remote agents.

Execution:
- For actionable tasks, use `send_task` to assign tasks to remote agents.
- If a remote agent repeatedly asks for confirmation, improve the task description with the needed context.
- Never ask user permission before contacting remote agents.
- Always show the detailed response from seller agents.
- If a remote seller asks for confirmation and user already confirmed earlier in chat, confirm on behalf of the user.
- Do not give irrelevant context to remote seller agent.
- Never ask order confirmation to a remote seller agent.

Please rely on tools to address the request and don't make up responses.
"""

    def __init__(self, remote_agent_addresses: list[str]):
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.remote_agent_addresses = remote_agent_addresses
        self.cards: dict[str, AgentCard] = {}
        self.a2a_client_init_status = False

        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        self.model = ChatOpenAI(
            model=openai_model,
            base_url=openai_base_url,
            api_key=openai_api_key,
            temperature=0,
        )
        self.memory = MemorySaver()
        self.graph = create_react_agent(
            self.model,
            tools=[self.send_task],
            checkpointer=self.memory,
            prompt=self.root_instruction,
        )

    @property
    def root_instruction(self) -> str:
        agents = self.list_remote_agents()
        agent_info = "\n".join(json.dumps(agent) for agent in agents) if agents else "No remote agents discovered yet."
        return f"{self.ROOT_INSTRUCTION}\n\nAgents:\n{agent_info}"

    async def _initialize_remote_agent_connections(self) -> None:
        if self.a2a_client_init_status:
            return

        httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout=30))
        try:
            for address in self.remote_agent_addresses:
                card_resolver = A2ACardResolver(base_url=address, httpx_client=httpx_client)
                try:
                    card = await card_resolver.get_agent_card()
                    remote_connection = RemoteAgentConnections(agent_card=card, agent_url=card.url)
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                except httpx.ConnectError:
                    print(f"ERROR: Failed to get agent card from : {address}")
        finally:
            await httpx_client.aclose()

        self.a2a_client_init_status = True

    def ensure_initialized(self) -> None:
        if self.a2a_client_init_status:
            return
        asyncio.run(self._initialize_remote_agent_connections())

    def list_remote_agents(self) -> list[dict[str, str]]:
        self.ensure_initialized()
        return [
            {"name": card.name, "description": card.description}
            for card in self.cards.values()
        ]

    @tool
    def send_task(self, agent_name: str, task: str, config: RunnableConfig) -> str:
        """Send a task to a remote seller agent via A2A."""
        self.ensure_initialized()

        if agent_name not in self.remote_agent_connections:
            available_agents = ", ".join(self.remote_agent_connections.keys())
            raise ValueError(f"Agent {agent_name} not found. Available agents: {available_agents}")

        client = self.remote_agent_connections[agent_name]
        if not client:
            raise ValueError(f"Client not available for {agent_name}")

        thread_id = config.get("configurable", {}).get("thread_id", str(uuid.uuid4()))
        message_id = str(uuid.uuid4())

        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task}],
                "messageId": message_id,
                "contextId": thread_id,
            }
        }

        message_request = SendMessageRequest(id=message_id, params=MessageSendParams.model_validate(payload))
        send_response: SendMessageResponse = client.send_message(message_request=message_request)

        if not isinstance(send_response.root, SendMessageSuccessResponse):
            raise ValueError("Received non-success response from remote agent")

        if not isinstance(send_response.root.result, Task):
            raise ValueError("Received non-task response from remote agent")

        parts = []
        for artifact in send_response.root.result.artifacts or []:
            for part in artifact.parts or []:
                if part.root.type == "text":
                    parts.append(part.root.text)

        return "\n".join(parts) if parts else "Remote agent completed with no text output."

    def invoke(self, query: str, session_id: str) -> str:
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        response = self.graph.invoke({"messages": [("user", query)]}, config=config)
        return response["messages"][-1].content
