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

import uuid

import gradio as gr
from dotenv import load_dotenv

from purchasing_concierge.agent import root_agent

load_dotenv()

SESSION_ID = str(uuid.uuid4())


def get_response_from_agent(message: str, history):
    """Send a user message to the local purchasing concierge agent."""
    del history
    return root_agent.invoke(message, session_id=SESSION_ID)


if __name__ == "__main__":
    demo = gr.ChatInterface(
        get_response_from_agent,
        title="Purchasing Concierge (Local)",
        description="Local LangGraph orchestrator that delegates to A2A seller agents.",
        type="messages",
    )

    demo.launch(server_name="0.0.0.0", server_port=8080)
