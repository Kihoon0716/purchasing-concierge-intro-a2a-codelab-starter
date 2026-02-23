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

import os
import uuid

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

load_dotenv()

memory = MemorySaver()


class OrderItem(BaseModel):
    name: str
    quantity: int
    price: int


class Order(BaseModel):
    order_id: str
    status: str
    order_items: list[OrderItem]


@tool
def create_burger_order(order_items: list[OrderItem]) -> str:
    """Creates a new burger order with the given order items."""
    try:
        order_id = str(uuid.uuid4())
        order = Order(order_id=order_id, status="created", order_items=order_items)
        print("===")
        print(f"order created: {order}")
        print("===")
    except Exception as e:
        print(f"Error creating order: {e}")
        return f"Error creating order: {e}"
    return f"Order {order.model_dump()} has been created"


class BurgerSellerAgent:
    SYSTEM_INSTRUCTION = """
# INSTRUCTIONS

You are a specialized assistant for a burger store.
Your sole purpose is to answer questions about what is available on burger menu and price also handle order creation.
If the user asks about anything other than burger menu or order creation, politely state that you cannot help with that topic and can only assist with burger menu and order creation.

# CONTEXT

Provided below is the available burger menu and its related price:
- Classic Cheeseburger: IDR 85K
- Double Cheeseburger: IDR 110K
- Spicy Chicken Burger: IDR 80K
- Spicy Cajun Burger: IDR 85K

# RULES

- For order creation:
    1. Ensure the user confirmed the order and total price (possibly already in their prompt)
    2. Use `create_burger_order`
    3. Return detailed ordered items, price breakdown, total, and order ID
- Never make up menu items or prices.
"""
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        self.model = ChatOpenAI(
            model=openai_model,
            base_url=openai_base_url,
            api_key=openai_api_key,
            temperature=0,
        )
        self.graph = create_react_agent(
            self.model,
            tools=[create_burger_order],
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
        )

    def invoke(self, query: str, sessionId: str) -> str:
        config = {"configurable": {"thread_id": sessionId}}
        response = self.graph.invoke({"messages": [("user", query)]}, config=config)
        return response["messages"][-1].content


if __name__ == "__main__":
    agent = BurgerSellerAgent()
    result = agent.invoke("1 classic cheeseburger pls", "default_session")
    print(result)
