"""End-to-end test client for the Legal Multi-Agent System.

Sends a legal question to the Customer Agent, prints the response, and
reports end-to-end latency metrics.
"""

import argparse
import asyncio
import os
import statistics
import sys
import time
from uuid import uuid4

import httpx
from dotenv import load_dotenv

load_dotenv()

CUSTOMER_AGENT_URL = os.getenv("CUSTOMER_AGENT_URL", "http://localhost:10100")
DEFAULT_QUESTION = (
    "If a company breaks a contract and avoids taxes, "
    "what are the legal and regulatory consequences?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a test question to the Customer Agent.")
    parser.add_argument(
        "--question",
        default=os.getenv("TEST_CLIENT_QUESTION", DEFAULT_QUESTION),
        help="Question to send to the distributed multi-agent system.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=int(os.getenv("TEST_CLIENT_RUNS", "1")),
        help="Number of end-to-end runs for latency measurement.",
    )
    return parser.parse_args()


def extract_text_response(response: object) -> str:
    """Extract concatenated text parts from an A2A response object."""
    result_text = ""
    if hasattr(response, "root"):
        root = response.root
        if hasattr(root, "result"):
            result = root.result
            if hasattr(result, "artifacts") and result.artifacts:
                for artifact in result.artifacts:
                    for part in artifact.parts:
                        p = part.root if hasattr(part, "root") else part
                        if hasattr(p, "text"):
                            result_text += p.text
            elif hasattr(result, "parts") and result.parts:
                for part in result.parts:
                    p = part.root if hasattr(part, "root") else part
                    if hasattr(p, "text"):
                        result_text += p.text
    return result_text


async def send_question(client: object, question: str) -> tuple[str, float]:
    """Send one question through A2A and return response text with latency."""
    from a2a.types import Message, Part, Role, SendMessageRequest, TextPart, MessageSendParams

    message = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text=question))],
        message_id=str(uuid4()),
    )
    request = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(message=message),
    )

    started = time.perf_counter()
    response = await client.send_message(request)
    elapsed = time.perf_counter() - started
    return extract_text_response(response), elapsed


async def main() -> None:
    args = parse_args()
    runs = max(1, args.runs)

    print(f"Connecting to Customer Agent at {CUSTOMER_AGENT_URL}")
    print(f"Question: {args.question}")
    print(f"Runs: {runs}")
    print("-" * 60)

    async with httpx.AsyncClient(timeout=300.0) as http_client:
        card_url = f"{CUSTOMER_AGENT_URL}/.well-known/agent.json"
        try:
            card_resp = await http_client.get(card_url)
            card_resp.raise_for_status()
        except Exception as exc:
            print(f"ERROR: Could not reach Customer Agent at {card_url}")
            print(f"  {exc}")
            print("Make sure all services are running (./start_all.sh)")
            sys.exit(1)

        from a2a.client import A2AClient
        from a2a.types import AgentCard

        agent_card = AgentCard.model_validate(card_resp.json())
        print(f"Connected to agent: {agent_card.name} v{agent_card.version}")
        print("-" * 60)

        client = A2AClient(httpx_client=http_client, agent_card=agent_card)
        latencies: list[float] = []
        last_response = ""

        for run_index in range(1, runs + 1):
            print(f"Run {run_index}/{runs}: sending request...\n")
            result_text, latency = await send_question(client, args.question)
            latencies.append(latency)
            last_response = result_text
            print(f"Run {run_index} latency: {latency:.2f}s")
            if runs > 1:
                print("-" * 60)

        if last_response:
            print("RESPONSE:")
            print("=" * 60)
            print(last_response)
            print("=" * 60)
        else:
            print("No text response received from the agent.")

        print("LATENCY METRICS")
        print("=" * 60)
        print(f"Total runs: {len(latencies)}")
        print(f"Average: {statistics.mean(latencies):.2f}s")
        print(f"Min: {min(latencies):.2f}s")
        print(f"Max: {max(latencies):.2f}s")
        if len(latencies) > 1:
            print(f"Median: {statistics.median(latencies):.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
