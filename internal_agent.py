import os
import json
from dotenv import load_dotenv
import anthropic
from internal_systems import (
    check_shared_drive_access, check_license_availability,
    check_vpn_status, check_badge_access,
)

load_dotenv()
client = anthropic.Anthropic()

MAX_TOOL_CALLS = 2  # hard cap - guards against runaway cost/latency

TOOLS = [
    {
        "name": "check_shared_drive_access",
        "description": "Check whether an employee has access to the shared drive.",
        "input_schema": {
            "type": "object",
            "properties": {"employee_id": {"type": "string", "description": "e.g. E1001"}},
            "required": ["employee_id"],
        },
    },
    {
        "name": "check_license_availability",
        "description": "Check available seats for a software license.",
        "input_schema": {
            "type": "object",
            "properties": {"software_name": {"type": "string", "description": "e.g. 'Visual Studio'"}},
            "required": ["software_name"],
        },
    },
    {
        "name": "check_vpn_status",
        "description": "Check current VPN/network status for an office location.",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "e.g. 'pune-office'"}},
            "required": ["location"],
        },
    },
    {
        "name": "check_badge_access",
        "description": "Check whether an employee's building badge is currently active.",
        "input_schema": {
            "type": "object",
            "properties": {"employee_id": {"type": "string", "description": "e.g. E1001"}},
            "required": ["employee_id"],
        },
    },
]

TOOL_FUNCTIONS = {
    "check_shared_drive_access": check_shared_drive_access,
    "check_license_availability": check_license_availability,
    "check_vpn_status": check_vpn_status,
    "check_badge_access": check_badge_access,
}

SYSTEM_PROMPT = """You are an internal IT support agent. You have tools to look up real
employee access, license availability, VPN status, and badge status. Use a tool whenever
the ticket requires real data you don't already have - never guess or make up a status or
number. If a ticket doesn't give you enough information to use a tool (e.g. no employee ID
given), ask for it instead of guessing."""

def answer_internal_ticket(ticket_text: str) -> str:
    messages = [{"role": "user", "content": ticket_text}]
    tool_calls_used = 0

    while tool_calls_used < MAX_TOOL_CALLS:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                fn = TOOL_FUNCTIONS[block.name]
                result = fn(**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
        messages.append({"role": "user", "content": tool_results})
        tool_calls_used += 1

    return "I wasn't able to resolve this within my tool-call limit. Escalating to a human."

if __name__ == "__main__":
    tickets = [
        "Employee E1015 says they can't access the shared drive, can you check?",
        "Do we have any free Figma licenses for a new hire?",
        "Is the VPN down at the Delhi office?",
        "Employee E1059's badge isn't working at the door.",
        "My internet is slow.",  # no tool applies - should ask for detail, not guess
    ]
    for t in tickets:
        print(f"TICKET: {t}")
        print(f"ANSWER: {answer_internal_ticket(t)}")
        print("-" * 60)
