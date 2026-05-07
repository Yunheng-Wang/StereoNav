import json
import argparse
from openai import OpenAI

PROMPT = """
You are a navigation assistant. Extract structured information from the user's natural language navigation command.

The robot uses a coordinate system where:
- X axis: forward direction (positive = forward)
- Y axis: left direction (positive = left, negative = right)
- Origin: robot's current position

Extract:
1. instruction: a clean English navigation instruction suitable for a VLN model
2. target_x: estimated target X coordinate in meters (forward distance)
3. target_y: estimated target Y coordinate in meters (left offset, negative if right)

User input: {user_input}

Respond with ONLY a JSON object, no markdown:
{{"instruction": "...", "target_x": <float>, "target_y": <float>}}
"""


def parse(user_input: str, api_key: str, base_url: str, model: str) -> dict:
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT.format(user_input=user_input)}],
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()

    result = parse(args.input, args.api_key, args.base_url, args.model)
    print(f'INSTRUCTION={json.dumps(result["instruction"])}')
    print(f'TARGET_X={result["target_x"]}')
    print(f'TARGET_Y={result["target_y"]}')
