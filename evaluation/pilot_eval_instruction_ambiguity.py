import base64
import io
import json
from pathlib import Path
from openai import OpenAI
from PIL import Image


DEFAULT_PROMPT = """
You are evaluating ambiguity in a Vision-Language Navigation (VLN) task.

Input:
- A human-written navigation instruction.
- A set of uniformly sampled observation frames from the ground-truth trajectory of a successful navigation episode.

Objective:
Your goal is to assess whether the task itself contains instruction ambiguity, not whether the trajectory is correct or successful. Even though the provided episode is a ground-truth successful example, the instruction may still be ambiguous.

Ambiguity types:

1. Directional Ambiguity
This occurs if, at some intermediate stage of navigation, the instruction does not uniquely specify the intended path or orientation, such that multiple plausible route choices, branches, turns, or movement directions could reasonably satisfy the instruction.

2. Docking Ambiguity
This occurs if, near the stopping stage, the instruction does not uniquely specify the final stopping location or target object, such that multiple nearby endpoints or candidate targets could reasonably satisfy the instruction.

Evaluation criteria:
- Judge ambiguity based on the instruction together with the observed scene context.
- Evaluate whether a reasonable agent could form more than one plausible interpretation.
- Do not judge action quality, policy quality, or trajectory correctness.
- Do not assume that a successful ground-truth trajectory means the instruction is unambiguous.
- Do not treat sparse observations or incomplete visual evidence alone as ambiguity.
- Mark ambiguity only when there is a genuine and reasonable alternative interpretation supported by the instruction and scene context.
- Directional Ambiguity concerns ambiguity during movement before the final stopping stage.
- Docking Ambiguity concerns ambiguity about the final stopping place or target at the end.
- The two types can both exist in the same episode.

Instruction:
{instruction}

Respond with exactly two integers separated by a comma:
- First integer: 1 if Directional Ambiguity exists, 0 otherwise
- Second integer: 1 if Docking Ambiguity exists, 0 otherwise

Example output:
1,0
"""


def encode_image(image_path: str, max_size: int = 320) -> str:
    img = Image.open(image_path)
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def sample_frames(images: list[str], n: int) -> list[str]:
    if len(images) <= n:
        return images
    indices = [int(i * (len(images) - 1) / (n - 1)) for i in range(n)]
    return [images[i] for i in indices]


def evaluate_ambiguity(
    images: list[str],
    instruction: str,
    prompt: str = DEFAULT_PROMPT,
    model: str = "gpt-4o",
    api_key: str = None,  # Set via environment variable or pass explicitly
    base_url: str = "https://api.openai.com/v1/",  # Default OpenAI endpoint
    num_frames: int = 8,
) -> tuple[int, int]:
    """
    Evaluate instruction ambiguity for a navigation episode.

    Args:
        images: List of image file paths
        instruction: Navigation instruction text
        prompt: Prompt template with {instruction} placeholder
        model: Model name
        api_key: API key
        base_url: API base URL

    Returns:
        (directional_ambiguity, docking_ambiguity) each 0 or 1
    """
    client = OpenAI(api_key=api_key, base_url=base_url)

    images = sample_frames(images, num_frames)

    content = [{"type": "text", "text": prompt.format(instruction=instruction)}]
    for img_path in images:
        b64 = encode_image(img_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0,
            )
            break
        except Exception as e:
            if attempt == 4:
                raise
            import time
            print(f"Attempt {attempt+1} failed: {e}, retrying in {2**attempt}s...")
            time.sleep(2 ** attempt)

    raw = response.choices[0].message.content.strip()
    parts = raw.split(",")
    return int(parts[0].strip()), int(parts[1].strip()), raw


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_json", type=str, required=True, help="Path to summary.json (one episode per line)")
    parser.add_argument("--images_dir", type=str, required=True, help="Root directory containing episode image folders")
    parser.add_argument("--output", type=str, default="ambiguity_results.jsonl")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--base_url", type=str, default=None, help="Optional API base URL (defaults to OpenAI)")
    args = parser.parse_args()

    with open(args.summary_json) as f:
        episodes = [json.loads(line) for line in f if line.strip()]

    # Resume: load already-processed episode IDs
    done_ids = set()
    if Path(args.output).exists():
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["episode_id"])
    print(f"Resuming: {len(done_ids)} already done, {len(episodes) - len(done_ids)} remaining.")

    with open(args.output, "a") as out_f:
        for ep in episodes:
            episode_id = ep["id"]
            if episode_id in done_ids:
                continue
            instruction = ep["instructions"][0]
            video_dir = Path(args.images_dir) / ep["video"] / "rgb"
            images = sorted(str(p) for p in video_dir.glob("*.jpg"))

            if not images:
                print(f"No images found for episode {episode_id}, skipping.")
                continue

            try:
                da, doa, raw = evaluate_ambiguity(
                    images=images,
                    instruction=instruction,
                    model=args.model,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    num_frames=args.num_frames,
                )
                result = {
                    "episode_id": episode_id,
                    "instruction": instruction,
                    "directional_ambiguity": da,
                    "docking_ambiguity": doa,
                    "raw_response": raw,
                }
                print(result)
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
            except Exception as e:
                print(f"Error on episode {episode_id}: {e}")
