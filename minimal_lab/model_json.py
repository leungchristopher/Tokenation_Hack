"""Decode model JSON without rejecting a harmless Markdown code fence."""
import json


def decode_reply(completion):
    text = completion.strip()
    if text.startswith('```') and text.endswith('```'):
        lines = text.splitlines()
        if lines[0].strip().lower() not in ('```', '```json'):
            raise ValueError('Expected JSON, not another fenced format.')
        text = '\n'.join(lines[1:-1])
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError('Expected a JSON object.')
    return result
