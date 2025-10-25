#!/usr/bin/env python3
"""Static story generator placeholder for Pumpkin AI."""

from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        payload = {}

    name = (payload.get("name") or "Friend").strip() or "Friend"
    character = (payload.get("character") or "pumpkin").strip() or "pumpkin"
    food = (payload.get("food") or "candy").strip() or "candy"

    story = (
        f"{name} the {character} wandered through the glowing pumpkin patch, "
        f"sharing stories about their love of {food}. The night breeze carried their laughter, "
        f"and even the jack-o'-lanterns seemed to smile a little brighter."
    )

    json.dump({"text": story}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
