"""Summarize benchmark loss traces and locate immediate tactical mistakes."""
import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(path):
    counts = Counter()
    mistakes = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["type"] == "run":
                if record["schema_version"] != 1:
                    raise ValueError("Unsupported trace schema")
                continue
            counts[record["outcome"]] += 1
            if record["outcome"] != "losses":
                continue
            counts[f"losses_as_{record['candidate_seat']}"] += 1
            counts[f"losses_vs_{record['opponent']}"] += 1
            found = False
            for move in record["moves"]:
                tactics = move.get("tactics", {})
                reasons = [key for key in ("missed_immediate_win", "avoidable_immediate_reply") if tactics.get(key)]
                if reasons:
                    found = True
                    mistakes.append(dict(game_id=record["game_id"], ply=move["ply"],
                                         action=move["action"], reasons=reasons,
                                         winning_actions=tactics["immediate_wins"], safe_actions=tactics["safe_actions"]))
            if not found:
                counts["losses_without_detected_one_ply_mistake"] += 1
    return dict(counts=dict(counts), mistakes=mistakes,
                note="No detected one-ply mistake does not establish correct deeper play. See move diagnostics for priors, visits and Q estimates.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.output and Path(args.output).resolve() == Path(args.trace).resolve():
        parser.error("output must be separate from input trace")
    text = json.dumps(summarize(args.trace), indent=2)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
