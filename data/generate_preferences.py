"""Build a chosen-vs-rejected preference dataset for a DPO pass on top of the
SFT adapter.

For each SFT example (instruction + messy complaint, with the ground-truth
ticket as `chosen`), synthesize a `rejected` ticket by injecting one or two
realistic failure modes a freshly-SFT'd model might still produce: a flipped
priority, a confused-but-plausible category, vague boilerplate details,
uncleaned verbatim complaint text dropped into `details`, a dropped location,
a generic non-specific action, or a chatty non-JSON wrapper around the ticket.

Usage:
    python generate_preferences.py --input train.jsonl --output dpo_train.jsonl --seed 11
"""

import argparse
import json
import random
from pathlib import Path

# Categories a rewriting model could plausibly confuse for one another.
CONFUSABLE_CATEGORIES = {
    "pothole": "water_leak",
    "water_leak": "pothole",
    "noise_complaint": "animal_control",
    "animal_control": "noise_complaint",
    "illegal_dumping": "missed_trash_pickup",
    "missed_trash_pickup": "illegal_dumping",
    "streetlight_outage": "tree_debris",
    "tree_debris": "streetlight_outage",
    "graffiti": "parking_violation",
    "parking_violation": "graffiti",
}

ACTION_BY_CATEGORY = {
    "pothole": "Dispatch street maintenance crew to inspect and patch pothole",
    "noise_complaint": "Send code enforcement or police for noise ordinance check",
    "illegal_dumping": "Dispatch sanitation crew for illegal dumping cleanup",
    "water_leak": "Dispatch water utility crew to inspect and repair leak",
    "graffiti": "Dispatch public works crew for graffiti removal",
    "streetlight_outage": "Dispatch electrical crew to repair streetlight",
    "missed_trash_pickup": "Notify sanitation route supervisor to schedule makeup pickup",
    "animal_control": "Dispatch animal control officer",
    "parking_violation": "Dispatch parking enforcement to ticket or tow vehicle",
    "tree_debris": "Dispatch forestry/public works crew to clear hazard",
}

GENERIC_ACTIONS = [
    "Dispatch appropriate city crew to address the issue",
    "Forward to relevant department for follow-up",
    "Assign to general services for review",
]


def flip_priority(ticket, complaint, rng):
    levels = [lvl for lvl in ("low", "medium", "high") if lvl != ticket["priority"]]
    new = dict(ticket)
    new["priority"] = rng.choice(levels)
    return new


def swap_category(ticket, complaint, rng):
    alt = CONFUSABLE_CATEGORIES.get(ticket["category"])
    if alt is None:
        return None
    new = dict(ticket)
    new["category"] = alt
    new["requested_action"] = ACTION_BY_CATEGORY[alt]
    return new


def vague_details(ticket, complaint, rng):
    new = dict(ticket)
    new["summary"] = "Resident reported an issue in the area"
    new["details"] = "Resident called about a problem. Needs follow-up."
    return new


def verbatim_dump(ticket, complaint, rng):
    new = dict(ticket)
    new["details"] = complaint
    new["summary"] = complaint[:60]
    return new


def missing_location(ticket, complaint, rng):
    new = dict(ticket)
    new["location"] = "unknown"
    return new


def generic_action(ticket, complaint, rng):
    new = dict(ticket)
    new["requested_action"] = rng.choice(GENERIC_ACTIONS)
    return new


FLAW_FUNCS = [
    flip_priority, swap_category, vague_details,
    verbatim_dump, missing_location, generic_action,
]


def format_violation(ticket, rng):
    ticket_json = json.dumps(ticket, ensure_ascii=False)
    wrappers = [
        f"Sure! Here's the ticket:\n\n{ticket_json}",
        f"{ticket_json}\n\nLet me know if you need anything else!",
        f"Here is the structured ticket you requested:\n{ticket_json}\nHope that helps.",
    ]
    return rng.choice(wrappers)


def build_rejected_ticket(chosen_ticket, complaint, rng):
    ticket = dict(chosen_ticket)
    n_flaws = 1 if rng.random() < 0.75 else 2
    for fn in rng.sample(FLAW_FUNCS, k=n_flaws):
        result = fn(ticket, complaint, rng)
        if result is not None:
            ticket = result
    if ticket == chosen_ticket:
        ticket = flip_priority(ticket, complaint, rng)
    return ticket


def build_rejected(chosen_ticket, complaint, rng):
    if rng.random() < 0.15:
        return format_violation(chosen_ticket, rng)
    return json.dumps(build_rejected_ticket(chosen_ticket, complaint, rng), ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pairs = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            chosen_ticket = json.loads(ex["output"])
            rejected = build_rejected(chosen_ticket, ex["input"], rng)
            assert rejected != ex["output"], "rejected must differ from chosen"
            pairs.append({
                "instruction": ex["instruction"],
                "input": ex["input"],
                "chosen": ex["output"],
                "rejected": rejected,
            })

    with args.output.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} preference pairs to {args.output}")


if __name__ == "__main__":
    main()
