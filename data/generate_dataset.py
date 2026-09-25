"""Synthesize messy 311 complaint / structured ticket pairs for instruction tuning.

Each example pairs a realistically noisy resident complaint (the kind a human
would actually type into a web form or read off a phone call) with the clean,
structured dispatch ticket a city worker would want to act on. The complaint
and the ticket are both derived from the same underlying "facts" for a given
category, so the two sides are guaranteed to agree with each other.

Usage:
    python generate_dataset.py --n-per-category 30 --seed 7 --output-dir .
"""

import argparse
import json
import random
from pathlib import Path

INSTRUCTION = (
    "Rewrite the resident's complaint below into a structured 311 dispatch "
    "ticket. Respond with a single JSON object with exactly these fields: "
    "category, priority (low, medium, or high), location, summary, details, "
    "requested_action."
)

STREET_NAMES = [
    "Maple", "Oak", "5th", "Elm", "Washington", "Sunset", "River", "Pine",
    "Cedar", "Franklin", "Lincoln", "Park", "Highland", "Chestnut", "Birch",
]
STREET_TYPES = ["St", "Ave", "Rd", "Blvd", "Ln", "Dr"]


def random_location(rng):
    number = rng.randint(100, 9999)
    street = f"{rng.choice(STREET_NAMES)} {rng.choice(STREET_TYPES)}"
    if rng.random() < 0.4:
        cross = f"{rng.choice(STREET_NAMES)} {rng.choice(STREET_TYPES)}"
        return f"{number} {street}", f"near {street} and {cross}"
    return f"{number} {street}", f"{number} {street}"


# ---------------------------------------------------------------------------
# Noise injection: turns clean sentences into messy, human-typed complaints.
# ---------------------------------------------------------------------------

FILLERS = ["literally", "honestly", "like", "smh", "ugh", "seriously", "omg"]
SLANG_SUBS = {
    "please": "pls", "people": "ppl", "right now": "rn", "as soon as possible": "asap",
    "do not": "dont", "does not": "doesnt", "cannot": "cant", "it is": "its",
    "I am": "im", "you are": "ur", "going to": "gonna",
}


def add_slang(text, rng):
    for full, short in SLANG_SUBS.items():
        if full in text and rng.random() < 0.5:
            text = text.replace(full, short)
    return text


def add_typo(word, rng):
    if len(word) < 4:
        return word
    i = rng.randint(0, len(word) - 2)
    chars = list(word)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def scatter_typos(text, rng, rate=0.06):
    words = text.split(" ")
    for i, w in enumerate(words):
        if any(ch.isdigit() for ch in w):
            continue  # never garble street numbers, they must match the ticket
        if rng.random() < rate:
            words[i] = add_typo(w, rng)
    return " ".join(words)


def shout_case(text, rng):
    words = text.split(" ")
    if len(words) < 3:
        return text
    n_shout = rng.randint(1, 2)
    for _ in range(n_shout):
        idx = rng.randint(0, len(words) - 1)
        words[idx] = words[idx].upper()
    return " ".join(words)


def strip_some_punctuation(text, rng):
    if rng.random() < 0.5:
        text = text.replace(",", "")
    if rng.random() < 0.4:
        text = text.rstrip(".")
    return text


def add_venting_prefix(text, rng):
    prefixes = [
        "This is the third time I'm calling about this. ",
        "Can someone PLEASE deal with this. ",
        "Not sure who to even call but. ",
        "I've lived here 20 years and never seen it this bad. ",
        "",
        "",
        "",
    ]
    return rng.choice(prefixes) + text


def add_filler_word(text, rng):
    if rng.random() < 0.4:
        words = text.split(" ")
        idx = rng.randint(0, len(words) - 1)
        words.insert(idx, rng.choice(FILLERS))
        return " ".join(words)
    return text


def messify(text, rng):
    text = add_venting_prefix(text, rng)
    text = add_filler_word(text, rng)
    text = add_slang(text, rng)
    text = scatter_typos(text, rng)
    text = strip_some_punctuation(text, rng)
    text = shout_case(text, rng)
    return text


# ---------------------------------------------------------------------------
# Category definitions. Each generates a "facts" dict, then renders a messy
# complaint template and a clean ticket from those same facts.
# ---------------------------------------------------------------------------

def gen_pothole(rng):
    _, loc = random_location(rng)
    size = rng.choice(["small", "large", "several"])
    hazard = rng.choice([
        "it blew out my tire", "a motorcyclist almost lost control on it",
        "it's damaging every car that hits it", None,
    ])
    duration = rng.choice(["a few days", "over a week", "almost a month"])

    complaint = f"theres a {size} pothole on {loc} been there for {duration}"
    if hazard:
        complaint += f", {hazard}"
    complaint += ", someone needs to fix this before it gets worse"

    priority = "high" if hazard else "medium"
    summary = f"{size.capitalize()} pothole reported on {loc}"
    details = f"Pothole present for {duration}."
    if hazard:
        details += f" Resident reports: {hazard}."
    action = "Dispatch street maintenance crew to inspect and patch pothole"
    return loc, complaint, priority, summary, details, action


def gen_noise(rng):
    _, loc = random_location(rng)
    noise_type = rng.choice(["loud music", "a barking dog", "construction work", "a house party"])
    time_of_day = rng.choice(["late at night", "since 2am", "every morning before 6am", "all afternoon"])
    repeated = rng.random() < 0.5

    complaint = f"neighbors at {loc} have had {noise_type} going {time_of_day}"
    if repeated:
        complaint += " and this happens almost every week"
    complaint += " I can't get any sleep"

    priority = "high" if repeated and "night" in time_of_day else "medium"
    summary = f"Noise complaint: {noise_type} near {loc}"
    details = f"Reported {noise_type} occurring {time_of_day}."
    if repeated:
        details += " Resident states this is a recurring issue."
    action = "Send code enforcement or police for noise ordinance check"
    return loc, complaint, priority, summary, details, action


def gen_dumping(rng):
    _, loc = random_location(rng)
    items = rng.choice(["old furniture", "tires", "construction debris", "several trash bags", "a mattress"])
    blocking = rng.choice(["the sidewalk", "the alley", "a vacant lot", None])

    complaint = f"somebody dumped {items} on {loc}"
    if blocking:
        complaint += f" and its blocking {blocking}"
    complaint += " this is disgusting and needs to be picked up"

    priority = "high" if blocking == "the sidewalk" or blocking == "the alley" else "medium"
    summary = f"Illegal dumping of {items} at {loc}"
    details = f"{items.capitalize()} dumped at location."
    if blocking:
        details += f" Blocking {blocking}."
    action = "Dispatch sanitation crew for illegal dumping cleanup"
    return loc, complaint, priority, summary, details, action


def gen_water_leak(rng):
    _, loc = random_location(rng)
    source = rng.choice(["a fire hydrant", "a broken water main", "a pipe under the street"])
    severity = rng.choice(["a small trickle", "flooding the whole street", "flooding my basement"])

    complaint = f"{source} near {loc} is leaking bad, its {severity}"
    priority = "high" if "flooding" in severity else "medium"
    summary = f"Water leak from {source} at {loc}"
    details = f"Leak from {source}, described as {severity}."
    action = "Dispatch water utility crew to inspect and repair leak"
    return loc, complaint, priority, summary, details, action


def gen_graffiti(rng):
    _, loc = random_location(rng)
    surface = rng.choice(["a wall", "a stop sign", "the underpass", "a park bench"])
    offensive = rng.random() < 0.3

    complaint = f"theres graffiti all over {surface} at {loc}"
    if offensive:
        complaint += " and some of it is really offensive language"
    complaint += " can someone come clean it up"

    priority = "high" if offensive else "low"
    summary = f"Graffiti reported on {surface} near {loc}"
    details = "Graffiti covering surface."
    if offensive:
        details += " Contains offensive language."
    action = "Dispatch public works crew for graffiti removal"
    return loc, complaint, priority, summary, details, action


def gen_streetlight(rng):
    _, loc = random_location(rng)
    duration = rng.choice(["a couple days", "over a week", "as long as I can remember"])
    safety = rng.random() < 0.5

    complaint = f"the streetlight at {loc} has been out for {duration}"
    if safety:
        complaint += " and its really dark and dangerous walking there at night"

    priority = "high" if safety else "medium"
    summary = f"Streetlight outage at {loc}"
    details = f"Light has been out for {duration}."
    if safety:
        details += " Resident reports safety concern due to darkness."
    action = "Dispatch electrical crew to repair streetlight"
    return loc, complaint, priority, summary, details, action


def gen_trash(rng):
    _, loc = random_location(rng)
    service = rng.choice(["trash", "recycling", "yard waste"])
    repeated = rng.random() < 0.35

    complaint = f"my {service} pickup at {loc} got missed again this week"
    if repeated:
        complaint += " this is like the third time this month"

    priority = "medium" if repeated else "low"
    summary = f"Missed {service} pickup at {loc}"
    details = f"{service.capitalize()} was not collected on scheduled day."
    if repeated:
        details += " Resident reports repeated missed pickups."
    action = f"Notify sanitation route supervisor to schedule makeup {service} pickup"
    return loc, complaint, priority, summary, details, action


def gen_animal(rng):
    _, loc = random_location(rng)
    animal = rng.choice(["a stray dog", "loose chickens", "an injured raccoon", "an aggressive dog off leash"])
    urgent = "aggressive" in animal or "injured" in animal

    complaint = f"theres {animal} wandering around near {loc}"
    if urgent:
        complaint += " I'm scared to let my kids outside"

    priority = "high" if urgent else "medium"
    summary = f"Animal control needed near {loc}"
    details = f"Report of {animal} in the area."
    action = "Dispatch animal control officer"
    return loc, complaint, priority, summary, details, action


def gen_parking(rng):
    _, loc = random_location(rng)
    violation = rng.choice([
        "blocking my driveway", "parked in front of a fire hydrant",
        "in the handicap spot without a permit", "an abandoned car that hasn't moved in weeks",
    ])

    complaint = f"theres a car {violation} at {loc}"
    priority = "high" if "hydrant" in violation or "handicap" in violation else "medium"
    summary = f"Parking violation reported at {loc}"
    details = f"Vehicle {violation}."
    action = "Dispatch parking enforcement to ticket or tow vehicle"
    return loc, complaint, priority, summary, details, action


def gen_tree(rng):
    _, loc = random_location(rng)
    issue = rng.choice([
        "a large branch fell and is blocking the road", "a tree is leaning over power lines",
        "overgrown branches are blocking the sidewalk", "a tree fell into the street after last night's storm",
    ])
    urgent = "blocking the road" in issue or "power lines" in issue or "fell into the street" in issue

    complaint = f"at {loc} {issue}"
    priority = "high" if urgent else "low"
    summary = f"Tree/debris hazard at {loc}"
    details = issue.capitalize() + "."
    action = "Dispatch forestry/public works crew to clear hazard"
    return loc, complaint, priority, summary, details, action


CATEGORIES = {
    "pothole": gen_pothole,
    "noise_complaint": gen_noise,
    "illegal_dumping": gen_dumping,
    "water_leak": gen_water_leak,
    "graffiti": gen_graffiti,
    "streetlight_outage": gen_streetlight,
    "missed_trash_pickup": gen_trash,
    "animal_control": gen_animal,
    "parking_violation": gen_parking,
    "tree_debris": gen_tree,
}


def build_example(category, rng):
    generator = CATEGORIES[category]
    location, complaint, priority, summary, details, action = generator(rng)
    messy_complaint = messify(complaint, rng)

    ticket = {
        "category": category,
        "priority": priority,
        "location": location,
        "summary": summary,
        "details": details,
        "requested_action": action,
    }
    return messy_complaint, ticket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-category", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    examples = []
    for category in CATEGORIES:
        for _ in range(args.n_per_category):
            complaint, ticket = build_example(category, rng)
            examples.append({
                "instruction": INSTRUCTION,
                "input": complaint,
                "output": json.dumps(ticket, ensure_ascii=False),
            })

    rng.shuffle(examples)
    n_val = int(len(examples) * args.val_fraction)
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in [("all", examples), ("train", train_examples), ("val", val_examples)]:
        path = args.output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for ex in subset:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"wrote {len(subset)} examples to {path}")


if __name__ == "__main__":
    main()
