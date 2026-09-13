"""
Hand-written reasoning layer. No frameworks — plain Python rules.

Pipeline:
  question + image
    -> classify_intent()      decides if the detector is even needed, and which
                               specific PPE item (if any) the question is about
    -> run_detection()        (only if needed)
    -> match_workers_to_ppe() links each Person to nearby PPE boxes
    -> compute_worker_status()applies the confidence guardrail per PPE item
    -> build_answer()         turns structured facts into a sentence
"""

from app.services.detector import run_detection

# ---- Config: what counts as a "required" PPE item, and where on the body to expect it ----
REQUIRED_PPE = {
    "helmet": {"positive": "Hardhat", "negative": "NO-Hardhat", "region": "head"},
    "vest":   {"positive": "Safety Vest", "negative": "NO-Safety Vest", "region": "torso"},
    "mask":   {"positive": "Mask", "negative": "NO-Mask", "region": "head"},
}

# Below this confidence, we don't trust the detection enough to make a claim about it.
# Chosen because it's the same threshold used during evaluation/failure analysis (Day 1),
# so live API behavior stays consistent with the reported metrics.
CONFIDENCE_THRESHOLD = 0.5


# STEP 1: Spatial matching — link each Person box to nearby PPE boxes


def _get_body_regions(person_bbox):
    """
    Given a person's box, estimate where their head and torso should be.
    Regions are relative to the person's own box size, so this scales
    naturally whether the person is close-up (big box) or far away (small box)
    - unlike a fixed pixel distance, which breaks across different image scales.
    """
    x1, y1, x2, y2 = person_bbox
    h = y2 - y1
    w = x2 - x1
    margin_x = w * 0.3   # allow PPE box to stick out sideways a bit
    margin_y = h * 0.15  # allow a helmet to sit above the person's box top

    head_region = [x1 - margin_x, y1 - margin_y, x2 + margin_x, y1 + h * 0.35]
    torso_region = [x1 - margin_x, y1 + h * 0.15, x2 + margin_x, y1 + h * 0.75]

    return {"head": head_region, "torso": torso_region}


def _box_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _center_in_region(center, region):
    cx, cy = center
    rx1, ry1, rx2, ry2 = region
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def match_workers_to_ppe(detections):
    """
    Groups detections into per-worker profiles.
    Returns a list of dicts, one per detected Person.
    """
    persons = [d for d in detections if d["class"] == "Person"]
    ppe_items = [d for d in detections if d["class"] not in ("Person", "Safety Cone", "machinery", "vehicle")]

    workers = []
    for person in persons:
        regions = _get_body_regions(person["bbox"])
        nearby = {"head": [], "torso": []}

        for item in ppe_items:
            item_center = _box_center(item["bbox"])
            # A PPE item belongs to whichever region (head/torso) it falls inside.
            if _center_in_region(item_center, regions["head"]):
                nearby["head"].append(item)
            elif _center_in_region(item_center, regions["torso"]):
                nearby["torso"].append(item)

        workers.append({
            "person_confidence": person["confidence"],
            "person_bbox": person["bbox"],
            "nearby_detections": nearby,
        })

    return workers


# STEP 2: Confidence guardrail — decide compliance status per PPE item


def compute_worker_status(worker):
    item_statuses = {}

    for item_name, cfg in REQUIRED_PPE.items():
        region_items = worker["nearby_detections"][cfg["region"]]

        positive_matches = [d for d in region_items if d["class"] == cfg["positive"]]
        negative_matches = [d for d in region_items if d["class"] == cfg["negative"]]

        best_positive = max((d["confidence"] for d in positive_matches), default=0)
        best_negative = max((d["confidence"] for d in negative_matches), default=0)

        if best_negative >= CONFIDENCE_THRESHOLD and best_negative >= best_positive:
            item_statuses[item_name] = {"status": "VIOLATION", "confidence": best_negative}
        elif best_positive >= CONFIDENCE_THRESHOLD and best_positive > best_negative:
            item_statuses[item_name] = {"status": "COMPLIANT", "confidence": best_positive}
        else:
            item_statuses[item_name] = {"status": "INSUFFICIENT_INFO", "confidence": max(best_positive, best_negative)}

    violations = [k for k, v in item_statuses.items() if v["status"] == "VIOLATION"]
    unknowns = [k for k, v in item_statuses.items() if v["status"] == "INSUFFICIENT_INFO"]

    if violations:
        overall = "AT_RISK"          # a confirmed violation always wins, regardless of unknowns
    elif unknowns:
        overall = "PARTIALLY_CONFIRMED"   # some items checked out fine, others we just don't know about
    else:
        overall = "COMPLIANT"

    return {
        "overall": overall,
        "items": item_statuses,
        "confirmed_compliant_items": [k for k, v in item_statuses.items() if v["status"] == "COMPLIANT"],
        "unknown_items": unknowns,
    }


# STEP 3: Intent routing — does this question even need the detector,
# and if so, is it asking about one SPECIFIC PPE item or the scene in general?

COUNT_WORDS = ["how many", "count", "number of"]
PRESENCE_WORDS = ["is there", "is anyone", "are there", "does anyone", "any worker"]
ABSENCE_WORDS = ["without", "not wearing", "missing", "no helmet", "no vest", "not compliant"]
COMPARISON_WORDS = ["most common", "which is more", "compare"]
SAFETY_WORDS = ["safe", "compliant", "compliance", "violation", "risk", "worker", "ppe",
                "helmet", "hardhat", "vest", "mask", "person", "people",
                "object", "objects", "detected", "item", "items"]

# Things our detector fundamentally cannot know - flag these before even calling it.
UNSUPPORTED_ATTRIBUTES = ["color", "colour", "age", "gender", "name", "brand", "emotion", "mood"]

# Maps a PPE item name (matching REQUIRED_PPE keys) to the words a question
# might use to refer to it. Lets the reasoning layer answer about ONE specific
# item instead of always dumping a report on every tracked item.
ITEM_KEYWORDS = {
    "helmet": ["helmet", "hardhat", "hard hat"],
    "vest": ["vest", "safety vest"],
    "mask": ["mask"],
}


def extract_target_item(question: str):
    """
    Checks if the question names a specific PPE item (helmet/vest/mask).
    Returns the item name if found, else None (meaning: question is about
    the scene/workers in general, not one specific item).
    """
    q = question.lower()
    for item_name, keywords in ITEM_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return item_name
    return None


def classify_intent(question: str):
    q = question.lower()

    needs_attribute_we_dont_have = any(word in q for word in UNSUPPORTED_ATTRIBUTES)
    mentions_image_content = any(word in q for word in SAFETY_WORDS)

    if needs_attribute_we_dont_have:
        return {"needs_detector": False, "reason": "unsupported_attribute", "target_item": None}

    if not mentions_image_content:
        return {"needs_detector": False, "reason": "unrelated_to_image", "target_item": None}

    target_item = extract_target_item(question)

    if any(w in q for w in COUNT_WORDS):
        qtype = "count"
    elif any(w in q for w in ABSENCE_WORDS):
        qtype = "absence"
    elif any(w in q for w in PRESENCE_WORDS):
        qtype = "presence"
    elif any(w in q for w in COMPARISON_WORDS):
        qtype = "comparison"
    else:
        qtype = "general_safety"

    return {"needs_detector": True, "reason": qtype, "target_item": target_item}


# STEP 4: Build the final answer

def build_answer(qtype, workers_status, detections, target_item=None):
    total_workers = len(workers_status)
    compliant = sum(1 for w in workers_status if w["overall"] == "COMPLIANT")
    at_risk = sum(1 for w in workers_status if w["overall"] == "AT_RISK")
    unsure = sum(1 for w in workers_status if w["overall"] in ("INSUFFICIENT_INFO", "PARTIALLY_CONFIRMED"))

    if total_workers == 0:
        return "No workers were detected in this image with sufficient confidence."

    if qtype == "count":
        if target_item:
            item_compliant = sum(1 for w in workers_status if w["items"].get(target_item, {}).get("status") == "COMPLIANT")
            item_violation = sum(1 for w in workers_status if w["items"].get(target_item, {}).get("status") == "VIOLATION")
            item_unknown = sum(1 for w in workers_status if w["items"].get(target_item, {}).get("status") == "INSUFFICIENT_INFO")
            return f"{item_compliant} of {total_workers} worker(s) confirmed wearing a {target_item}. {item_violation} confirmed missing one. {item_unknown} uncertain."
        return f"{total_workers} worker(s) detected. {compliant} compliant, {at_risk} at risk, {unsure} with insufficient information to confirm."

    if qtype in ("presence", "absence", "general_safety"):
        if target_item:
            violated = sum(1 for w in workers_status if w["items"].get(target_item, {}).get("status") == "VIOLATION")
            unknown = sum(1 for w in workers_status if w["items"].get(target_item, {}).get("status") == "INSUFFICIENT_INFO")
            confirmed = total_workers - violated - unknown

            if violated > 0:
                return f"{violated} of {total_workers} worker(s) are missing a {target_item}. {confirmed} confirmed compliant, {unknown} uncertain."
            elif unknown == total_workers:
                return f"Insufficient information to confirm {target_item} status for any worker in this image."
            else:
                return f"No confirmed {target_item} violations. {confirmed} worker(s) confirmed compliant, {unknown} uncertain."

        if at_risk > 0:
            violated_items = set()
            for w in workers_status:
                for item, info in w["items"].items():
                    if info["status"] == "VIOLATION":
                        violated_items.add(item)
            return (f"{at_risk} of {total_workers} worker(s) have a detected violation "
                    f"(missing: {', '.join(sorted(violated_items))}). "
                    f"{unsure} worker(s) have insufficient information to confirm compliance.")
        elif unsure > 0 and compliant == 0:
            return f"Insufficient information to confirm PPE compliance for {unsure} of {total_workers} worker(s). No confirmed violations."
        else:
            return f"All {total_workers} worker(s) appear compliant based on detected PPE, with {unsure} uncertain case(s)."

    if qtype == "comparison":
        class_counts = {}
        for d in detections:
            class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1
        if not class_counts:
            return "No objects were detected to compare."
        most_common = max(class_counts, key=class_counts.get)
        return f"The most common detected item is '{most_common}' ({class_counts[most_common]} instance(s))."

    return f"{total_workers} worker(s) detected: {compliant} compliant, {at_risk} at risk, {unsure} uncertain."


# Entry point called by the /ask endpoint

def answer_question(question: str, image_bytes: bytes):
    intent = classify_intent(question)

    if not intent["needs_detector"]:
        if intent["reason"] == "unsupported_attribute":
            answer = "Insufficient information — this detector does not track that attribute (e.g. color, age, identity)."
        else:
            answer = "This question does not relate to detectable image content, so no detection was performed."
        return {
            "question": question,
            "answer": answer,
            "detector_called": False,
            "reasoning_trace": {"intent": intent},
        }

    detections = run_detection(image_bytes)
    workers = match_workers_to_ppe(detections)
    workers_status = [compute_worker_status(w) for w in workers]

    answer = build_answer(intent["reason"], workers_status, detections, target_item=intent.get("target_item"))

    return {
        "question": question,
        "answer": answer,
        "detector_called": True,
        "reasoning_trace": {
            "intent": intent,
            "total_detections": len(detections),
            "workers": [
                {
                    "person_confidence": w["person_confidence"],
                    "status": s["overall"],
                    "items": s["items"],
                }
                for w, s in zip(workers, workers_status)
            ],
        },
    }