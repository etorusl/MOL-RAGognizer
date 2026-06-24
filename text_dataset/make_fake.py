"""Shuffle image_name across entries to break image-text alignment.
Usage: python make_fake.py
Reads shroom-vision.train.en.labeled.jsonl, shuffles image_name, writes .fake.jsonl
"""
import json, random

SRC = "shroom-vision.train.en.labeled.jsonl"
DST = "shroom-vision.train.en.fake.jsonl"

with open(SRC) as f:
    entries = [json.loads(l) for l in f]

image_names = [e["image_name"] for e in entries]
random.shuffle(image_names)

for e, img in zip(entries, image_names):
    e["image_name"] = img

with open(DST, "w") as f:
    for e in entries:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"Written {len(entries)} entries to {DST} (images shuffled)")
