import argparse
import copy
import os
import random

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


def shuffle_context(split, seed):
    column_names = list(split.column_names)
    cols = {k: split[k] for k in column_names}
    n = len(split)

    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)

    rows = []
    for i in range(n):
        old_ctx = cols["documents_str"][i]
        new_ctx = cols["documents_str"][perm[i]]

        row = {k: cols[k][i] for k in column_names}
        row["documents"] = cols["documents"][perm[i]]
        row["documents_str"] = new_ctx

        prompt = copy.deepcopy(cols["rag_prompt"][i])
        replaced = False
        for msg in prompt:
            if old_ctx and old_ctx in msg["content"]:
                msg["content"] = msg["content"].replace(old_ctx, new_ctx)
                replaced = True
        if not replaced and old_ctx != new_ctx:
            raise ValueError(f"Could not locate context string in rag_prompt at row {i}")
        row["rag_prompt"] = prompt

        rows.append(row)

    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle the context column of a RAG dataset across samples, "
        "breaking the context-response alignment while keeping everything else identical."
    )
    parser.add_argument("dataset", help="HF dataset id (e.g. F4biian/RAGognize) or local path")
    parser.add_argument("outdir", help="Output directory for the shuffled dataset")
    parser.add_argument("--seed", type=int, default=432)
    parser.add_argument("--splits", nargs="+", default=["train"], help="Splits to shuffle (default: train)")
    args = parser.parse_args()

    if os.path.exists(args.dataset):
        dataset = load_from_disk(args.dataset)
    else:
        dataset = load_dataset(args.dataset)

    shuffled = {}
    for split_name in dataset:
        if split_name in args.splits:
            shuffled[split_name] = shuffle_context(dataset[split_name], args.seed)
        else:
            shuffled[split_name] = dataset[split_name]

    result = DatasetDict(shuffled)
    result.save_to_disk(args.outdir)
    print(f"Saved shuffled dataset to {args.outdir}")
    for name in result:
        print(f"  {name}: {len(result[name])} rows")


if __name__ == "__main__":
    main()
