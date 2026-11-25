"""Prepare APHD's fixed training, validation, and test splits from the downloaded datasets."""

import argparse
import hashlib
import random
from filecmp import cmp
from pathlib import Path
from tempfile import TemporaryDirectory

# Fingerprints cover train, test, and social files, ignoring platform line endings.
SOURCE_DIGESTS = {
    "douban-book": "523d3b2eb8b417f0b60e0a0c4884aa7da00bccdb9d353a09b4f62ca79f7c1c6e",
    "douban-movie": "8ebfbe11173c2a6a4a9e2fdb926cf7d0b71f4452c9c105c7a36b51d52ed3b6cb",
    "yelp": "e9ed739a3013054aca113e5442a6425c5e1ecd45613a0fa0921508717c0ca488",
}


def validation_edges(name, rows):
    rng = random.Random(3407)
    remaining = {user: len(items) for user, items in rows}
    target = round(sum(remaining.values()) / 8)
    if name == "douban-book":
        # Sample globally, retaining at least three training items per user when available.
        candidates = [(user, item) for user, items in rows for item in items]
        rng.shuffle(candidates)
        held_out = set()
        for user, item in candidates:
            if remaining[user] > 3:
                held_out.add((user, item))
                remaining[user] -= 1
                if len(held_out) == target:
                    break
        return held_out

    # Movie and Yelp reserve three training items per user before the global draw.
    candidates = []
    for user, items in rows:
        if len(items) > 3:
            shuffled = items.copy()
            rng.shuffle(shuffled)
            candidates.extend((user, item) for item in shuffled[:-3])
    rng.shuffle(candidates)
    return set(candidates[:target])


def prepare_dataset(name, source, destination):
    files = {}
    fingerprint = hashlib.sha256()
    for stem in ("train", "test", "social"):
        content = (source / f"{stem}.txt").read_bytes().replace(b"\r\n", b"\n")
        fingerprint.update(hashlib.sha256(content).digest())
        files[stem] = content
    if fingerprint.hexdigest() != SOURCE_DIGESTS[name]:
        raise ValueError(f"{name}: source files differ from the dataset version linked in README.md.")

    rows = []
    for line in files["train"].splitlines():
        user, *items = line.split()
        rows.append((user, items))
    held_out = validation_edges(name, rows)
    destination.mkdir()
    # Keep the released split's row order, empty validation rows, and line endings.
    keep_empty = name == "douban-book"
    newline = b"\n" if keep_empty else b"\r\n"
    with (destination / "train.txt").open("wb") as train, (destination / "val.txt").open("wb") as val:
        for user, items in rows:
            train_items, val_items = [user], [user]
            for item in items:
                if (user, item) in held_out:
                    val_items.append(item)
                else:
                    train_items.append(item)
            train.write(b" ".join(train_items) + newline)
            if keep_empty or len(val_items) > 1:
                val.write(b" ".join(val_items) + newline)
    for stem in ("test", "social"):
        (destination / f"{stem}.txt").write_bytes(files[stem].replace(b"\n", b"\r\n"))


def prepare(source, output):
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Choose separate, non-overlapping source and output directories.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        staged = Path(tmp) / "prepared"
        staged.mkdir()
        for name in SOURCE_DIGESTS:
            prepare_dataset(name, source / name, staged / name)
        if output.exists():
            for file in staged.glob("*/*.txt"):
                existing = output / file.relative_to(staged)
                if not existing.is_file() or not cmp(file, existing, shallow=False):
                    raise ValueError(f"{output} already exists with different contents. Choose a new output directory.")
        else:
            staged.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="directory containing the douban-book, douban-movie, and yelp folders")
    parser.add_argument("--output", type=Path, default=Path("Data"), help="prepared dataset directory (default: ./Data)")
    args = parser.parse_args()
    try:
        prepare(args.source.resolve(), args.output.resolve())
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"APHD datasets ready in {args.output.resolve()}")
