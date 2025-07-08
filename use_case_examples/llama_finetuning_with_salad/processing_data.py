# Prepare a question-answer data-set for language model fine-tuning.
# The data-set is hold by the client
# The dev side also knows some information about the data-set
# (for compilation purposes)

import argparse

from datasets import load_dataset, load_from_disk
from utils_dev import *

raw_dataset = load_dataset(DATASET_NAME, split="train")


def length_filter(example):
    q_len = len(TOKENIZER(example["question"], add_special_tokens=False)["input_ids"])
    a_len = len(TOKENIZER(example["answer"], add_special_tokens=False)["input_ids"])
    return (q_len + a_len + 1) <= MAX_LENGTH


def get_lengths(example):
    q_len = len(TOKENIZER(example["question"], add_special_tokens=False)["input_ids"])
    a_len = len(TOKENIZER(example["answer"], add_special_tokens=False)["input_ids"])
    total_len = q_len + a_len + 1
    return {"q_len": q_len, "a_len": a_len, "total_len": total_len}


def show_stats(raw_dataset, filtered_dataset, lengths):
    q_lengths = [x["q_len"] for x in lengths]
    a_lengths = [x["a_len"] for x in lengths]
    total_lengths = [x["total_len"] for x in lengths]

    print("\nLength Distribution Statistics:")
    print(f"Original dataset size: {len(raw_dataset):,}")
    print(f"Filtered dataset size: {len(filtered_dataset):,}")
    print(f"Percentage kept: {100 * len(filtered_dataset)/len(raw_dataset):.1f}%\n")
    print("Question lengths: ")
    print(f"  Min: {min(q_lengths)}, Max: {max(q_lengths)}")
    print(f"  Mean: {sum(q_lengths)/len(q_lengths):.1f}")
    print(f"  Median: {sorted(q_lengths)[len(q_lengths)//2]}")
    print("\nAnswer lengths:")
    print(f"  Min: {min(a_lengths)}, Max: {max(a_lengths)}")
    print(f"  Mean: {sum(a_lengths)/len(a_lengths):.1f}")
    print(f"  Median: {sorted(a_lengths)[len(a_lengths)//2]}")
    print("\nTotal lengths (including newline):")
    print(f"  Min: {min(total_lengths)}, Max: {max(total_lengths)}")
    print(f"  Mean: {sum(total_lengths)/len(total_lengths):.1f}")
    print(f"  Median: {sorted(total_lengths)[len(total_lengths)//2]}\n")


def process_example(example):
    """Tokenize a question-answer pair and prepare labels for training.

    Args:
        example (dict): Dictionary with 'question' and 'answer' strings
    Returns:
        dict: Processed tokens with masked labels for the question portion
    """
    question = example["question"].strip()
    answer = example["answer"].strip()
    tokens = TOKENIZER(
        question + "\n" + answer,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    question_length = len(TOKENIZER(question, add_special_tokens=False)["input_ids"]) + 1
    labels = tokens["input_ids"].copy()
    for i in range(question_length):
        if i < len(labels):
            labels[i] = -100
    tokens["labels"] = labels
    return tokens


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", default=True, help="Print dataset statistics")
    parser.add_argument("--reset", default=False, help="Recreate the train and the test dataset")
    args = parser.parse_args()

    DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)

    if args.reset or not (TRAIN_PATH.exists() and TEST_PATH.exists()):

        filtered_dataset = raw_dataset.filter(length_filter)
        lengths = filtered_dataset.map(get_lengths)

        if args.verbose:
            show_stats(raw_dataset, filtered_dataset, lengths)

        tokenized_dataset = filtered_dataset.map(
            process_example,
            batched=False,
            remove_columns=filtered_dataset.column_names,
        )

        tokenized = tokenized_dataset.train_test_split(test_size=0.05, seed=SEED, shuffle=True)
        train_dataset, test_dataset = tokenized["train"], tokenized["test"]

        if args.verbose:
            print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
            print(f"Train samples: {type(train_dataset)}, Test samples: {type(test_dataset)}")

        # Save
        train_dataset.save_to_disk(TRAIN_PATH)
        test_dataset.save_to_disk(TEST_PATH)

    else:
        print(f"Datasets already processed and saved in '{DATA_DIR_PATH.absolute()}'")
