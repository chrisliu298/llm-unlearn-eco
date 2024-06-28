from datasets import DatasetDict, concatenate_datasets
from tqdm import tqdm


def chunk_text(text_lines, max_length, tokenizer):
    tokenized_lengths = tokenizer(
        text_lines, add_special_tokens=False, return_length=True
    )["length"]

    chunks = []  # List to hold the final chunks of text
    current_chunk = []  # Initialize the current chunk as a list of lines
    current_length = 0  # Initialize the current length of the chunk in tokens
    total_tokens = 0  # Initialize the total number of tokens

    for line, length in tqdm(
        zip(text_lines, tokenized_lengths), total=len(text_lines), desc="Chunking text"
    ):
        # -2 for the bos and eos tokens
        if current_length + length <= max_length - 2:
            current_chunk.append(line)
            current_length += length
        else:
            # Join the current_chunk into a single string and add it to chunks
            if current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = [line]  # Start a new chunk with the current line
                current_length = (
                    length  # Reset the current length to the length of the current line
                )
            else:
                # Directly add the line as a chunk if it's too long and current_chunk is empty
                chunks.append(line)
                current_length = 0  # Reset the current length
        total_tokens += length  # Update the total token count

    # Don't forget to add the last chunk if it exists
    if current_chunk:
        chunks.append("".join(current_chunk))

    print(f"Total tokens: {total_tokens}")

    # Remove all empty chunks
    chunks = [chunk for chunk in chunks if len(chunk) > 1]
    return chunks


mmlu_subjects = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def merge_datasets(datasets):
    merged_splits = {}
    all_splits = set(split for dataset in datasets for split in dataset.keys())
    for split in all_splits:
        split_datasets = [dataset[split] for dataset in datasets if split in dataset]
        if split_datasets:
            merged_splits[split] = concatenate_datasets(split_datasets)
    return DatasetDict(merged_splits)
