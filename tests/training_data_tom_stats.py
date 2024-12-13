"""
Script for exploring the matches of tokens
corresponding to the ToM data we want to remove / replace 
in the pre-tokenized OLMO training data.
"""
import numpy as np
from cached_path import cached_path

from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
import transformers
import argparse
from time import time

tokenizer = transformers.AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")
tom_vocab = [" think", " thinks", " believe", " believes", " know", " knows", " thought", " knew", " believed"]
tom_tokens = [
    tokenizer.encode(tom_token, add_special_tokens=False)
    for tom_token in tom_vocab
]

sequence_counts = {tuple(seq): 0 for seq in tom_tokens}

# Convert sequences to tuples for easier comparison
tom_sequences = [tuple(seq) for seq in tom_tokens]

# Update these paths to what you want:
data_order_file_path = cached_path("https://olmo-checkpoints.org/ai2-llm/olmo-small/46zc5fly/train_data/global_indices.npy")
train_config_path = "configs/official/OLMo-1B.yaml"


cfg = TrainConfig.load(train_config_path)
dataset = build_memmap_dataset(cfg, cfg.data)
batch_size = cfg.global_train_batch_size
global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)

def count_in_line(line, sequence):
    """
    Function checking whether particular tokens are present in the training data.
    """
    seq_len = len(sequence)
    count = 0
    for i in range(len(line) - seq_len + 1):
        if tuple(line[i:i + seq_len]) == sequence:
            count += 1
    return count

def get_batch_instances(batch_idx: int) -> list[list[int]]:
    batch_start = batch_idx * batch_size
    batch_end = (batch_idx + 1) * batch_size
    batch_indices = global_indices[batch_start:batch_end]
    batch_instances = []
    for index in batch_indices:
        token_ids = dataset[index]["input_ids"].tolist()
        # check occurrences of ToM tokens
        for sequence in tom_sequences:
            sequence_counts[sequence] += count_in_line(token_ids, sequence)
        batch_instances.append(token_ids)
    return batch_instances


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--num_batches", type=int, default=100)

    args = args_parser.parse_args()
    print(" ---- initial ToM token counts: ----- \n", sequence_counts)
    # time the function
    start = time()
    overall_tokens = 0
    # Get all 2048 x 2048 token IDs in the first 100 batches.
    for i in range(args.num_batches):
        current_batch = get_batch_instances(i)
        # add the number of tokens in the batch to the overall count
        overall_tokens += len(current_batch) * len(current_batch[0])
    # print resulting ToM token counts
    print("----- final ToM token counts: ------ \n", sequence_counts)
    print("\n* total tokens:", overall_tokens)
    print("\n* proportion of ToM tokens in the data:", sum(sequence_counts.values()) / overall_tokens)
    print("\n* time taken:", time() - start)