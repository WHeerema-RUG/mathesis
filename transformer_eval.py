# TRANSFORMER BUILDER AND EVALUATOR
# For the Master's thesis project
# Creates a basic PyTorch transformer for next word prediction,
# in order to test a language's ease of learning.
# Most of the code is written by GPT-4 Turbo, per my (Wessel's) requirements;
# the docstrings, comments and compliance with pycodestyle are still my work,
# as are some patches and idiosyncrasies.
# Date: 26/02/2025

import torch
from torch.nn import Embedding, Parameter, Linear, \
    TransformerEncoder, TransformerEncoderLayer, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from statistics import median
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data(tokenized, pad_token=0):
    """Pads tokenized sequences and creates attention masks"""
    # Get max length to pad to
    max_len = max(len(seq) for seq in tokenized)
    # Initialize variables
    padded_data = []
    attn_masks = []
    # Iterate to create mask
    for sent in tokenized:
        # Get necessary padding length for a sentence, then pad it
        pad_len = max_len - len(sent)
        padded_seq = sent + [pad_token] * pad_len
        # Create mask, only targeting real tokens
        mask = [1] * len(sent) + [0] * pad_len
        padded_data.append(padded_seq)
        attn_masks.append(mask)
    # Return as tensors
    return torch.tensor(padded_data), torch.tensor(attn_masks)


def build_transformer(vocab_size, embed_dim=128, num_heads=4, num_layers=3,
                      dropout=0.1):
    """Creates a basic transformer and sendS it to the device
    Parameters are a derivative of Vaswani et al. (2017), with 50% of head no.
    and layer count and 25% of embed dimension, as to handle a smaller dataset
    FF dimension calculated the same way as in Vaswani et al. (2017)
    """
    # Create embeddings
    embedding = Embedding(vocab_size, embed_dim, padding_idx=0).to(device)
    pos_embedding = Parameter(torch.randn(1, 512, embed_dim)).to(device)
    # Create transformer itself
    transformer = TransformerEncoder(
        TransformerEncoderLayer(embed_dim, num_heads, 4 * embed_dim, dropout),
        num_layers
    ).to(device)
    # Apply linear transformation for probabilities
    fc_out = Linear(embed_dim, vocab_size).to(device)
    # Return every layer separately
    return embedding, pos_embedding, transformer, fc_out


def forward(x, attn_mask, embedding, pos_embedding, transformer, fc_out):
    """Does NWP by encoding a sentence and predicting probabilities"""
    # Get sentence length
    sent_len = x.size(1)
    # Apply word and positional embeddings
    x = embedding(x) + pos_embedding[:, :sent_len, :]
    # Encode sentence with the transformer, minding the mask
    x = transformer(x.permute(1, 0, 2), src_key_padding_mask=(attn_mask == 0))
    # Return vocab probabilities
    return fc_out(x.permute(1, 0, 2))


def train_epoch(dataloader, optimizer, criterion, embedding,
                pos_embedding, transformer, fc_out, loss_out="avg"):
    """Trains the transformer for a single epoch, in order to calculate loss
    """
    # Initialize loss
    total_loss = []
    # Iterate over each batch
    for batch, attn_mask in tqdm(dataloader):
        # Send data to device
        batch, attn_mask = batch.to(device), attn_mask.to(device)
        # Let transformer work its magic
        optimizer.zero_grad()
        output = forward(batch[:, :-1], attn_mask[:, :-1], embedding,
                         pos_embedding, transformer, fc_out)
        # Calculate loss for batch and add to total epoch loss
        loss = criterion(output.reshape(-1, output.size(-1)),
                         batch[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    # Return that epoch's loss
    # Average
    if loss_out == "avg":
        return sum(total_loss) / len(total_loss)
    # Median
    elif loss_out == "median":
        return median(total_loss)
    # Invalid
    else:
        raise ValueError("Specify valid loss calculation")


def eval_epoch(dataloader, criterion, embedding, pos_embedding,
               transformer, fc_out, loss_out="avg"):
    """Makes predictions for the dataloader, returns loss
    Similar to train_epoch, but without optimizer or gradient calculation
    """
    # Initialize loss
    total_loss = []
    # Ensures no training is done
    with torch.no_grad():
        # Iterate over each batch
        for batch, attn_mask in dataloader:
            # Send data to device
            batch, attn_mask = batch.to(device), attn_mask.to(device)
            # Make prediction
            output = forward(batch[:, :-1], attn_mask[:, :-1], embedding,
                             pos_embedding, transformer, fc_out)
            # Calculate loss for batch and add to total epoch loss
            loss = criterion(output.reshape(-1, output.size(-1)),
                             batch[:, 1:].reshape(-1))
            total_loss.append(loss.item())
    # Return that epoch's loss
    # Average
    if loss_out == "avg":
        return sum(total_loss) / len(total_loss)
    # Median
    elif loss_out == "median":
        return median(total_loss)
    # Invalid
    else:
        raise ValueError("Specify valid loss calculation")


def transformer_ops(tokenized, vocab, epochs, verbose=True,
                    ratio=[0.8, 0.1, 0.1], return_vals="train",
                    loss_out="avg"):
    """Perform every operation for creating and evaluating a transformer"""
    # Sanity check
    if return_vals not in ["train", "val", "test"]:
        raise ValueError("""Specify one of ["train", "val", "test"]\
for returning""")
    # Preprocess data to get mask and padded dataset
    pad_token = 0
    data, attn_masks = prepare_data(tokenized, pad_token)
    dataset = TensorDataset(data, attn_masks)
    # Do the train/val/test split
    train_sub, val_sub, test_sub = random_split(dataset,
                                                [int(dec * len(dataset))
                                                 for dec in ratio])
    train_loader = DataLoader(train_sub, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_sub, batch_size=16, shuffle=True)
    # Get every component of the transformer
    embedding, pos_embedding, transformer, fc_out = build_transformer(vocab)
    optimizer = Adam(list(embedding.parameters()) +
                     list(transformer.parameters()) +
                     list(fc_out.parameters()), lr=0.001)
    # Set cross-entropy as the loss criterion
    criterion = CrossEntropyLoss(ignore_index=pad_token)
    # Iterate to get loss; print as both loss and perplexity
    # Initialize array of validation perplexity scores if specified as mixed
    if return_vals == "mixed":
        per_epoch = []
    # Only print every epoch if verbose; else, only print last epoch result
    if verbose:
        # The 25 * "=" is just to prettify the output
        print(25 * "=")
    for epoch in range(epochs):
        # Training
        train_loss = train_epoch(train_loader, optimizer, criterion,
                                 embedding, pos_embedding, transformer,
                                 fc_out, loss_out=loss_out)
        train_perplexity = np.exp(train_loss)
        # Validation
        val_loss = eval_epoch(val_loader, criterion, embedding, pos_embedding,
                              transformer, fc_out, loss_out=loss_out)
        val_perplexity = np.exp(val_loss)
        # Print out every value
        if verbose:
            print("EPOCH", epoch, "\nTrain Loss:", train_loss,
                  "\nTrain Perplexity:", train_perplexity,
                  "\nVal. Loss:", val_loss,
                  "\nVal. Perplexity:", val_perplexity,
                  "\n" + 25 * "=")
            if return_vals == "mixed":
                per_epoch.append((train_perplexity, val_perplexity))
    if not verbose:
        print("\nTrain Loss:", train_loss,
              "\nPerplexity:", train_perplexity,
              "\nVal. Loss:", val_loss,
              "\nVal. Perplexity:", val_perplexity)
    # Do test run
    test_loss = eval_epoch(test_loader, criterion, embedding, pos_embedding,
                           transformer, fc_out)
    test_perplexity = np.exp(test_loss)
    print("Test Loss:", test_loss, "\nTest Perplexity:", test_perplexity)
    # Return specified values
    if return_vals == "train":
        return train_loss, train_perplexity
    elif return_vals == "val":
        return val_loss, val_perplexity
    elif return_vals == "test":
        return test_loss, test_perplexity
    elif return_vals == "mixed":
        return per_epoch, test_perplexity
