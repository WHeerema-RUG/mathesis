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
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

# Set to CPU; efficient model is not necessary
device = torch.device("cpu")


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


def build_transformer(vocab_size, embed_dim=128, num_heads=4, num_layers=2,
                      ff_dim=256, dropout=0.1):
    """Creates a basic transformer and send it to the device"""
    # Create embeddings
    embedding = Embedding(vocab_size, embed_dim, padding_idx=0).to(device)
    pos_embedding = Parameter(torch.randn(1, 512, embed_dim)).to(device)
    # Create transformer itself
    transformer = TransformerEncoder(
        TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout),
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
                pos_embedding, transformer, fc_out):
    """Trains the transformer for a single epoch, in order to calculate loss
    """
    # Initialize loss
    total_loss = 0
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
        total_loss += loss.item()
    # Return that epoch's loss
    return total_loss / len(dataloader)


def transformer_ops(tokenized, vocab, epochs, verbose=True):
    """Perform every operation for creating and evaluating a transformer"""
    # Preprocess data to get mask and padded dataset
    pad_token = 0
    data, attn_masks = prepare_data(tokenized, pad_token)
    dataset = TensorDataset(data, attn_masks)
    dataloader = DataLoader(dataset, batch_size=16)
    # Get every component of the transformer
    embedding, pos_embedding, transformer, fc_out = build_transformer(vocab)
    optimizer = Adam(list(embedding.parameters()) +
                     list(transformer.parameters()) +
                     list(fc_out.parameters()), lr=0.001)
    # Set cross-entropy as the loss criterion
    criterion = CrossEntropyLoss(ignore_index=pad_token)
    # Iterate to get loss; print as both loss and perplexity
    # Only print every epoch if verbose; else, only print last epoch result
    if verbose:
        # The 25 * "=" is just to prettify the output
        print(25 * "=")
    for epoch in range(epochs):
        train_loss = train_epoch(dataloader, optimizer, criterion, embedding,
                                 pos_embedding, transformer, fc_out)
        perplexity = np.exp(train_loss)
        if verbose:
            print("EPOCH", epoch, "\nTrain Loss:", train_loss,
                  "\nPerplexity:", perplexity, "\n" + 25 * "=")
    if not verbose:
        print("\nTrain Loss:", train_loss,
              "\nPerplexity:", perplexity)
    return train_loss, perplexity
