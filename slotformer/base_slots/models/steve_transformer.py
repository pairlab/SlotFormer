"""Borrowed from https://github.com/singhgautam/slate/blob/master/transformer.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange

from .steve_utils import linear


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0., gain=1.):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads

        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        self.proj_q = linear(d_model, d_model, bias=False)
        self.proj_k = linear(d_model, d_model, bias=False)
        self.proj_v = linear(d_model, d_model, bias=False)
        self.proj_o = linear(d_model, d_model, bias=False, gain=gain)

    def forward(self, q, k, v, attn_mask=None):
        """
        q: batch_size x target_len x d_model
        k: batch_size x source_len x d_model
        v: batch_size x source_len x d_model
        attn_mask: target_len x source_len
        return: batch_size x target_len x d_model
        """
        B, T, _ = q.shape
        _, S, _ = k.shape

        q = self.proj_q(q).view(B, T, self.num_heads, -1).transpose(1, 2)
        k = self.proj_k(k).view(B, S, self.num_heads, -1).transpose(1, 2)
        v = self.proj_v(v).view(B, S, self.num_heads, -1).transpose(1, 2)

        q = q * (q.shape[-1]**(-0.5))
        attn = torch.matmul(q, k.transpose(-1, -2))

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        output = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        output = self.proj_o(output)
        output = self.output_dropout(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, max_len, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(
            torch.zeros(1, max_len, d_model), requires_grad=True)
        nn.init.trunc_normal_(self.pe)

    def forward(self, input):
        """
        input: batch_size x seq_len x d_model
        return: batch_size x seq_len x d_model
        """
        T = input.shape[1]
        return self.dropout(input + self.pe[:, :T])


class TransformerEncoderBlock(nn.Module):

    def __init__(self,
                 d_model,
                 num_heads,
                 dropout=0.,
                 gain=1.,
                 is_first=False):
        super().__init__()

        self.is_first = is_first

        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout, gain)

        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            linear(d_model, 4 * d_model, weight_init='kaiming'), nn.ReLU(),
            linear(4 * d_model, d_model, gain=gain), nn.Dropout(dropout))

    def forward(self, input):
        """
        input: batch_size x source_len x d_model
        return: batch_size x source_len x d_model
        """
        if self.is_first:
            input = self.attn_layer_norm(input)
            x = self.attn(input, input, input)
            input = input + x
        else:
            x = self.attn_layer_norm(input)
            x = self.attn(x, x, x)
            input = input + x

        x = self.ffn_layer_norm(input)
        x = self.ffn(x)
        return input + x


class TransformerEncoder(nn.Module):

    def __init__(self, num_blocks, d_model, num_heads, dropout=0.):
        super().__init__()

        if num_blocks > 0:
            gain = (2 * num_blocks)**(-0.5)
            self.blocks = nn.ModuleList([
                TransformerEncoderBlock(
                    d_model, num_heads, dropout, gain, is_first=True)
            ] + [
                TransformerEncoderBlock(
                    d_model, num_heads, dropout, gain, is_first=False)
                for _ in range(num_blocks - 1)
            ])
        else:
            self.blocks = nn.ModuleList()

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        """
        input: batch_size x source_len x d_model
        return: batch_size x source_len x d_model
        """
        for block in self.blocks:
            input = block(input)

        return self.layer_norm(input)


class TransformerDecoderBlock(nn.Module):

    def __init__(
        self,
        max_len,
        d_model,
        num_heads,
        dropout=0.,
        gain=1.,
        is_first=False,
    ):
        super().__init__()

        self.is_first = is_first

        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, gain)

        mask = torch.triu(
            torch.ones((max_len, max_len), dtype=torch.bool), diagonal=1)
        self.self_attn_mask = nn.Parameter(mask, requires_grad=False)

        self.encoder_decoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_decoder_attn = MultiHeadAttention(d_model, num_heads,
                                                       dropout, gain)

        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            linear(d_model, 4 * d_model, weight_init='kaiming'), nn.ReLU(),
            linear(4 * d_model, d_model, gain=gain), nn.Dropout(dropout))

    def forward(self, input, encoder_output):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        T = input.shape[1]

        if self.is_first:
            input = self.self_attn_layer_norm(input)
            x = self.self_attn(input, input, input,
                               self.self_attn_mask[:T, :T])
            input = input + x
        else:
            x = self.self_attn_layer_norm(input)
            x = self.self_attn(x, x, x, self.self_attn_mask[:T, :T])
            input = input + x

        x = self.encoder_decoder_attn_layer_norm(input)
        x = self.encoder_decoder_attn(x, encoder_output, encoder_output)
        input = input + x

        x = self.ffn_layer_norm(input)
        x = self.ffn(x)
        return input + x


class TransformerDecoder(nn.Module):

    def __init__(
        self,
        num_blocks,
        max_len,
        d_model,
        num_heads,
        dropout=0.,
    ):
        super().__init__()

        block = TransformerDecoderBlock
        if num_blocks > 0:
            gain = (3 * num_blocks)**(-0.5)
            self.blocks = nn.ModuleList([
                block(
                    max_len, d_model, num_heads, dropout, gain, is_first=True)
            ] + [
                block(
                    max_len, d_model, num_heads, dropout, gain, is_first=False)
                for _ in range(num_blocks - 1)
            ])
        else:
            self.blocks = nn.ModuleList()

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input, encoder_output):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model, *slots*
        return: batch_size x target_len x d_model
        """
        for block in self.blocks:
            input = block(input, encoder_output)

        return self.layer_norm(input)


class STEVETransformerDecoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        n_head,
        max_len,
        num_slots,
        num_layers,
        dropout=0.1,
    ):
        super().__init__()

        self.max_len = max_len
        self.vocab_size = vocab_size
        self.num_slots = num_slots

        # input embedding stem
        # we use (max_len + 1) because of the BOS token
        self.in_proj = nn.Linear(d_model, d_model)
        self.tok_emb = nn.Embedding(vocab_size + 1, d_model)
        self.pos_emb = PositionalEncoding(max_len + 1, d_model, dropout)

        # TransformerDecoder
        self.tf_dec = TransformerDecoder(
            num_blocks=num_layers,
            max_len=max_len + 1,
            d_model=d_model,
            num_heads=n_head,
            dropout=dropout,
        )

        # final idx prediction
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, slots, idx):
        """Forward pass.

        Args:
            slots (torch.FloatTensor): (B, t1, C)
            idx (torch.LongTensor): input token indices of shape (B, t2)
                this already excludes the last GT token
        """
        assert slots.shape[1] == self.num_slots
        B, T = idx.shape
        assert T <= self.max_len

        # forward the TransformerDecoder model
        slots = self.in_proj(slots)  # [B, t1, C]

        # add [BOS] token
        BOS = torch.ones((B, 1)).type_as(idx) * self.vocab_size
        idx = torch.cat([BOS, idx], dim=1)  # [B, 1 + t2]
        token_embeddings = self.tok_emb(idx)  # [B, 1 + t2, C]
        tokens = self.pos_emb(token_embeddings)  # [B, 1 + t2, C]

        x = self.tf_dec(tokens, slots)  # [B, 1 + t2, C]

        logits = self.head(x)  # [B, 1 + t2, vocab_size]

        return logits

    def generate(self, slots, steps, sample=False, temperature=1.0):
        """Generate `steps` tokens conditioned on slots."""
        assert not self.training
        B = slots.shape[0]
        assert steps - 1 <= self.max_len
        idx_cond = torch.zeros((B, 0)).long().to(slots.device)
        all_logits = []
        for _ in range(steps):
            # predict one step
            logits = self.forward(slots, idx_cond)
            logits = logits[:, -1]  # [B, vocab_size]
            # we have to do .cpu() to handle large number of tokens
            # luckily `generate` is only used for evaluation
            all_logits.append(logits.cpu())
            # apply softmax to convert to probabilities
            probs = F.softmax(logits / temperature, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            idx_cond = torch.cat((idx_cond, ix), dim=1)
            torch.cuda.empty_cache()
        all_logits = torch.stack(all_logits, dim=1)  # [B, steps, vocab_size]
        return idx_cond, all_logits
