import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
        Multi-head Attention mechanism used in Transformer models.

        Args:
            model_size (int): The dimensionality of input and output vectors.
            heads_num (int): Number of attention heads.

        Attributes:
            model_size (int): The dimensionality of input and output vectors.
            heads_num (int): Number of attention heads.
            self.head_dim (int): Dimensionality of each attention head.
            self.values (nn.Linear): Linear layer to transform values.
            self.keys (nn.Linear): Linear layer to transform keys.
            self.queries (nn.Linear): Linear layer to transform queries.
            self.fc_out (nn.Linear): Linear layer for final output transformation.
    """
    def __init__(self, model_size, heads_num):
        super(MultiHeadAttention, self).__init__()
        assert (
                model_size % heads_num == 0
        ), "Model size needs to be divisible by num of heads"

        self.model_size = model_size
        self.heads_num = heads_num
        self.head_dim = model_size // heads_num

        self.values = nn.Linear(self.model_size, self.model_size)
        self.keys = nn.Linear(self.model_size, self.model_size)
        self.queries = nn.Linear(self.model_size, self.model_size)
        self.fc_out = nn.Linear(self.model_size, self.model_size)  # (fully connected)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
            Scaled dot-product attention mechanism.

            Args:
                q (torch.Tensor): Queries tensor of shape (batch_size, seq_length_q, model_size).
                k (torch.Tensor): Keys tensor of shape (batch_size, seq_length_k, model_size).
                v (torch.Tensor): Values tensor of shape (batch_size, seq_length_v, model_size).
                mask (torch.Tensor, optional): Mask tensor indicating which elements should be masked. Defaults to None.

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, seq_length_q, model_size).
        """

        attn = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.FloatTensor([self.head_dim]))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        return output

    def split_heads(self, x):
        """
            Split input tensor into multiple heads for parallel computation in multi-head attention.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_length, model_size).

            Returns:
                torch.Tensor: Tensor split into multiple heads of shape (batch_size, num_heads, seq_length, head_dim).
        """
        batch_size, seq_length, model_size = x.size()
        return x.view(batch_size, seq_length, self.heads_num, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        """
            Combine the multiple heads back to the original shape.

            Args:
                x (torch.Tensor): Tensor of shape (batch_size, num_heads, seq_length, head_dim).

            Returns:
                torch.Tensor: Combined tensor of shape (batch_size, seq_length, model_size).
        """
        batch_size, _, seq_length, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.model_size)

    def forward(self, queries, keys, values, mask):
        """
            Forward pass through the MultiHeadAttention layer.

            Args:
                queries (torch.Tensor): Tensor of queries with shape (batch_size, seq_length_q, model_size).
                keys (torch.Tensor): Tensor of keys with shape (batch_size, seq_length_k, model_size).
                values (torch.Tensor): Tensor of values with shape (batch_size, seq_length_v, model_size).
                mask (torch.Tensor): Mask tensor indicating which elements should be masked.

            Returns:
                torch.Tensor: Output tensor after multi-head attention with shape (batch_size, seq_length_q, model_size).
        """
        queries = self.split_heads(self.queries(queries))
        keys = self.split_heads(self.keys(keys))
        values = self.split_heads(self.values(values))

        attention = self.scaled_dot_product_attention(queries, keys, values, mask)
        output = self.fc_out(self.combine_heads(attention))
        return output


class PositionWiseFeedForward(nn.Module):
    """
        Position-wise feed-forward layer used in Transformer models.

        Args:
            model_size (int): The dimensionality of input and output vectors.
            ff_size (int): The size of the intermediate layer.

        Attributes:
            fc1 (nn.Linear): First fully connected layer.
            fc2 (nn.Linear): Second fully connected layer.
            relu (nn.ReLU): ReLU activation function.
    """
    def __init__(self, model_size, ff_size):  # (feed-forward)
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(model_size, ff_size)
        self.fc2 = nn.Linear(ff_size, model_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
            Forward pass through the PositionWiseFeedForward layer.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_length, model_size).

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, seq_length, model_size).
        """
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    """
        Positional encoding layer used in Transformer models to inject positional information into input embeddings.

        Args:
            model_size (int): The dimensionality of input vectors.
            max_seq_length (int): The maximum sequence length for positional encoding.

        Attributes:
            pe (torch.Tensor): Positional encoding tensor of shape (1, max_seq_length, model_size).
    """
    def __init__(self, model_size, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, model_size)  # (positional encodings)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_size, 2).float() * -(torch.log(torch.FloatTensor([10000.0])) / model_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
            Forward pass through the PositionalEncoding layer.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_length, model_size).

            Returns:
                torch.Tensor: Output tensor with positional encodings added, of shape (batch_size, seq_length, model_size).
        """
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    """
        Encoder layer of the Transformer model.

        Args:
            model_size (int): The dimensionality of input and output vectors.
            heads_num (int): Number of attention heads.
            ff_size (int): The size of the intermediate layer in the feed-forward network.
            dropout (any): Dropout probability.

        Attributes:
            self_attn (MultiHeadAttention): Multi-head self-attention mechanism.
            feed_forward (PositionWiseFeedForward): Position-wise feed-forward network.
            norm1 (nn.LayerNorm): Layer normalization after the first sub-layer.
            norm2 (nn.LayerNorm): Layer normalization after the second sub-layer.
            dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, model_size, heads_num, ff_size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(model_size, heads_num)
        self.feed_forward = PositionWiseFeedForward(model_size, ff_size)
        self.norm1 = nn.LayerNorm(model_size)
        self.norm2 = nn.LayerNorm(model_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
            Forward pass through the EncoderLayer.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_length, model_size).
                mask (torch.Tensor): Mask tensor indicating which elements should be masked.

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, seq_length, model_size).
        """
        attn = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn))
        ff = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff))
        return x


class DecoderLayer(nn.Module):
    """
        Decoder layer of the Transformer model.

        Args:
            model_size (int): The dimensionality of input and output vectors.
            heads_num (int): Number of attention heads.
            ff_size (int): The size of the intermediate layer in the feed-forward network.
            dropout (any): Dropout probability.

        Attributes:
            self_attn (MultiHeadAttention): Multi-head self-attention mechanism.
            cross_attn (MultiHeadAttention): Multi-head cross-attention mechanism.
            feed_forward (PositionWiseFeedForward): Position-wise feed-forward network.
            norm1 (nn.LayerNorm): Layer normalization after the first sub-layer.
            norm2 (nn.LayerNorm): Layer normalization after the second sub-layer.
            norm3 (nn.LayerNorm): Layer normalization after the third sub-layer.
            dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, model_size, heads_num, ff_size, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(model_size, heads_num)
        self.cross_attn = MultiHeadAttention(model_size, heads_num)
        self.feed_forward = PositionWiseFeedForward(model_size, ff_size)
        self.norm1 = nn.LayerNorm(model_size)
        self.norm2 = nn.LayerNorm(model_size)
        self.norm3 = nn.LayerNorm(model_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        """
            Forward pass through the DecoderLayer.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_length, model_size).
                enc_output (torch.Tensor): Output of the encoder stack of shape (batch_size, seq_length, model_size).
                src_mask (torch.Tensor): Mask tensor for source sequences indicating which elements should be masked.
                tgt_mask (torch.Tensor): Mask tensor for target sequences indicating which elements should be masked.

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, seq_length, model_size).
        """
        attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn))
        attn = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn))
        ff = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff))
        return x


class Transformer(nn.Module):
    """
        Transformer model composed of an encoder and a decoder.

        Args:
            src_vocab_size (int): Vocabulary size of the source language.
            tgt_vocab_size (int): Vocabulary size of the target language.
            model_size (int): The dimensionality of input and output vectors.
            heads_num (int): Number of attention heads.
            layers_num (int): Number of layers in both the encoder and the decoder.
            ff_size (int): The size of the intermediate layer in the feed-forward network.
            max_seq_length (int): Maximum sequence length for positional encoding.
            dropout (any): Dropout probability.

        Attributes:
            encoder_embedding (nn.Embedding): Embedding layer for the encoder input.
            decoder_embedding (nn.Embedding): Embedding layer for the decoder input.
            positional_encoding (PositionalEncoding): Positional encoding layer.
            encoder_layers (nn.ModuleList): List of encoder layers.
            decoder_layers (nn.ModuleList): List of decoder layers.
            fc (nn.Linear): Linear layer for final output transformation.
            dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, model_size, heads_num, layers_num, ff_size, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, model_size)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, model_size)
        self.positional_encoding = PositionalEncoding(model_size, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(model_size, heads_num, ff_size, dropout) for _ in range(layers_num)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(model_size, heads_num, ff_size, dropout) for _ in range(layers_num)])
        self.fc = nn.Linear(model_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    # noinspection PyUnresolvedReferences
    @staticmethod
    def generate_mask(src, tgt):
        """
            Generate masks for source and target sequences.

            Args:
                src (torch.Tensor): Source sequence tensor.
                tgt (torch.Tensor): Target sequence tensor.

            Returns:
                torch.Tensor, torch.Tensor: Source and target masks.
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        """
            Forward pass through the Transformer model.

            Args:
                src (torch.Tensor): Source sequence tensor.
                tgt (torch.Tensor): Target sequence tensor.

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, tgt_seq_length, tgt_vocab_size).
        """
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


if __name__ == '__main__':
    """ Sample data and transformer preparation """
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    model_size = 512
    heads_num = 8
    layers_num = 6
    ff_size = 2048
    max_seq_length = 100
    dropout = 0.1

    transformer = Transformer(src_vocab_size, tgt_vocab_size, model_size, heads_num, layers_num, ff_size, max_seq_length,
                              dropout)

    # Generate random sample data
    src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

    """ Training the model """
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()

    for epoch in range(100):
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1])
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    """ Transformer model performance evaluation """
    transformer.eval()

    # Generate random sample validation data
    val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

    with torch.no_grad():
        val_output = transformer(val_src_data, val_tgt_data[:, :-1])
        val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size),
                             val_tgt_data[:, 1:].contiguous().view(-1))
        print(f"Validation Loss: {val_loss.item()}")
