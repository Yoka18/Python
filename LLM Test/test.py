import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def simple_tokenizer(text):
    return text.split()

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Katıştırma boyutu başlık sayısına bölünebilmelidir."

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out


class TransformerDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.lines = file.read().splitlines()
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab = self.build_vocab()
        
    def build_vocab(self):
        unique_tokens = set(token for line in self.lines for token in self.tokenizer(line))
        vocab = {token: idx + 1 for idx, token in enumerate(unique_tokens)}
        vocab['<PAD>'] = 0  # Pad token
        return vocab

    def encode(self, text):
        tokens = self.tokenizer(text)
        padded_tokens = [self.vocab.get(token, self.vocab['<PAD>']) for token in tokens[:self.max_length]]
        padded_tokens += [self.vocab['<PAD>']] * (self.max_length - len(padded_tokens))
        return padded_tokens

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        encoded_line = self.encode(line)
        return torch.tensor(encoded_line), torch.tensor(encoded_line)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            x = layer(x, enc_out, x, trg_mask)
        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)

    def forward(self, src, trg, src_mask, trg_mask):
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

file_path = '/content/turkish.txt'
dataset = TransformerDataset(file_path, simple_tokenizer, max_length=50)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


model = Transformer(
    src_vocab_size=len(dataset.vocab),
    trg_vocab_size=len(dataset.vocab),
    embed_size=64,
    num_layers=6,
    heads=8,
    device=device,
    forward_expansion=4,
    dropout=0.1,
    max_length=70
).to(device)


print("Output layer size:", model.decoder.fc_out.out_features)
print("Vocab size:", len(dataset.vocab))


assert model.decoder.fc_out.out_features == len(dataset.vocab), "Çıktı katmanı boyutu kelime dağarcığı boyutuyla eşleşmiyor!"

def create_masks(question, answer, pad_idx):
    
    src_mask = (question != pad_idx).unsqueeze(1).unsqueeze(2)

   
    size = answer.size(1)  
    trg_pad_mask = (answer != pad_idx).unsqueeze(1).unsqueeze(2)
    trg_sub_mask = torch.tril(torch.ones((size, size), device=question.device)).bool()
    trg_mask = trg_pad_mask & trg_sub_mask

    return src_mask, trg_mask
# öğrenme değeri
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.vocab['<PAD>'])
pad_idx = dataset.vocab['<PAD>']


def create_padding_mask(seq, pad_idx):
    mask = (seq == pad_idx)  
    return mask.unsqueeze(1).unsqueeze(2)  

def create_look_ahead_mask(size, device):
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).type(torch.bool)
    return mask  



def train(model, data_loader, optimizer, loss_fn, pad_idx, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for question, answer in data_loader:
            question, answer = question.to(device), answer.to(device)
            src_mask = create_padding_mask(question, pad_idx).to(device)
            trg_mask = create_padding_mask(answer, pad_idx).to(device) & create_look_ahead_mask(answer.size(1), device)

            optimizer.zero_grad()
            output = model(question, answer, src_mask, trg_mask)
            loss = loss_fn(output.view(-1, output.size(-1)), answer.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Average Loss: {total_loss/len(data_loader)}")

train(model, loader, optimizer, loss_fn, pad_idx, device, epochs=10)

def predict(model, sentence, vocab, max_length, device, pad_idx):
    model.eval()
    tokens = [vocab.get(token, vocab['<PAD>']) for token in simple_tokenizer(sentence)]
    padded_tokens = tokens[:max_length] + [vocab['<PAD>']] * (max_length - len(tokens))
    input_tensor = torch.tensor([padded_tokens], device=device)
    src_mask = create_padding_mask(input_tensor, pad_idx)
    trg_mask = create_look_ahead_mask(input_tensor.size(1), device)
    with torch.no_grad():
        outputs = model(input_tensor, input_tensor, src_mask, trg_mask)
    predicted_indices = outputs.argmax(2).squeeze().tolist()
    predicted_tokens = [key for idx in predicted_indices for key, value in vocab.items() if value == idx]
    return " ".join(predicted_tokens)



test_sentence = "1896 Yaz Olimpiyatlarını anlatır mısın?"
predicted_tokens = predict(model, test_sentence, dataset.vocab, max_length=15, device=device, pad_idx=pad_idx)
print(f"Tahmini devamı: {predicted_tokens}")
