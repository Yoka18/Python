import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embed boyutunun Heads'e göre bölünebilir olması gerekir"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

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
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.fc_out = nn.Linear(embed_size, src_vocab_size)

    def forward(self, x, mask):
        enc_src = self.encoder(x, mask)
        output = self.fc_out(enc_src)
        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Basit bir tokenizer fonksiyonu
def simple_tokenizer(text):
    return text.split()

class SimpleDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.sentences = file.read().splitlines()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab = self.build_vocab()
        
    def build_vocab(self):
        unique_tokens = set(token for sentence in self.sentences for token in self.tokenizer(sentence))
        vocab = {token: idx + 1 for idx, token in enumerate(unique_tokens)}
        vocab['<PAD>'] = 0
        vocab['<MASK>'] = len(vocab)  # Mask tokeni ekle
        return vocab

    def encode(self, text):
        tokens = self.tokenizer(text)
        padded_tokens = [self.vocab.get(token, self.vocab['<PAD>']) for token in tokens[:self.max_length]]
        padded_tokens += [self.vocab['<PAD>']] * (self.max_length - len(padded_tokens))
        return padded_tokens

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        encoded_sentence = self.encode(sentence)
        return torch.tensor(encoded_sentence), torch.tensor(encoded_sentence)


file_path = '/content/turkish.txt'
dataset = SimpleDataset(file_path, simple_tokenizer, max_length=20)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Transformer modeli, optimizer ve loss fonksiyonunu tanımlama
model = Transformer(
    src_vocab_size=len(dataset.vocab),  # Kelime haznesi büyüklüğü
    embed_size=32,                     # Gömme vektör boyutu
    num_layers=4,                       # Transformer bloğundaki katman sayısı
    heads=8,                            # Multi-head attention için başlık sayısı
    device=device,                      # Cihaz (CPU veya CUDA)
    forward_expansion=4,                # Feed-forward ağının genişleme oranı
    dropout=0.1,                        # Dropout oranı
    max_length=20                      # Pozisyonel kodlama için maksimum uzunluk
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
loss_fn = nn.CrossEntropyLoss()

# Rastgele maskeleme fonksiyonu
def random_masking(inputs, vocab, mask_prob=0.15):
    mask_idx = vocab['<MASK>']  # Vocab içindeki mask indexi
    mask = (torch.rand(inputs.shape, device=inputs.device) < mask_prob).long()
    masked_inputs = inputs * (1 - mask) + mask_idx * mask
    return masked_inputs

# Modeli eğitme fonksiyonu
def train(model, data_loader, loss_fn, optimizer, device, vocab, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, _ in data_loader:  
            inputs = inputs.to(device)
            masked_inputs = random_masking(inputs, vocab)
            optimizer.zero_grad()
            outputs = model(masked_inputs, None)  # Maskelenmiş girdilerle modeli çalıştır
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), inputs.view(-1)) 
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(data_loader)}")

# Modeli eğitme
train(model, loader, loss_fn, optimizer, device, dataset.vocab, epochs=100)

# Tahmin yapma fonksiyonu
def predict(model, sentence, vocab, max_length, device):
    model.eval()
    tokens = [vocab.get(token, vocab['<PAD>']) for token in simple_tokenizer(sentence)]
    padded_tokens = tokens[:max_length] + [vocab['<PAD>']] * (max_length - len(tokens))
    input_tensor = torch.tensor([padded_tokens], device=device)
    
    with torch.no_grad():
        outputs = model(input_tensor, None)
    
    predicted_indices = outputs.argmax(2).squeeze().tolist()
    predicted_tokens = [key for idx in predicted_indices for key, value in vocab.items() if value == idx]
    
    return " ".join(predicted_tokens)

# Örnek bir test cümlesi ile tahmin yapma
test_sentence = "aman tanrım bu normal olmaya başladığınız anlamına mı geliyor "
predicted_tokens = predict(model, test_sentence, dataset.vocab, max_length=5, device=device)
print(f"Predicted continuation: {predicted_tokens}")
