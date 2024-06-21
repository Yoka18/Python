import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


dataset = load_dataset("Mursel/Turkish-wikipedia-10k")

# SelfAttention Class
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embed size needs to be divisible by heads"

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

        attention = torch.softmax(energy / (self.embed_size ** 0.5), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

# TransformerBlock Class
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

# Encoder Class
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

# Transformer Class
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.fc_out = nn.Linear(embed_size, src_vocab_size)

    def forward(self, x, mask):
        enc_src = self.encoder(x, mask)
        output = self.fc_out(enc_src)
        return output

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple tokenizer function
def simple_tokenizer(text):
    return text.split()

class CombinedDataset(Dataset):
    def __init__(self, data_qa, data_wiki, tokenizer, max_length):
        self.data_qa = data_qa
        self.data_wiki = data_wiki
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab = self.build_vocab()

    def build_vocab(self):
        unique_tokens = set(token for qa in self.data_qa for text in qa if isinstance(text, str) for token in self.tokenizer(text))
        unique_tokens.update(token for text in self.data_wiki if isinstance(text, str) for token in self.tokenizer(text))
        vocab = {token: idx + 1 for idx, token in enumerate(unique_tokens)}
        vocab['<PAD>'] = 0
        vocab['<MASK>'] = len(vocab)
        return vocab

    def encode(self, text):
        tokens = self.tokenizer(text)
        padded_tokens = [self.vocab.get(token, self.vocab['<PAD>']) for token in tokens[:self.max_length]]
        padded_tokens += [self.vocab['<PAD>']] * (self.max_length - len(padded_tokens))
        return padded_tokens

    def __len__(self):
        return len(self.data_qa) + len(self.data_wiki)

    def __getitem__(self, idx):
        if idx < len(self.data_qa):
            qa_pair = self.data_qa[idx]
            if len(qa_pair) != 2:
                raise ValueError(f"QA pair does not have exactly two elements: {qa_pair}")
            question, answer = qa_pair
            encoded_question = self.encode(question)
            encoded_answer = self.encode(answer)
            return torch.tensor(encoded_question), torch.tensor(encoded_answer)
        else:
            text = self.data_wiki[idx - len(self.data_qa)]
            encoded_text = self.encode(text)
            return torch.tensor(encoded_text), torch.tensor(encoded_text)


# Load Wikipedia dataset
wiki_dataset = load_dataset("Mursel/Turkish-wikipedia-10k")
wiki_data = wiki_dataset['train']['content']

# Question-answer dataset
qa_data = [
    ("Türk Tarih Kurumu nedir?", "Türk Tarih Kurumu, Atatürk tarafından 1931'de kurulmuştur."),
    ("Yapay zeka nedir?", "Yapay zeka, insan zekasını taklit eden bilgisayar sistemleridir."),
    ("Python programlama dili nedir?", "Python, yüksek seviyeli, genel amaçlı bir programlama dilidir."),
    ("İklim değişikliği nedir?", "İklim değişikliği, uzun vadeli hava durumu değişikliklerini ifade eder."),
    ("Blockchain teknolojisi nedir?", "Blockchain, verilerin güvenli ve merkezi olmayan bir şekilde saklanmasını sağlar."),
    # Daha fazla soru-cevap eklenebilir
]


max_length = 30
combined_dataset = CombinedDataset(qa_data, wiki_data, simple_tokenizer, max_length)
loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

# Transformer modeli, optimizer ve loss fonksiyonunu tanımlama
model = Transformer(
    src_vocab_size=len(combined_dataset.vocab),  # Kelime haznesi büyüklüğü
    embed_size=256,                     # Gömme vektör boyutu
    num_layers=6,                       # Transformer bloğundaki katman sayısı
    heads=8,                            # Multi-head attention için başlık sayısı
    device=device,                      # Cihaz (CPU veya CUDA)
    forward_expansion=4,                # Feed-forward ağının genişleme oranı
    dropout=0.1,                        # Dropout oranı
    max_length=max_length               # Pozisyonel kodlama için maksimum uzunluk
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Rastgele maskeleme fonksiyonu
def random_masking(inputs, vocab, mask_prob=0.15):
    mask_idx = vocab['<MASK>']  # Vocab içindeki mask indexi
    mask = (torch.rand(inputs.shape, device=inputs.device) < mask_prob).long()
    masked_inputs = inputs.clone()
    masked_inputs[mask == 1] = mask_idx
    return masked_inputs

# Modeli eğitme fonksiyonu
def train(model, data_loader, loss_fn, optimizer, device, vocab, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            masked_inputs = random_masking(inputs, vocab)
            optimizer.zero_grad()
            outputs = model(masked_inputs, None)  # Maskelenmiş girdilerle modeli çalıştır
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(data_loader)}")

# Modeli eğitme
train(model, loader, loss_fn, optimizer, device, combined_dataset.vocab, epochs=50)

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
test_sentence = "Türk Tarih Kurumu nedir?"
predicted_tokens = predict(model, test_sentence, combined_dataset.vocab, max_length=30, device=device)
print(f"Predicted continuation: {predicted_tokens}")

