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

# Örnek bir veri seti sınıfı
class SimpleDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length):
        self.sentences = sentences
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

# Veri seti
sentences = [
    "Merhaba, bugün nasılsınız? Umarım keyifli bir gün geçiriyorsunuzdur.",
    "Yapay zeka, bilgisayar bilimlerinin en heyecan verici dallarından biridir.",
    "Bu kurs, Python programlama dilinde temel bilgileri öğretmeyi amaçlamaktadır.",
    "Futbol, dünya genelinde en popüler sporlardan biridir.",
    "İklim değişikliği, gezegenimiz için ciddi sonuçlar doğurabilir.",
    "Dengeli bir diyet, sağlıklı bir yaşam tarzının olmazsa olmazıdır.",
    "Küresel ekonomi, birçok ülkenin finansal sistemleri arasındaki etkileşime dayanır.",
    "Sanat, insan duygularını ifade etmenin ve toplumsal mesajlar vermenin güçlü bir yoludur.",
    "Bilgisayarlar, modern iş dünyasının vazgeçilmez bir parçası haline gelmiştir.",
    "Sürdürülebilir kalkınma, ekonomik büyüme ile çevresel korumanın dengelenmesini gerektirir.",
    "Kuantum bilgisayarlar, bilgi işlemde devrim yaratabilir.",
    "Müzik, evrensel bir dil olarak kabul edilir ve insanları birleştiren bir güçtür.",
    "Okyanuslar, dünya yüzeyinin yaklaşık yüzde yetmişini kaplar ve büyük bir biyoçeşitlilik sunar.",
    "Edebiyat, toplumun aynası olarak görülebilir; toplumsal değişimlere ışık tutar.",
    "Yenilenebilir enerji kaynakları, fosil yakıtlara bir alternatif sunar ve çevre üzerindeki etkileri azaltır.",
    "Sağlık teknolojisi, medikal teşhis ve tedavi yöntemlerinde önemli ilerlemeler sağlamıştır.",
    "Finans piyasaları, ekonomik istikrar için kritik öneme sahiptir ve sürekli olarak izlenir.",
    "Eğitim, bireylerin kişisel ve mesleki gelişiminde temel bir rol oynar.",
    "Siber güvenlik, dijital çağda veri korumanın önemini vurgular.",
    "Uzay keşfi, insanlığın bilgi sınırlarını genişletir ve yeni dünyaları keşfetme olanağı sunar."
]

additional_sentences = [
    "Blockchain teknolojisi, finansal işlemlerin güvenliğini artırmada devrim yaratmıştır.",
    "Yenilikçi teknolojiler, çevresel sürdürülebilirliği destekleme potansiyeline sahiptir.",
    "Küresel ısınma, dünya çapında ciddi ekolojik ve sosyoekonomik etkilere neden olmaktadır.",
    "Modern mimari, şehir planlaması ve sürdürülebilir tasarım arasındaki ilişki giderek artmaktadır.",
    "Otonom araçlar, ulaşım sektöründe nasıl bir devrim yaratabilir?",
    "Sanal gerçeklik, eğitimden eğlenceye birçok alanda kullanım olanakları sunar.",
    "Yapay zekanın etik kullanımı, teknolojinin toplum üzerindeki etkilerini şekillendirmede kritik öneme sahiptir.",
    "Genetik mühendisliği, tıbbi araştırmalarda yeni kapılar açmaktadır.",
    "Robotik teknolojiler, üretim süreçlerini nasıl dönüştürmektedir?",
    "Veri bilimi ve büyük veri analizi, iş kararları üzerinde giderek daha fazla etkiye sahiptir.",
    "1) Bol bol dinlendiğinizden ve bol sıvı tükettiğinizden emin olun. 2) Buhar soluyun, lavaboda sıcak su akıtın. Buharı hapsetmek için üzerine bir havlu örtün ve kişinin suyun aktığı lavabonun üzerine eğilmesini sağlayın. Ona 5 ila 10 dakika boyunca ağzı ve burnundan derin nefes almasını söyleyin. Günde birkaç kez tekrarlayın. 3) Ona tavuk suyu veya ballı sıcak çay içirin. 12 aydan küçük bir çocuğa bal vermeyin.",
    "Gaz problemleri nasıl tedavi edilir?", "Gastrointestinal problemlerim olursa ne yapmalıyım?", 
    "Gaz problemi için hangi ilacı almalıyım?", 
    "Gaz problemleri nasıl iyileştirilir?",
    "Cilt problemleri nasıl tedavi edilir?", 
    "Cilt alerjim olursa ne yapmalıyım?", 
    "Cilt alerjisi için hangi ilacı almalıyım?", 
    "Cilt alerjisi nasıl iyileştirilir?",
    "1) Hidrokortizon kremi. 2) Kalamin losyonu gibi merhemler. 3) Antihistaminikler. 4) Soğuk kompresler. 5) Yulaf ezmesi banyoları. 6) Spesifik döküntünüz için en iyi olanı doktorunuzla konuşun.",
    "Karın ağrısı nasıl tedavi edilir?", 
    "Karın ağrısı yaşarsam ne yapmalıyım?", 
    "Karın ağrısı için hangi ilacı almalıyım?", 
    "Karın ağrısı nasıl iyileştirilir?",
    "1) Su, et suyu veya su ile seyreltilmiş meyve suyu gibi içilecek berrak sıvılar sağlayın. 2) Tuzlu krakerler, sade ekmek, kuru tost, pirinç, jelatin veya elma püresi gibi hafif yiyecekler servis edin. 3) Tüm semptomlar geçene kadar 48 saat boyunca baharatlı veya yağlı yiyeceklerden ve kafeinli veya gazlı içeceklerden kaçının.",
    "1) Kırık bir parmakta ağrı ve şişliği azaltmak için ayağı yükseltin, yarayı buzlayın ve ayağı kullanmayın. 2) Kırığın ciddiyetine bağlı olarak, parmak yerine oturtulması (redükte edilmesi) gerekebilir ve bazı bileşik parmak kırıkları ameliyat gerektirebilir. 3) Çoğu kırık parmak altı hafta içinde komplikasyon olmadan iyileşir.",
    "Devlet, her yıl bir kez aldığı mtv'yi, geçmişte anayasa mahkemesinin iptal etmiş olmasına rağmen pat diye iki kez almaya karar verek halka yaptığı sürprizi mesela sildiği vergileri de 'vazgeçtim istiyorum' diyerek şirketlere yapsa...",
    "Eskişehir'de bir pavyonda 'Afet' isimli kadın için çıkan silahlı çatışmada 1 kişi öldü. ",
    "Oxi'nin internetin her yerinde easter egg gibi çıkması beni çok mutlu ediyor",
    "5597 sayılı Yurt Dışına Çıkış Harcı Hakkında Kanun ile Çeşitli Kanunlarda Değişiklik Yapılması Hakkında Kanun uyarınca, yurt dışına çıkış yapan Türkiye Cumhuriyeti vatandaşlarından, çıkış başına 50 Türk Lirası harç alınmaktadır. Cumhurbaşkanının, bu miktarı üç katına kadar artırma veya sıfıra kadar indirme yetkisi bulunmaktadır.",
    "18/3/2022 tarihinde yayınlanan CK ile çıkış başına tahsil edilecek harcın tutarı **150 TL** olarak belirlenmiş olup, çıkış başına tahsil edilecek harcın 15 TL'si TOKİ'ye aktarılmakta ve kalan kısım genel bütçeye gelir kaydedilmektedir.",
    "Çıkış tarihi itibarıyla yurt dışında oturma izni bulunanlar",
    "Yurt dışına çıkış yapan Türkiye Cumhuriyeti vatandaşlarından çıkış başına alınan harç tutarının **3.000 TL** olarak belirlenmesi, harcın her yıl yeniden değerleme oranında artırılmasının sağlanması ve Cumhurbaşkanına verilen yetkinin buna göre düzenlemesi önerilmektedir.",

]
sentences.extend(additional_sentences)

dataset = SimpleDataset(sentences, simple_tokenizer, max_length=10)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Transformer modeli, optimizer ve loss fonksiyonunu tanımlama
model = Transformer(
    src_vocab_size=len(dataset.vocab),  # Kelime haznesi büyüklüğü
    embed_size=256,                     # Gömme vektör boyutu
    num_layers=4,                       # Transformer bloğundaki katman sayısı
    heads=8,                            # Multi-head attention için başlık sayısı
    device=device,                      # Cihaz (CPU veya CUDA)
    forward_expansion=4,                # Feed-forward ağının genişleme oranı
    dropout=0.1,                        # Dropout oranı
    max_length=20                      # Pozisyonel kodlama için maksimum uzunluk
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
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
train(model, loader, loss_fn, optimizer, device, dataset.vocab, epochs=50)

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
test_sentence = "Robotlar gelecekte işgücünün hangi alanlarını dönüştürecek?"
predicted_tokens = predict(model, test_sentence, dataset.vocab, max_length=10, device=device)
print(f"Predicted continuation: {predicted_tokens}")
