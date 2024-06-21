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
    "YKS Başarı Sıralaması Bursu: Üniversitelerin tarih, tarih eğitimi ve tarih öğretmenliği bölümleri ile Yönetim Kurulu tarafından belirlenen sosyal bilim bölümlerinde öğrenim gören lisans öğrencilerine verilen burs türüdür. Bursa, belirtilen bölümlerden birine burs duyurusunun yapıldığı yılda yerleşmiş olmak ve ilk üç tercihinden birine yerleşmiş olmak ve burs ilanında duyurulacak olan şartlarını yerine getiren öğrenciler başvurabilmektedir. YKS başarı sıralamasında ilk 1-1000 arasında yer alanlara aylık 6.000 TL, 1001-5000 arasında yer alanlara aylık 4.500 TL, 5001-15000 arasında yer alanlara ise aylık 3.000 TL, 15.001-30.000 arasında yer alanlara 2.250 TL olmak üzere dört kademede YKS başarı sıralaması bursu verilmektedir.",
    "Yerleştirme Sıralaması Bursu: Burs duyurusunun yapıldığı yılda üniversitelerin tarih, tarih eğitimi ve tarih öğretmenliği bölümlerine yerleşmiş olup YKS puanıyla öğrenim gördüğü bölüme birinci sırada (en yüksek puan ile) yerleşen lisans öğrencilerine verilen burs türüdür. Yerleştirme sıralaması bursuna hak kazanan bursiyerlere aylık 2.250 TL burs verilmektedir.",
    "Burs Verilecek Lisans Programları: Tarih, Tarih Eğitimi, Tarih Öğretmenliği, Arkeoloji, Sanat Tarihi, Arkeoloji ve Sanat Tarihi, Antropoloji, Coğrafya, Felsefe, Sosyoloji, Siyaset Bilimi, Siyaset Bilimi ve Uluslararası İlişkiler, Uluslararası İlişkiler",
    "Türk Tarih Kurumu, ülkemizde bizzat Atatürk’ün direktifleriyle kurulan kurumların başında gelmektedir. Atatürk, özellikle Avrupa devletlerinin ders kitaplarında yer alan Türkler hakkındaki olumsuz iddialara ve “barbar” deyimi kullanılarak bir istilacı kavim şeklinde gösterilmelerine karşılık, bunun böyle olmadığının, cihan tarihinde en eski çağlardan beri hakiki yerinin ne olduğunun ve medeniyete ne gibi hizmetlerinin bulunduğunun araştırılması gerektiğine inanmaktaydı.",
    "İşte bu sebeple, 28 Nisan 1930 tarihinde, Atatürk’ün de bizzat katıldığı Türk Ocakları’nın VI. Kurultayı’nın son oturumunda, O’nun direktifleriyle, Âfet İnan tarafından 40 imzalı bir önerge sunulmuş ve “Türk tarih ve medeniyetini ilmî surette tedkik etmek için hususî ve daimî bir heyetin teşkiline karar verilmesini ve bu heyetin azasını seçmek salahiyetinin Merkez heyetine bırakılmasını teklif ederiz” denilmiştir.",
    "Aynı gün Kurultay’da yapılan görüşme sonucunda Türk Ocakları Kanunu’na, 84. madde olarak “Merkez Heyeti, Türk tarih ve medeniyetini ilmî surette tedkik ve tetebbu eylemek vazifesiyle mükellef olmak üzere bir Türk Tarih heyeti teşkil eder.” şeklinde bir madde eklenmiştir. Bu karar çerçevesinde 16 üyeden oluşan bir “Türk Tarihi Tedkik Heyeti” teşkil edilmiş, heyet ilk toplantısını 4 Haziran 1930 tarihinde yapmış, Yönetim Kurulu ve diğer üyeleri seçmiştir.",
    "Yönetim Kurulu: Başkan Tevfik Bıyıklıoğlu, Başkanvekilleri Yusuf Akçura ve Samih Rıfat, Genel Sekreter Dr. Reşit Galip.",
    "Üyeler: Âfet İnan, İsmail Hakkı Uzunçarşılı, Hâmid Zübeyir Koşay, Halil Edhem, Ragıb Hulûsi, Reşid Safvet Atabinen, Zâkir Kadîrî, Sadri Maksudi Arsal, Mesaroş (Ankara Etnografya Müzesi uzmanı), Mükrimin Halil Yinanç, Vâsıf Çınar ve Yusuf Ziya Özer’den teşekkül etmiştir.",
    "Bu heyet, Türk Tarihinin Ana Hatları adıyla yaptığı ilk çalışmayı yayımlamıştır.",
    "29 Mart 1931 tarihinde Türk Ocakları’nın VII. Kurultay’ında kapatılma kararı alınınca, bu defa 15 Nisan 1931’de “Türk Tarihi Tedkik Cemiyeti adı ile yeniden teşkilatlanmış ve 1930’daki ilkeler temel alınarak faaliyetlerine devam etmiştir. Kurumun adı 1935 yılında “Türk Tarihi Araştırma Kurumu” olarak değiştirilmiş, daha sonra ise “Türk Tarih Kurumu”na çevrilmiştir. Kurum bu dönem içerisinde dört ciltlik lise tarih kitaplarını hazırlamış ve bu kitaplar MEB yayını olarak basılmıştır. Türk Tarih Kurumu’nun bastığı ilk kitap, Birinci Türk Tarih Kongresi: Konferanslar, Müzakere Zabıtları adlı bildiri metinleri kitabıdır (1932). Fakat bu kitabın üstünde de TTK değil Maarif Vekâleti kaydı vardır. Pîrî Reis Haritası Hakkında İzahname (1935) adlı eserin üzerinde ise “Türk Tarihi Araştırma Kurumu Yayınlarından: No: 1” yazmaktadır.",
    "Yine Pîrî Reis’in Kitâb-ı Bahriye’sinin (1935) yayın numarası 2’dir. Daha sonra Alacahöyük kazı raporları basılmış, 1937 yılından itibaren ise, adını bizzat Atatürk’ün koyduğu, BELLETEN yayın hayatına başlamıştır.",
    "Atatürk, hayatının son dönemlerine kadar Kurumun çalışmalarıyla yakından ilgilenmiş, birçok defa çalışma planını kendisi tespit etmiş ve birçok toplantıya bizzat katılmıştır. O’nun bu Kurum’a ve tarihe verdiği önem, 5 Eylül 1938’de düzenlediği vasiyetnâme ile İş Bankası’ndaki hisselerinin gelirinin yarısını Türk Tarih Kurumu’na bağışlamasından anlaşılmaktadır. Nitekim Atatürk’ten sonra gelen bütün Cumhurbaşkanları da bir gelenek olarak Kurum’un koruyucu başkanları olmuştur. 25 Mayıs 1940’ta İçişleri Bakanlığı’nca onaylanan yeni cemiyetler kanununa göre yeniden düzenlenen tüzüğünün 2. maddesinde, Kurum’un Reisicumhur İsmet İnönü’nün yüksek himayeleri altında bulunduğu hükmü yer almış, 3. maddesinde de, “Maarif Vekili bu Kurum’un fahrî reisidir” denilmiştir. Kurum, Bakanlar Kurulu’nun 21 Ekim 1940 tarih ve 2/14556 sayılı kararnamesiyle “Kamu Yararına Çalışan Dernekler” arasına alınmıştır. Türk Tarih Kurumu, tüzelkişiliğe sahip olarak, 7 Kasım 1982’de kabul edilen Türkiye Cumhuriyeti Anayasası’nın 134. maddesi ile kurulan Atatürk Kültür, Dil ve Tarih Yüksek Kurumu bünyesine dâhil edilmiştir. Türk Tarih Kurumu bu dönemden itibaren de ilk kuruluş amaçları doğrultusunda çalışmalarına devam etmiş ve etmektedir.",
    "Türk Tarih Kurumu, ilmî araştırma ve yayınları yanı sıra, ilki 2-11 Temmuz 1932 tarihlerinde toplanan ve belli aralıklarla günümüze kadar XVII.’sini gerçekleştirdiği milletlerarası nitelikte “Türk Tarih Kongreleri” yapmaktadır. 20-25 Eylül 1937 yılında Dolmabahçe’de yapılan II. Kongre, uluslararası nitelik kazanmış, yabancı bilim adamları da bu kongreye katılmışlardır. Bu Kongre, Türk tarihinin açıklanması ve belgelenmesi amacını gütmüştür. Ayrıca, Kongre dolayısıyla, tarih öncesinden Cumhuriyet dönemine dek yurdumuzda ve Ortadoğu’da gelişen büyük uygarlıkları, maketler, mülajlar, resimler ve grafiklerle canlandıran bir sergi düzenlenmiş ve bu sergi Atamızın ölümüne dek Dolmabahçe’de kalmıştır.",
    "Kuruluşundan başlayarak çalışmalarını eski Türk Ocağı Halkevleri binasında sürdüren Kurum, 1940 yılı sonlarında Dil ve Tarih-Coğrafya Fakültesi’nde ayrılan bir bölüme geçmiştir. Ancak her gün zenginleşen kütüphanesi, çalışmaları ve gelişen basımevi dolayısıyla bu yer yetersiz kalmış, 12 Kasım 1967 günü, projesi Sayın Turgut Cansever tarafından çizilen bugünkü modern binasına taşınmıştır. Bu bina 1980 yılında “Uluslararası Ağahan Mimarî Ödülü”nü almıştır.",
    "Türk Tarih Kurumu, 2876 sayılı kanununda da belirtildiği üzere, 1931 yılındaki kuruluş gayesi olan Türk ve Türkiye tarihini ve bunlarla ilgili konuları, Türklerin medeniyete hizmetlerini ilmî yoldan incelemek, araştırmak, tanıtmak, yaymak ve yayımlar yapmak, bunlara dayanarak da Türk tarihini ve Türkiye tarihini yazmak konusunda çalışmalarını günümüze kadar sürdürmüş ve yayınlarını toplam XXXI dizide (Eski diziler) toplamıştır. Bu diziler Türk Tarih Kurumu Yayın Çalışma Kurulu tarafından yenilenmiştir. Eski ve yeni dizilerde başlangıçtan bu yana, dergiler dâhil toplam 1400’e yakın eser yayımlanmıştır. 1937 yılından bu yana yayımlanan Belleten dergisinin Aralık 2017 sonu itibariyle 292. sayısı çıkmıştır. Belgeler dergisinin ise 38. sayısı basılmıştır. İlk kez yayınladığımız Tarih Yıllığı ise Türk Tarih Kurumu – Kırkambar 2013 adıyla 2 cilt olarak basılmıştır.",
    "Dergi yayıncılığı açısından bir önemli gelişme de 1988 yılında tek sayı çıkartılıp sürdürülmeyen Höyük dergisinin yeniden yayınlanmaya başlamasıdır. Arkeoloji, sanat tarihi vb. yazılarının değerlendirileceği ve yılda iki kez yayınlanacak olan Höyük’ün 7 sayısı 2014’de çıkartılmıştır.",
    "Atatürk’ün direktifleriyle 22 Ağustos 1935’te Kurum’un kendi parası ve kendi elemanlarıyla başlattığı ilk kazı olan “Alacahöyük Kazısı”ndan ayrı olarak Trakya ve Anadolu’nun çeşitli bölgelerinde arkeolojik araştırmalar sürdürülmüştür. Bu kazılardan çıkan eserler pek çok müzemizi süslemektedir. Bugün her yıl yaklaşık 40-50 kazıya maddî destek verilmektedir. Bununla birlikte Türk Tarih Kurumu, amaçları doğrultusunda Türk tarihinin pek çok meselesine ışık tutmak maksadıyla hazırlanan projeleri de desteklemektedir.",
    "Bunlardan “Türkiye’nin Sosyal ve Kültürel Tarihi”, “Başlangıçtan Günümüze Türk Dünyası Tarihi”, “Türk Sufilik Tarihi”, “Yozgat ve Çevresi Sosyal ve Ekonomik Tarihi”, “Sondaj Metoduyla Ordu ve Yöresi İktisadî Tarihi” gibi kapsamlı olanlarını saymak mümkündür. Öte yandan Türk Tarih Atlası çalışmaları sürmektedir. “Türk Kültür Varlıkları Envanteri” çalışmaları kapsamında beş kitap, sekiz cilt olarak basılmıştır ve bu dizi sürmektedir.",
    "Yayınlarımızı, Ankara merkez bina, Kızılay Bayındır Sokak ve İstanbul Üsküdar semtinde yer alan kitap satış bürolarından, yanı sıra elektronik olarak https:\\emagaza-ttk.ayk.gov.tr adresinden, ayrıca Türkiye genelinde çeşitli illerdeki anlaşmalı bayilerimizden de temin edebilirsiniz. Yayınlarımız sadece bu satış yerlerinden okuyucuya ulaştırılmaktadır. Kurum yayınlarının işportada vb. yerlerde satılması yasal değildir. Talepler doğrultusunda Erzurum, İzmir, Kayseri, Konya gibi Anadolu’da merkezi konumda bulunan illerde de yayın satış büroları açılması planlanmaktadır.",
    "Türk Tarih Kurumu’nun ilmî çalışmalar için kurduğu Kütüphane ise ülkemizin en zengin ihtisas kütüphanelerinden biridir. Yaklaşık 250.000 cilt kitabın bulunduğu kütüphaneye, değişim ve satın alma yoluyla en son yayınlar sağlanmakta, yurtdışında 220, yurtiçinde 60 kurum ve kuruluş ile değişim yapılmaktadır. Kütüphanemiz son teknolojiye uygun donanımlar ile hizmet vermektedir. Kütüphanemizde elektronik katalog tarama hizmeti verilmektedir.. Kütüphanemizdeki Osmanlıca, Arapça ve Farsça kaynak eserler başta olmak üzere Batı dillerindeki eserlerden Uluslar arası Telif Yasasına göre telif bakımından yasal süresi dolmuş nadir eserler dijitalleştirilmiştir. TTK Kütüphanesi çevrimiçi kataloğundaki detaylı arama kısmından bu eserlerin metinlerine de ulaşılabilmektedir. Şu ana kadar nadir eser ve elyazması eserlerden yaklaşık 7500’ü dijitalleştirilmiş ve dijitalleştirme işlemi devam etmektedir. Böylece bu eserlere, katalog tarama programı üzerinden erişim sağlanarak elektronik ortamda tüm dünyanın erişimine açılmıştır.",
    "Görme engelli akademisyenlerin, kütüphane hizmetinden faydalanması için çalışmalar başlatılmıştır.",
    "Arşiv ve dokümantasyon açısından da Kurum hayli zengindir. Yakın zamanlar tarihi bakımından önemli arşiv belgeleri ve zengin bir Atatürk fotoğraf koleksiyonu bulunmaktadır. Fotoğraf ve belge koleksiyonunda bulunan malzemeden, yakın dönemde İzmir’in Kurtuluşu, Ayasofya Sergisi, Atatürk Fotoğrafları Sergisi, Kıbrıs’ta Hala Sultan Sergisi ve 100. Yılında Kudüs Sergisi düzenlenmiş olup sergi katalogları da basılmıştır. Arşivdeki materyallerin dijitalleştirme ve tasniflenmesi işlemi devam etmektedir. En son Enver Paşa ve Fahir İz Arşivinin analitik tasnifi tamamlanmıştır.",
    "Türk Tarih Kurumu, Osmanlı Öncesi ve Osmanlı Araştırmaları Uluslararası Komitesi (CIÉPO), Güney-Doğu Avrupa Araştırmaları Birliği (AIÉSEE), Uluslararası Tarih Bilimleri Komitesi (CISH), Uluslararası Askerî Tarih Komisyonu (CIHMC) ve Uluslararası Akademiler Birliği (UAI) gibi çeşitli uluslararası bilim kurullarının da üyesidir.",
    "Osman Gazi (1299 – 1326) Osmanlı Devleti’nin kurucusu olan Osman Gazi, 1258’de, Söğüt’te doğdu. Babası Ertuğrul Gazi, Annesi Halime Hatun’dur. Osman Gazi, uzun boylu, yuvarlak yüzlü, esmer tenli, ela gözlü ve kalın kaslıydı. Omuzları arası oldukça geniş, vücudunun belden yukarı kısmı, aşağı kısmına oranla daha uzundu. Başına kırmızı çuhadan yapılmış Çağatay tarzında Horasan tacı giyerdi. İç ve dış elbiseleri geniş yenliydi. Osman Gazi değerli bir devlet adamıydı. Dürüst, tedbirli, cesur, cömert ve adalet sahibiydi. Fakirlere yedirip, onları giydirmeyi çok severdi. Üzerindeki elbiseye kim biraz dikkatlice baksa, hemen çıkartıp ona hediye ederdi. Her ikindi vakti, evinde kim varsa onlara ziyafet verirdi. Osman Gazi, 1281 yılında Sögüt’te, Kayı Boyu’nun yönetimine geçtiğinde henüz 23 yaşındaydı. Ata binmekte, kılıç kullanmakta ve savaşmakta çok ustaydı. Aşiretin ileri gelenlerinden, Ömer Bey’in kızı Mal Hatun ile evlendi ve bu evlilikten ileride Osmanlı Devleti’nin başına geçecek olan oğlu Orhan Gazi doğdu. Sögüt’te temelleri atılan, altı yüzyıllık bir tarih diliminde ve üç kıtada hüküm sürecek olan Osmanlı Devleti’nin kurucusu Osman Gazi, 1326’da Bursa’da Nikris (goutte) hastalığından öldü.",
    "Orhan Gazi, 1281 yılında doğdu. Babası Osman Gazi, annesi Kayı aşiretinin ileri gelenlerinden Ömer Bey’in kızı Mal Hatun’du. Orhan Gazi, sarı sakallı, uzunca boylu, mavi gözlüydü. Yumuşak huylu, merhametli, fakir halkı seven, ûlemaya hürmetli, dindar, adalet sahibi, hesabını bilen ve hiçbir zaman telaşa kapılmayan, halka kendisini sevdirmiş bir beydi. Sık sık halkın arasına karışır, onları ziyaret etmekten çok hoşlanırdı. Orhan Gazi, Babası Osman Gazi’nin 1326’da vefatı üzerine beyliğin başına geçti. Orhan Gazi, 1346’da Bizans İmparatoru VI. Yoannis Kantakuzenos’un kızı Teodora ile evlendi. Ayrıca, Yarhisar Tekfur’unun kızı Holofira, Bilecik tekfuruyla evlendirilirken, düğün basılıp Holofira esir alındı ve Orhan Gazi ile evlendirildi. Müslüman olduktan sonra adı Nilüfer Hatun olarak değiştirildi; bu evlilikten, ileride Osmanlı Devleti’nin üçüncü hükümdarı olacak Murad Hüdavendigâr doğdu.",
    "Sultan Birinci Murad, 1326’da, Bursa’da doğdu. Babası Orhan Gazi, annesi Bizans tekfurlarından Yar Hisar Tekfuru’nun kızı olan Nilüfer Hatun’dur (Holofira). Sultan Birinci Murad, uzun boylu, değirmi yüzlü ve iri burunluydu. Kalın ve adaleli bir vücuda sahipti. Başına mevlevî sikkesi üzerine destar sarılı bir başlık giyerdi. Çok sade giyinir ve kırmızı zeminli beyaz elbiseden hoşlanırdı. İlk eğitimini, annesi Nilüfer Hatun’dan aldı. Daha sonra tahsilini tamamlamak için Bursa’ya gitti. Buradaki Medreselerde ilim ve sanat adamları ile beraber çalıştı. Sultan Birinci Murad, gayet nazik, sevimli ve çok halim selim bir insandı. Âlim ve sanatkârlara hürmet gösterir, fakirlere ve kimsesizlere şefkatli davranırdı. Dahî bir asker ve devlet adamıydı. “Derviş Gazilerin, Şeyhlerinin, Kralı Murad Gazi” diye anılan Sultan Birinci Murad, bütün hayatı boyunca plânlı ve programlı hareket etti. Sultan Birinci Murad, Bizans Kilisesi’ne göre bir kâfir ve İsa düşmanı olarak görülse de, fethettiği yerlerde yaşayan Hristiyan halka iyi davrandığı için onların sevgisini kazanmıştı. 1382 yılından itibaren “Murad Hüdavendigâr” diye anılan Sultan Birinci Murad, Birinci Kosova Savaşı’ndan sonra savaş alanını gezerken, Sırp Asilzâdesi Milos Obraviç (Sırp Kralı Lazar’ın damadı) tarafından hançerlenerek şehit oldu (1389).",
    "Kurumumuzun Şeref Üyelerinden, Amerikalı akademisyen Prof. Justin McCarthy tarafından Osmanlı Devleti’nin son dönemine (takriben 1911 yılına) ait Anadolu ve Rumeli Vilayetleri haritaları hazırlanmıştır. Haritalar, ders ve sunumlar başta olmak üzere akademik amaçlara uygun olarak araştırmacıların istifadesine sunulmaktadır.",
    "Bilim ve Fen Kavramları; Bu yazımızın konusu, Atatürk’ün bilim ve fen kavramları hakkındaki düşünceleridir. Atatürk’ün bu konudaki görüşlerini incelemeye başlamadan önce, belleklerimizi biraz tazelemek için, bilim (ilim, science) ve fen kavramlarını kısaca tanımlamanın yerinde olacağını sanıyorum. Eski dilimiz Osmanlıca’da ilim, bilgi, vukuf ya da marifet anlamlarına geliyordu. Klâsik Orta Çağ İslâm düşüncesi İçinde ilim, öncelikle medrese ağırlıklı disiplinleri kapsıyordu. Yani, ilim denilince, özellikle dini bilgiler anlaşılıyordu. Günümüzde, bilim denilince de kendine özgü konusu, yöntemi olan ve olayların nedenlerini, yasalarını bulma amacına yönelen disiplinler anlaşılmaktadır. Bu anlamda bilim, doğru yöntemle elde edilen ve pratikle saptanan bilgilerin bütünüdür, diyebiliriz.",
    "Bilimsel bilgi; Bilimsel bilgiyi diğer bilgilerden ayıran şey onun objektif, yani nesnel olmasıdır. Çünkü, bu bilgi kişiden kişiye değişen biçimlerde yorumlanamaz. Bilimsel bilgi aynı zamanda geneldir, çünkü yalnız özel bazı olaylara değil, tüm olaylar topluluğuna uygulanır. Örneğin, yerçekimi yasası yere bırakılan bütün nesneler için geçerlidir. Bunun aksini kimse ileri süremez. Güneşin doğuşu ve batışı da kesindir. Öte yandan bütün madenlerin ısıtılınca genişleyeceğini tüm bilim adamları kabul etmektedir. Bilimler, günümüzde kabaca, toplum bilimleri ve doğa bilimleri olarak ikiye ayrılmaktadır. Doğa bilimleri denilince genelde matematik, fizik, kimya, biyoloji, astronomi gibi disiplinler anlaşılmaktadır. Toplum bilimleri içerisine ise tarih, ekonomi, sosyoloji, hukuk ve siyaset bilimi gibi disiplinler girmektedir. Atatürk’ün söylevlerinde ve konuşmalarında sık sık kullandığı “müspet ilim” (pozitif bilim) kavramına gelince, bu olgulara, olaylara (vakıalara, hâdiselere) dayanan bilim demektir. Pozitivizmin olgulara, olaylara bakışı genel ve objektiftir. Pozitivizm, özünde akılcılığa, gerçekçiliğe ve deneyciliğe dayanır. Yani, bu bilim anlayışına göre akla, mantığa, deneye dayanmayan hiç bir bilgi bilimsel bilgi olarak kabul edilemez.",
    "Fen kavramı; Fennin bugün kullanılan genel anlamı matematik, fizik, kimya gibi bilgilerin iş hayatına ve günlük hayata uygulanmasıdır. Bu anlamda fen, daha çok teknik ya da teknoloji anlamına gelir. Şemseddin Sami’nin “Kamus-u Türki”sine göre “fen” bir anlamda ilimlerin her çeşidi ve dalıdır. Fakat aslında ilim fenden daha geniş kapsamlıdır. Pek çok disiplini içine alır. Halbuki fen daha dar kapsamlı olup, hukuk, siyaset, sosyoloji, iktisat, etik, estetik, gramer, edebiyat gibi disiplinleri, fıkıh, hadis, kelâm, tefsir gibi dini bilgileri içermez. Kısacası, bu anlamda fen daha çok müspet (pozitif) bilimleri ve bu bilimlere dayanılarak yapılan uygulamaları göstermektedir. İşte, Atatürk, ilim ve fen derken ve çok defa bu iki kavramı beraber, yan-yana kullanırken daha çok pozitif bilim anlayışını vurgulamak istiyordu.",
    "II. Meşrutiyet Döneminde Bilim ve Fen Hakkında Düşünceler; Bilim ve fen kavramlarına bu şekilde genel olarak değindikten sonra şimdi de Atatürk’teki bilim ve fen anlayışını etkilemiş olan yerli ve yabancı düşün akımlarına, yazarlara ve bunların bu konudaki fikirlerine kısaca göz atmak istiyoruz. Prof. Dr. Sina Akşin’in “Jön Türkler ve İttihat ve Terakki” adlı eserinde de belirttiği gibi, 1908 yılında Hürriyetin İlanıyla başlayan İkinci Meşrutiyet ile birlikte bütün ülkede büyük bir özgürlük havası esmişti. Bu dönemde çeşitli fikir akımları aydın kamuoyuna tanıtılmaya başlandığı gibi, bu akımların temsilcileri olan düşünürler de ülkenin içinde bulunduğu duruma karşı ne yapılması gerektiği konusunda çözümler önermeye başlamışlardı. Profesör Hilmi Ziya Ülken İkinci Meşrutiyet’teki fikir akımlarını incelerken, Ziya Gökalp’i izleyerek, bu dönemde geçerli, yaygın düşünceleri başlıca Garpçılık, İslamcılık ve Türkçülük olarak üçe ayırmaktadır. Bu akımlardan Garpçılık (Batıcılık)ta, ona göre dörde ayrılmaktaydı: Tanzimat Medeniyetçileri, yani Tanzimat Batıcıları: bunlar Osmanlı İmparatorluğunu ıslahatlarla, reformlarla değiştirerek, yenileştirerek korumak isteyenlerdi.",
    "İkinci tür Batıcılar, Anglo-Sakson toplumsal ve siyasal yapısını örnek alarak oradaki temel siyasal yapıyı Osmanlı İmparatorluğu’na getirmek isteyenlerdi. Özellikle, Prens Sabahattin’in başını çektiği bu grup “şahsî teşebbüs” (ekonomide özel, kişisel girişim) ve adem-i merkeziyyet ilkelerini, yani yerinden yönetim ilkelerini savunuyordu. Batıcıların üçüncü grubu Pozitivistlerdi. O dönemin en etkili dergileri olan “Ulûm-u İktisadiye ve İçtimaiyye” dergisi ile “Servet-i Fünun” dergisi etrafında toplanan bu grup İkinci Meşrutiyetin en önemli siyasi partisi olan İttihat ve Terakki’nin temel dünya görüşünü ve programını oluşturmuştu diyebiliriz. Profesör Taner Timur, İttihat ve Terakki’deki bu görüşlerin önemli bir kısmının daha sonra Cumhuriyet Halk Partisi’nde de devam ettiğini söylemektedir. İkinci Meşrutiyet Pozitivistlerinin ünlü simaları arasında, bir. ara Meclis-i Mebusan Reisliği de yapmış olan, Ahmet Rıza Bey’i burada hatırlayabiliriz.",
    "Hilmi Ziya Ülken, Batı’ya hayran, köktenci, radikal batıcıları bu tasnifinde en sona almış bulunmaktadır. Bu akımın en ünlü siması, İttihat ve Terakki Partisi’ne öncülük etmiş ve “İttihad-ı Osmani” adlı örgütü kurmuş bulunan “İçtihat” dergisi sahibi Abdullah Cevdet’in üzerinde burada önemle durmak gerekir. Çünkü, Abdullah Cevdet daha İkinci Meşrutiyet’te Lâtin harflerini savunmuş ve Sirkeci’de şapka giyerek dolaşmıştı. Ünlü Fransız sosyologu Gustave le Bonn’un teorilerine hayran olan Abdullah Cevdet ülkeyi sarmış olan gerilik çemberini bir an önce kırmak ve yurdu esenliğe, refaha kavuşturmak gereğine inanmış bulunuyordu. Abdullah Cevdet askeri bir hekim olarak pozitif bilimlerin ve fennin kalkınmada ve ilerlemede rolünü iyi kavramıştı. Daha sonra gittiği Fransa’da bu görüşlerini daha da derinleştirmişti.",
    "O, bilimlerin, felsefenin, biyolojik materyalizmin ve Sosyal Darwinizm’in bütün insanlığın yönelmesi gereken temel amaçlar olması gerektiğini ve olacağını söylemekteydi. Genç araştırmacı Doç. Dr. Şükrü Hanioğlu’na göre Abdullah Cevdet din bilginlerinin görüşlerinin materyalist (maddeci) düşünce karşısında kesin olarak yanlış olduğuna ve yenileceğine kani bulunuyordu.",
    "Atatürk’ün bilim ve fen konusundaki düşüncelerinin oluşmasında pozitivizmin yanı sıra rasyonalizmin, yani akılcılığın da önemli yeri ve katkısı bulunmaktadır. Profesör Dr. Şerafettin Turan’a göre, aklı ve bilimi düşünce ve aksiyonda temel kılavuz, önder olarak kabul eden, bu nedenle de safsatalara ve hurafelere karşı çıkan Atatürk düşüncesinde ve özellikle onun lâiklik anlayışında Descartes’ın akılcı görüşünün tüm özelliklerini bulmaktayız. Bu nedenledir ki Descartes’ın “Metod Üzerine Düşünceler” adlı kitabı Atatürk’ün isteği ile Türkçeye çevrilmiştir. Akılcı düşüncenin bir başka büyük temsilcisi olan Kant hakkında da “Kant ve Felsefesi” adlı bir inceleme yayınlanmıştır.",
    "Atatürk’ün bilim, din, Tanrı, ilerlemede bilimin rolü konularında 1916 yılında okuduğu kitaplar arasında yer alan Şehbenderzade Ahmet Hilmi Efendi’nin görüşlerinden de burada kısaca bahsetmek yerinde olur.",
    "O devirlerde yetişmiş düşünürlerimizden olan Şehbenderzade Ahmet Hilmi, çağdaş yaşama geçmenin uzun sürecek yavaş bir gelişmeyle, yani evrimle, gerçekleşmeyeceğini belirterek, hızlı bir ilerlemeyi zorunlu görüyordu. O, İlerlememize engel olan nedenleri, yeni fikirlere düşmanlık, durağanlığı sevmek, derinliğe inmeyen taklitçilik ile yüzeysel bilgi olarak özetliyordu.",
    "Atatürk’ün düşünce oluşumunda, özellikle bilim, fen, ilerleme, uygarlık gibi konularda etkisi altında kaldığı kaynaklardan biri de İkinci Meşrutiyet’te Kılıçzade Hakkı Bey’in de dahil olduğu “İçtihat” dergisidir. Yine, bu fikir adamları içinde burada zikredilmesi gerekenler arasında Celal Nuri’yi de sayabiliriz. Özellikle hurafelere karşı bakış açısında ve bilimsel düşünceyi Türkiye’de Dayanışmacı Durkheim ekolüne sadık kalarak, yayan ve geliştiren büyük düşünür Ziya Gökalp’ı da bu bağlam içinde anmadan geçemeyeceğim. Atatürk’ün bilim ve fen kavramları hakkındaki düşüncelerini bu şekilde kısaca özetledikten sonra şimdi, bir kaç cümleyle de olsa, onu bu yolda teşvik eden, cesaretlendiren bazı şairlerden, söz etmek istiyoruz. Bunlar arasında, kuşkusuz en başta Tevfik Fikret’en bahsetmek gerekir. Sanırım Fikret, “Ferda” (Yarın, Gelecek) adlı şiirinde zamanın gençliğine seslenerek şöyle diyordu:",
    "Üniversiteler ve Atatürk; Yazımın son bölümünde bilimin üretildiği ana, temel kaynaklardan biri olan Üniversiteler hakkında Atatürk’ün düşüncelerine kısaca değinmek isterim. O, 1933 yılında şöyle demişti: “Üniversite kurmaya verdiğimiz önemi söylemek isterim. Bütün işlerimizde olduğu gibi maarifte ve yeni kurulan Üniversitede radikal tedbirlerle yürümek kati kararımızdır.”",
    "Nitekim aynı yıl Dar-ül-Fünun lağvedilmiş ve yerine İstanbul Üniversitesi adıyla anılan kurum kurulmuştu. Cumhuriyet döneminde Üniversite Reformu diye anılan bu hareketle, Türk devrimcileri hala skolastiğin ve medresenin izlerini taşıyan eski bir kurumun yerine programları çağdaş bilime ve teknolojiye dayanan bir yeni kurum kurmak istemişlerdi. İlâve edelim ki, yeni Üniversitede programların yenilenmesiyle birlikte, bilim adamları kadrosu da yenilenmiş ve bazı hocalar tensikata (ayıklanmaya) tâbi tutulmuşlardı. Üniversiteden ayrılan bu hocaların yerine de özellikle Almanya’dan gelen, daha doğrusu getirtilen, çoğu Musevi asıllı bilim adamları yerleştirilmişti.",
    "Hemen işaret edelim ki bu yabancı bilim adamlarının yeni Türk Üniversitesinin kurulmasında ve ilerlemesinde büyük katkıları olmuştur. Bunlar, bilimsel düşünceyi ve bilimsel yöntemleri Türkiye’ye tanıttıkları gibi çok değerli öğrenciler ve kadrolar da yetiştirmişlerdi. Bu hocaların hizmetlerini burada şükranla yad etmek bir gönül ve insanlık borcudur. Bu konudaki ilginç noktalardan biri de yurdumuza gelen yabancı bilim adamlarına aylık olarak o zamana göre çok yüksek bir meblağ olan, bin liranın verilmiş olmasıydı. Hemen hatırlatalım ki o dönemde bir milletvekili üç yüz lira maaş almaktaydı. Bu da, Atatürk’ün üretken, değerli bilim adamlarına verdiği önemi çok açık şekilde göstermektedir, sanırım.",
    "Profesör Dr. Utkan Kocatürk’ün ‘Atatürk’ün Fikir ve Düşünceleri” adlı eserinden öğrendiğimize göre Atatürk, Üniversite’de ders yılının açılması münasebetiyle kendisine çekilen saygı ve bağlılık telgrafına şu cevabı vermişti: “İstanbul Üniversitesi’nin açılışından çok sevinç duydum. Bu yüksek ilim ocağında kıymetli profesörlerin elinde Türk çocuğunun müstesna zekâ ve eşsiz kabiliyetinin çok büyük gelişmelere erişeceğine eminim.” Yine, Utkan Kocatürk’ün belirttiğine göre, O, 1932 yılında bilim adamlarına ışık tutacak şu sözleri de söylemiştir: “İlim tercümeyle olmaz, tetkikle, yani araştırmayla olur.” Bu ilginç sözlerin yanında Atatürk’ün bilim terimleriyle de ilgilendiğini burada belirtmek istiyoruz. Profesör Dr. Akil Muhtar Özden’in, naklettiğine göre, O, Üniversitelerde ve bilim âleminde kullanılan Arapça terimler hakkında şöyle demiştir: “Söz konusu tâbirler beynelmilel ilim sahasında kolaylıkla ilerlememize mânidir. Fen terimleri o surette yapılmalı ki, mânaları ancak istenilen şeyi ifade edebilsin.”",
    "Sonuç ve Genel Değerlendirme; Atatürk’ün bilim ve fen kavramları ve Üniversite hakkındaki düşüncelerine böylece genel olarak değindikten sonra şu sorulara da cevap vermenin yerinde olacağını sanıyorum. Atatürk’ün ölümünden bu yana geçen elli yıldan fazla zamandan beri Türk Üniversiteleri ve diğer bilim teknoloji odakları onun istediği çağdaş düzeye bir bütün olarak erişebilmiş midir? Her alanda olduğu gibi bu alanda da kaydedilen ilerlemelere, yetişen değerli bilim ve fikir adamlarına, yapılan bilimsel yayın ve araştırmalar ortaya konan eserlere rağmen, dünya genelinde, Türkiye’nin bugün çok ileri bir bilimsel ve teknolojik düzeyde olduğunu söylemek pek mümkün değildir. Ben, müsaadenizle bu konuda bazı önemli noktaları hatırlatmak isterim.",
    "Çağdaş uygarlığa süratle ulaşmak yolundaki Atatürk’ün büyük özlemini gerçekleştirmek için bilim, fen ve teknoloji alanında çok daha büyük atılımlar hattâ sıçramalar yapmaya mecburuz. Bu, kuşkusuz toplumdaki bütün katmanların, Parlamentonun, hükümetlerin, bürokrasinin, üniversitelerin, kamuoyunun, halkın, basının, TRT, gibi etkili medyaların elbirliğiyle güçlü desteğiyle belirli bir plan ve programa göre çözeceği bir problemdir. Aynı zamanda, şunu da belirteyim ki, bilimi yükseltmek, bilim adamını yüceltmek sadece bu yolda harcanacak parayla, bütçeyle, fonlarla, ilgili değildir. Bunlar kadar önemli olan bir husus da, bilime, bilimsel düşünceye, Özgür fikre ve özgür tartışmaya gösterilecek itibar ve saygıdır. Öğretmenlerini ve bilim adamlarını toplumun yüksek saygı sınırında görmek uzun asırlar Türk toplumuna egemen bir düşünce olmuştur. Maalesef, son zamanlarda bu düşünce çizgisinden yer yer sapan, öğretmeni, bilim adamını ve üniversiteyi sırf para ölçüleriyle ölçmeye ve değerlendirmeye çalışan bazı kısır görüşler toplumumuzda uç vermeye ve revaç bulmaya başlamıştır. Bu düşüncelerde ve yönde ileri gidilmesi, kuşkusuz Atatürk’ün zikrettiğimiz düşünceleriyle uyuşmadığı gibi, toplumumuzu bir süre sonra kısır bir döngüye sokmak istidadı gösteren yoz bir anlayışı yansıtmaktadır. Şu halde, bilime, bilim adamına, Öğretmene gereken saygıyı ve önemi göstermek ve göstertmek en başta bizzat bilim adamlarının kendilerine düşen bir görev olduğu gibi, toplumun yukarıda işaret ettiğimiz bütün katmanlarına ait temel bir ödevdir.",
    "Perihan Naci ELDENİZ; Atatürk annesini çok severdi. Annesinin sevdiği bir şarkıyı duyduğu zaman gözleri yaşarırdı. Şu hikâyeyi hem Atatürk’den hem de sonradan, kardeşi rahmetli Makbule Atadan’ dan dinlemiştim. Büyük taarruzdan önce idi. Biz harp hazırlıkları ile meşguldük. Cepheye hareket gününü Ankara’daki ecnebi mümessiller fark etmesinler diye bir çay ziyafeti tertiplenmişti. O gün Ankara’da herkes çay ziyafetine gitmişti. Ben de sefer kıyafetlerimi giyerek anneme veda için odasına girdim. Annem rahatsızdı. Yatağında oturuyordu. Elini öptüm izin istedim. “Nereye?” dedi. “Çay ziyafetine” dedim. “Bu kıyafet ziyafete mahsus değil” dedi. Çizmelerimi gösteriyordu. Fakat üstelemedi. Üzülmesin istiyordum. Biz gittikten, saatler geçtikten sonra meraklanmış, Merkez Kumandanını çağırtmış, ”Nerede benim oğlum?” “Efendim, çay ziyafetine gitti“. “Hayır, çay ziyafetine gitmedi. Ben biliyorum. O, muharebeye gitti. Bir kalem kâğıt getirin, benden ona bir mektup yazın.” Ve Atatürk’ün annesi Zübeyde Hanım oğluna şu mektubu yazdırmış: “Oğlum, seni bekledim. Dönmedin. Çay ziyafetine gideceğini söyledin. Ama ben biliyorum, sen cepheye gittin. Sana dua ettiğimi bilesin. Harbi kazanmadan dönme“.",
    "Atatürk, bu mektubu aldığını anlatırken sanki yeni almış gibi mütahassis oluyordu. “İşte benim annem” diye arkadaşlarına gösterdiği o mektup, zaferin kazanılmasına yardım etmişti. Çünkü o mübarek annenin dileği, cephede onu okuyan ve duyanların hepsini harekete geçirmişti. Atatürk, bütün annelere saygı gösterirdi. Zaten kadınları saymak, onlara yer vermek, onları öne geçirmek, anayı saymak ve bunu belirtmek değil midir? Küçük Ülkü’yü bile, dört yaşında olduğu halde sağına oturturdu, Bir gün o da anne olacak diye onu korurdu. Atatürk, tesadüf ettiği her anneye kendi annesini sayar gibi saygı göstermekle kendi öz annesini her defasında azizlemiştir. Atatürk’ümüzün kardeşi Makbule Atadan, annelerinin çok vatanperver ve evlat seven bir anne olduğunu söylemiştir. Selanik’teki evlerinde, küçük yaşta Mustafa’sına ayrı bir oda vermiş, bir yazı masası hazırlamış, onu dersleri ve düşünceleri ile yalnız bırakmasını her zaman bilmiştir.",
    "Atatürk’ümüzün kadın hakkındaki anlayışını kendi eliyle yazdığı notlardan okuyalım. Bu notlar, Yardım sevenler Derneğinin ismi, son şeklini almadan önce bir toplantıda isim münakaşa edilirken kendisi tarafından yazılmış, o zamanki reis Bayan Eldeniz’e okutulmuştu. Birisi “İsim, yoksul kadınlara yardım derneği olsun” demişti. Atatürk’ün yüzüne, düşünceli çatık kaşlı bir hava gelmiş ve fotokopisini bu yazının sonuna koyduğum şu notları yazmıştı:",
    "Yoksul kadın, burada hiçbir şeyi olmayan kadın anlamında alınmıştır. Halbuki kadın denilen bir varlık bizatihi yüksek bir varlıktır. Onun yoksulluğu olamaz. Kadın yoksul demek onun bağrından kopup gelen bütün beşeriyetin yoksulluğu demektir. Eğer beşeriyet bu halde ise kadına yoksul demek reva görülebilir. Hakikat bu mudur? Eğer kadın dünyada çalışan, muvaffak olan, zengin olan, maddi ve manevi zengin eden insanları yetiştirmişse, ona yoksul sıfatı verilebilir mi? verenler varsa onlara nankör denirse doğru olmaz mı?",
    "Bizce: Türkiye Cumhuriyeti anlamınca kadın bütün Türk tarihinde olduğu gibi bugün de en muhterem mevkide, her şeyin üstünde yüksek ve şerefti bir mevcudiyettir. Şimdi anlıyorum. Yoksul kadınlar cemiyetini teşkil edenlerin insanı mefkûrelerini …",
    "Anladığımı da birkaç sözle izah edeyim. Birkaç asırdan beri Türk kadınlığının manası unutulmuş, o, bunca varlıkların maddi manevi kaynağı olduğu halde yoksul bırakılmış; unutulmuş! Kadın Esirgeme Kurumu bu fahiş, müthiş hatanın tashihini lüzumlu gören Türk kadınlarının kurumudur. a) Büyük varlık ve faziletleri unutulmuş olan Türk kadınlığına ayağa kalkarak hürmetlerimizi göstermeliyiz. b) Bunu düşünerek Kadın Esirgeme Kurumunu kuran bugünki Türk kadınlarını dahi ayakta selâmlamalıyız.”",
    "Örüntüleri oluşturacak veri seti hazırlanırken özel bir alan seçilmesi ve değerlendirilecek cümlelerin bu alana ait olması daha sağlıklı sonuçlar verecektir. Bu alanlar ekonomi, spor, sağlık gibi konular olabilir.",
    "Örnek veri setini oluşturmak için seçilen alanın uzmanlarıyla, teknik uzmanlar bir araya gelip beraber çalışırsa daha verimli sonuçlar çıkacaktır. Ülkeler kendi resmi dillerine göre ulusal çapta, açık kaynak kodlu veri setleri oluşturmalıdır. Bunun için hem uzman bir teknik ekip hem de veri seti oluşturulacak alanla ilgili uzman kişiler bir araya gelmelidir. Burada en önemli noktalardan birisi de, oluşturulacak veri seti ulusal çapta olacağından verinin saklanacağı altyapıyı iyi kurgulamak olacaktır. Veriler herkes tarafından erişilebilir olacağı için en optimum çözümler bulunmalıdır. Ülkemizde yapılacak böyle bir çalışma doğal dil işleme konusunda büyük bir adım olacaktır. Cümleden anlam çıkarma konusunda yeni",
    "I. Dünya Savaşı sırasında Osmanlı ordusunda görev yapan Atatürk, Çanakkale Cephesi'nde miralaylığa, Sina ve Filistin Cephesi'nde ise Yıldırım Ordular Grubu komutanlığına atandı. Savaşın sonunda, Osmanlı İmparatorluğu'nun yenilgisini izleyen Kurtuluş Savaşı ile simgelenen Anadolu Hareketi'ne öncülük ve önderlik etti. Türk Kurtuluş Savaşı sürecinde Ankara Hükûmeti'ni kurdu, Türk Orduları Başkomutanı olarak Sakarya Meydan Muharebesi'ndeki başarısından dolayı 19 Eylül 1921 tarihinde 'gazi' sanını aldı ve mareşallik rütbesine yükseldi. Askerî ve siyasal eylemleriyle İtilaf Devletleri ve destekçilerine karşı yengi kazandı. Savaşın ardından Cumhuriyet Halk Partisini 'Halk Fırkası' adıyla kurdu ve ilk genel başkanı oldu. 29 Ekim 1923'te Cumhuriyetin İlanı ardından cumhurbaşkanı seçildi. 1938'deki ölümüne dek dört dönem bu görevi yürütmüş olup günümüze değin Türkiye'de en uzun süre cumhurbaşkanlığı yapmış kişidir.",
    "Atatürk; çağdaş, ilerici ve laik bir ulus devlet kurmak için siyasal, ekonomik ve kültürel alanlarda sekülarist ve milliyetçi nitelikte yenilikler gerçekleştirdi. Yabancılara tanınan ekonomik ayrıcalıklar kaldırıldı ve onlara ait üretim araçları ve demir yolları millîleştirildi. Tevhîd-i Tedrîsât Kanunu ile eğitim, Türk hükûmetinin denetimine girdi. Seküler ve bilimsel eğitim esas alındı. Binlerce yeni okul yapıldı. İlköğretim ücretsiz ve zorunlu duruma getirildi. Yabancı okullar devlet denetimine alındı. Köylülerin sırtına yüklenen ağır vergiler azaltıldı. Erkeklerin serpuşlarında ve giysilerinde bazı değişiklikler yapıldı. Takvim, saat ve ölçülerde değişikliklere gidildi. Mecelle kaldırılarak yerine seküler Türk Kanunu Medenisi yürürlüğe konuldu. Kadınların sivil ve siyasal hakları pek çok Batı ülkesinden önce tanındı. Çok eşlilik yasaklandı. Kadınların tanıklığı ve miras hakkı, erkeklerinkiyle eşit duruma getirildi. Benzer olarak, dünyanın çoğu ülkesinden önce olarak Türkiye'de kadınlara ilkin yerel seçimlerde (1930), sonra genel seçimlerde (1934) seçme ve seçilme hakkı tanındı. Ceza ve borçlar hukukunda seküler yasalar yürürlüğe konuldu. Sanayi Teşvik Kanunu kabul edildi. Toprak reformu için çabalandı. Arap harfleri temelli Osmanlı alfabesinin yerine Latin harfleri temelli yeni Türk alfabesi kabul edildi. Halkı okuryazar kılmak için eğitim seferberliği başlatıldı. Üniversite Reformu gerçekleştirildi. Birinci Beş Yıllık Sanayi Planı yürürlüğe konuldu. Sınıf ve durum ayrımı gözeten lakap ve unvanlar kaldırıldı ve soyadları yürürlüğe konuldu. Bağdaşık ve birleşmiş bir ulus yaratılması için Türkleştirme siyaseti yürütüldü.",
    "Atatürk, Mustafa Kemal adını askeriyede faaliyet gösterdiği yıllar içindeki hizmeti ve başarılarından dolayı hak ettiği Bey (1911), Paşa (1916) ve Gazi (1921) unvanlarıyla birlikte kullandı ve 1934'e dek sıkça 'Gazi' unvanıyla anıldı. Mustafa Kemal'e 21 Haziran 1934 tarih ve 2525 sayılı Soyadı Kanunu'nun kabulünden sonra TBMM tarafından çıkarılan 24 Kasım 1934 tarih ve 2587 sayılı Kemal öz adlı Cümhur Reisimize verilen soyadı hakkında kanun ile Atatürk soyadı verildi. Yine aynı kanuna göre 'Atatürk' soyadı veya öz adı başka kimse tarafından alınamaz, kullanılamaz."
    "“Türk Genci, devrimlerin ve cumhuriyetin sahibi ve bekçisidir. Bunların gereğine, doğruluğuna herkesten çok inanmıştır. Yönetim biçimini ve devrimleri benimsemiştir. Bunları güçsüz düşürecek en küçük ya da en büyük bir kıpırtı ve bir davranış duydu mu, “Bu ülkenin polisi vardır, jandarması vardır, ordusu vardır, adalet örgütü vardır” demeyecektir. Elle, taşla, sopa ve silahla; nesi varsa onunla kendi yapıtını koruyacaktır.",
    "Polis gelecek, asıl suçluları bırakıp, suçlu diye onu yakalayacaktır. Genç, “Polis henüz devrim ve cumhuriyetin polisi değildir” diye düşünecek, ama hiç bir zaman yalvarmayacaktır. Mahkeme onu yargılayacaktır. Yine düşünecek, “demek adalet örgütünü de düzeltmek, yönetim biçimine göre düzenlemek gerek”",
    "Onu hapse atacaklar. Yasal yollarla karşı çıkışlarda bulunmakla birlikte bana, başbakana ve meclise telgraflar yağdırıp, haksız ve suçsuz olduğu için salıverilmesine çalışılmasını, kayrılmasını istemeyecek. Diyecek ki, “ben inanç ve kanaatimin gereğini yaptım. Araya girişimde ve eylemimde haklıyım. Eğer buraya haksız olarak gelmişsem, bu haksızlığı ortaya koyan neden ve etkenleri düzeltmek de benim görevimdir.”",
    "uzayın derinliklerinde bulunan ve olağanüstü çekim gücüyle çekim alanına giren her şeyi kendine çekip yok eden, herhangi bir biçimi olmayan, varsayımsal gökcismi.",
    "Güneş'ten çok daha büyük kütleli bir yıldızın bir süpernova ile patlaması ve dış katmanlarını uzaya püskürten yıldızın bir yandan da kendi üzerine çökmesi sonucu bu tip kara delikler oluşur.",
    "Kara delikler, evrendeki yer çekiminin çok güçlü olduğu ve bu sayede etrafındaki zaman ve uzayı bozan yerlere denir. Bir kara deliğin içine girdikten sonra hiçbir şey, hatta ışık bile dışarı çıkamaz.",
    "Bir kara deliğin olay ufku, geri dönüşü olmayan noktadır. Bu noktadan geçen her şey, kara delik tarafından yutulacak ve bilinen evrenimizden sonsuza dek yok olacaktır. Olay ufkunda, kara deliğin yerçekimi o kadar güçlüdür ki, hiçbir mekanik kuvvet onun üstesinden gelemez veya ona karşı koyamaz.",
    "Olay ufkunun radyal boyutu, ilgili kara deliğin kütlesine bağlıdır ve bir kişinin bir karadeliğe düşerek hayatta kalması için anahtar öneme sahiptir. Güneşimizin kütlesine (1 Güneş kütlesi) sahip bir kara delik için olay ufkunun yarıçapı, 3.2 kilometrenin biraz altında olacaktır. Buna karşılık, Samanyolu galaksimizin merkezindeki süper kütleli kara delik, kabaca 4 milyon güneş kütlesine sahiptir ve 11.7 milyon kilometre veya 17 Güneş yarıçapına eşit bir olay ufkuna sahiptir. Bu nedenle, yıldız boyutunda bir kara deliğe düşen biri, süper kütleli bir kara deliğe düşmenin aksine, olay ufkunu geçmeden önce kara deliğin merkezine çok daha fazla yaklaşacaktır.",
    "Bu, kara deliğin merkezinin yakınlığından ötürü kara deliğin bir kişi üzerindeki kütleçekim kuvvetinin, serbest düşüşe bağlı olarak başınız ile ayak parmağınız arasında 1 trilyon kat fark olacağı anlamına gelir. Bu tür kuvvetlere aynı zamanda gelgit kuvvetleri denir. Başka bir deyişle kişi, yıldız kütleli bir kara deliğin olay ufkuna ayakları önden gidecek şekilde düşüyorsa, ayaklarındaki çekim kuvveti, kafasına uygulanan kütleçekime kıyasla kat kat daha büyük olacaktır.",
    "A vitamini: A vitamini biyolojik aktivitesine sahip hayvansal kaynaklı bileşiklerin tümüne verilen isimdir. Ana işlevlerini, retinol ile onun iki türevi olan retinal ve retinoik asit gerçekleştirir. Sıcağa ve alkaliye dayanıklı; aside, oksidasyona ve ultraviyole ışınlara duyarlıdır. Karaciğer, süt, yumurta sarısı ve meyveler gibi gıdalarda bulunmaktadır.",
    "A posteriori: Deneyime dayalı anlamına gelmektedir. Doğru önermeler deneye ve duyu verilerine dayanan önermesel bilgilerdir. “Güneş doğudan doğar” veya “Dünya yuvarlaktır” gibi önermeler, bu bilgi sınıflandırmasına örnek olarak verilebilir. Algılarımız ve tümevarım yoluyla edindiğimiz bilgilerimizin büyük bir kısmını 'a posteriori' bilgiler oluşturmaktadır. Immanuel Kant, matematik gibi zihinsel süreçlerin 'a priori', Dünya'nın varlığı ve durumu ile ilgili olanları 'a posteriori' olarak kabul etmeyi önermiştir. Bilgiye dair en temel tartışmalardan birisi olan 'a priori' ve 'a posteriori' ayrımı hâlâ çağdaş epistemolojinin süregelen konularından birisidir.",
    "Abell Kataloğu: 1958 yılında Amerikalı astronom George Ogden Abell tarafından yayınlanan ve 2712 gökada kümesini içeren bir gökadalar kataloğudur. Bir kümenin bu kataloğa eklenmesi için; içerisinde en az 50 gökada bulundurması ve Abell yarıçapı olarak adlandırılan bölge içerisinde bulunacak kadar düzenli olmaları gibi belirli kriterleri karşılaması gerekmektedir. Daha tutarlı bir hata payı elde edebilmek için bu kriterler daima uygulanmamıştır. Örneğin, kataloğa yapılan son eklemelerde üye sayısı 50'nin altında pek çok küme bulunmaktadır.",
    "Abiyoz: 'Yaşamın noksanlığı ve yokluğu' anlamına gelen bir terimdir.",
    "Abiyotik: 'Cansızlığa özgü' ve 'cansızlığa ait' anlamına gelen bir terimdir.",
    "Adalet psikolojisi: Yasaların yapım ve uygulanmasının suç davranışıyla ilgisini araştıran psikoloji dalıdır.",
    "Adaptasyon: Genetik dağılım üzerine uzun süreli etki eden doğal seçilim sonucunda, türlerin çevresel ihtiyaçlara uygun özellikler kazanması veya var olan özelliklerin değişimidir. Örneğin, yassı balıkların atalarında, gözler çift taraflı simetriye uygun olarak gözün iki yanında bulunmaktayken balıkların nesiller boyunca okyanus tabanlarında ve yatay biçimde yaşamaya adapte olmaları nedeniyle gözlerden tabana bakan, vücudun diğer tarafında okyanusun içine bakan gözün yanına doğru kaymıştır. Yassı balıklar, okyanus tabanında yaşadıkları için sadece yukarıdan gelebilecek saldırılara karşı bu şekilde bir adaptasyon geçirmişlerdir.",
    "Adaptif bağışıklık: Lenf hücrelerinin antijenlere belirli ve uzun süreli tepkilerini anlatmak için kullanılan genel bir terimdir. Majör histokompatibilite kompleksi, T-hücresi alıcıları (TCR), immunoglobulinlerle birlikte rekombinaz aktivitesine sahip enzimlere ihtiyaç duyar. Çenesiz balıklar haricindeki tüm omurgalılarda bulunmaktadır.",
    "Adaptif bağışıklık sistemi: Omurgalılarda patojenlere karşı oldukça spesifik ve uzun süreli savunma sağlayan lenfosit sistemidir. İki ana lenfosit sınıfından oluşur: Patojene veya patojen kaynaklı moleküllere spesifik olarak bağlanan antikorları salgılayan B lenfositleri (B hücreleri) ve patojen tarafından enfekte edilmiş hücreleri doğrudan öldürebilen veya patojeni ortadan kaldırabilecek diğer hücreleri uyaran sinyal proteinleri (bunlar hücre yüzeyi proteinleri veya hücre dışına salgılanan proteinler olabilir) üreten T lenfositleri (T hücreleri).",
    "Adaptör protein: Temel görevi, iki veya daha fazla sayıda proteini bir hücre içi sinyal yolağında veya protein kompleksinde birbirine bağlamak olan proteinlerin genel adı.",
    "Addüktör kas: Kol ya da bacağın, vücudun orta düşey eksenine doğru yer değiştirmesini sağlayan kaslara verilen isimdir. 'Yakınlaştırıcı kas' olarak da isimlendirilmektedir.",
    "Adenom: Kanserli olmayan, iyi huylu ve yavaş büyüyen tümörlerdir. Bezsel kökenlidirler. Sıklıkla glandüler organlar boyunca büyürler. Adrenal adenomlar, kolon polipleri, paratirod adenomları, hipofiz adenomları ve pleomorfik adenomlar olmak üzere çeşitli türleri mevcuttur.",
    "Adezyon: Birbirinden farklı yüzeylerin birbirine yapışma eğilimi. Bazı farklar dolayısıyla kimyasal adezyon, dağıtıcı adezyon, elektrostatik adezyon ve difüzyon adezyon gibi kategorilere ayrılabilir.",
    "Adipoz doku: Yağı depolayan dokudur. İki tip adipoz doku bulunmaktadır: beyaz adipoz doku (uniloküler) ve kahverengi adipoz doku (multiloküler).",
    "Aerob: Oksijenin varlığında üreyebilen ve yaşamını sürdüren organizmalardır.",
    "hackleme öğürme ve tükürme kısmı değil lütfen",
    "tamam  o zaman biraz fransız mutfağını denemeye ne dersiniz  cumartesi  gece",
    "mesele şu ki cameron  özellikle korkunç bir ezik türünün insafına kaldım kız kardeşim  o çıkana kadar ben çıkamam",
    "kolayca bir randevu bulabilir gibi görünüyor",
    "tanrım kate bir erkek arkadaş bulabilsek",
    "ne yapabileceğime bir bakayım",
    "küçük fahişeyi bul bir randevu planımız nasıl ilerliyor",
    "pekala olabileceğini düşündüğüm biri var",
    "demek sevdiği erkek tipi bu güzeller mi",
    "kim bilir onun tüm söylediğini sigara içen bir erkekle çıkmadan önce içki içtiğiydi",
    "tanrıya şükür saç stiliniz hakkında bir hikaye daha duymak zorunda kalsaydım",
    "yirmi dakika içinde evde olmam gerekiyor",
    "i̇kiye kadar evde olmam gerekmiyor",
    "dinle seninle balo hakkında konuşmak istiyorum",
    "anlaşmayı biliyorsun kat gitmezse ben gidemem",
    "eğer yolumdan çekilmezsen seni mahvedecek potansiyele sahibim",
    "balo mu katin bir randevusu mu var",
    "o bir  sıcak çubuk  değil her ne ise",
    "şaka yapıyordum i̇nsanlar gerçekten orada mı yaşıyor",
    "evet bir çift  yine de inekler bizi sayıca geride bıraktı",
    "neden böyle kızlar hep böyle erkeklerden hoşlanır",
    "çünkü onlar için yetiştirildiler anneleri böyle adamları severdi onlardan önce de büyükanneleri gen havuzları nadiren seyreltilir",
    "bilin bakalım kim bir öğretmene kaydoldu",
    "evet fahişeyle küçük bir karşılaşma",
    "şaka değil o bir suçlu bir eyalet polisini ateşe verdiğini duydum alcatrazdan yeni çıktı",
    "her zaman suçluların honors biologyde yer almasına izin verirler mi",
    "ciddiyim dostum bayıldı yeni hoparlörler alabilmek için karaborsada kendi karaciğerini sattı"



]


dataset = SimpleDataset(sentences, simple_tokenizer, max_length=20)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

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
train(model, loader, loss_fn, optimizer, device, dataset.vocab, epochs=450)

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
test_sentence = "Atatürk’ümüzün kadın hakkındaki anlayışını "
predicted_tokens = predict(model, test_sentence, dataset.vocab, max_length=10, device=device)
print(f"Predicted continuation: {predicted_tokens}")
