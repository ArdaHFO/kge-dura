KGE-DURA, birbirine bağlı bilgiler arasındaki eksik bağlantıları tahmin eden bir araştırma yazılımıdır.

<h1 align="center">KGE-DURA</h1>

<p align="center"><strong>Bilgi grafı tamamlama için düzenlileştirme.</strong></p>
<p align="center">Tensör çarpanlarına ayırmaya dayalı bilgi grafı gömmelerinde dualite kaynaklı düzenlileştirme için araştırma kodu.</p>

<p align="center">
  <a href="README.md">English</a> · <strong>Türkçe</strong><br />
  <a href="https://arxiv.org/abs/2011.05816"><strong>Makaleyi oku</strong></a> ·
  <a href="docs/REPRODUCING.md">Deney komutları</a> ·
  <a href="#atıf">Çalışmaya atıf yap</a>
</p>

---

Bu repo, **Zhanqiu Zhang, Jianyu Cai ve Jie Wang** tarafından yazılan ve **NeurIPS 2020** kapsamında yayımlanan **Duality-Induced Regularizer for Tensor Factorization Based Knowledge Graph Completion** makalesinin araştırma uygulamasını içerir. CP, ComplEx ve RESCAL modelleri, DURA düzenlileştiricileri, veri ön işleme kodu ve özgün deney komutları birlikte sunulur.

Bilimsel katkının atfı özgün yazarlara ve makaleye aittir. İki dilli sunum yeni yazarlık, yeniden üretilmiş sonuçlar veya modernleştirilmiş bir uygulama iddiası taşımaz. [arXiv makalesi](https://arxiv.org/abs/2011.05816) · [NeurIPS bildirisi](https://proceedings.neurips.cc/paper/2020/hash/f6185f0ef02dcaec414a3171cd01c697-Abstract.html)

## Deneylerle başla

| Gereksinim | Projenin belgelediği tarihsel ortam |
| --- | --- |
| Python | 3.6 veya sonrası |
| PyTorch | 1.0–1.7 |
| NumPy | 1.17.2 veya sonrası |
| tqdm | 4.41.1 veya sonrası |
| Hesaplama | CUDA destekli GPU ve buna uygun CUDA etkin PyTorch kurulumu |
| Veri | `src_data/` altında sağlanan WN18RR, FB15k-237 ve YAGO3-10 bölümleri |

Bunlar özgün bağımlılık gereksinimleridir, daha yeni bütün sürümlerin uyumlu olduğu iddia edilmez. Eğitim, değerlendirme ve bazı düzenlileştiriciler doğrudan CUDA kullanır. Yalnız CPU ile çalışma yolu sunulmaz. Sabitlenmiş bir requirements veya ortam dosyası bulunmaz.

```bash
git clone https://github.com/ArdaHFO/kge-dura.git
cd kge-dura
```

Tarihsel yazılım bileşenleri ve GPU'nla uyumlu, yalıtılmış bir ortam hazırla. Sonra sağlanan veriyi `code` dizininden ön işle.

```bash
cd code
python process_datasets.py
```

Betik, eşlenmiş üçlüleri, varlık/ilişki kimliklerini ve filtreli değerlendirme verilerini repo kökündeki `data/` dizinine yazar. Önceden bulunan veri kümesi çıktı dizinleri atlanır.

[Deney rehberinden](docs/REPRODUCING.md) model ve veri kümesi birleşimini seç. Rehber, dokuz özgün eğitim komutunu ve hiperparametrelerini korur. Komutları `code/` içinden çalıştır. FB15k-237 için komut satırında kullanılan ad `FB237` olur.

## Uygulamanın içinde

- **Üç tensör çarpanlarına ayırma modeli.** CP, ComplEx ve RESCAL, `models.py` içindeki bilgi grafı tamamlama arayüzünü paylaşır.
- **Modele özel düzenlileştirme.** DURA, ağırlıklı DURA ve RESCAL çeşitleri, mevcut karşılaştırma düzenlileştiricileriyle birlikte uygulanır.
- **Ters ilişkili eğitim üçlüleri.** Veri yükleyici, baş ve kuyruk varlık tahmini için üçlülere ters ilişkilerini ekler.
- **Filtreli sıralama.** Değerlendirme bilinen doğru üçlüleri filtreleyerek ortalama ters sıra ve Hits@1/3/10 değerlerini hesaplar.
- **Deney komutları.** Özgün ayarlar her modeli WN18RR, FB15k-237 ve YAGO3-10 üzerinde kapsar.
- **Dışa aktarılan çıktılar.** `-save` ile eğitim, `logs/` altında yapılandırma, günlük, checkpoint ve varlık/ilişki gömmelerini kaydeder.

## Raporlanan sonuçlar

Özgün repo, WN18RR, FB15k-237 ve YAGO3-10 için aşağıdaki sonuç görselini içerir. Görsel projenin raporladığı araştırma sonucu olarak korunur. Bu dokümantasyon güncellemesinde yeni bir deney çalıştırılmadı.

<p align="center">
  <img src="result.png" alt="WN18RR, FB15k-237 ve YAGO3-10 için özgün DURA sonuçları" />
</p>

Bilimsel kapsam ve yorum için makaleye başvur. Özgün görsel ve deney hiperparametreleri değiştirilmedi.

## Geliştirme ve doğrulama

Bağımlılıkları kurduktan sonra repo kökünden başlayarak komut satırı seçeneklerini inceleyebilirsin.

```bash
cd code
python learn.py --help
```

Repoda otomatik test paketi veya CI iş akışı yoktur. Python uygulaması ve özgün talimatlar statik olarak incelendi. Kaynak düzeltmeleri batch boyutlarını, filtreli sıralamada liste değişimini, veri boyutlarını ve komut satırı akışlarını kapsar. Bu değişiklikler için kod, test, ön işleme, eğitim veya performans deneyi çalıştırılmadı. Eğitim giriş noktası sabit bir rastgelelik tohumu belirlemiyor. Güncel Python/PyTorch sürümleriyle uyumluluk burada doğrulanmadı.

Sağlanan komutlar günlük, checkpoint ve gömmeleri kaydetmek için `-save` seçeneğini korur. Bu seçenek olmadan eğitim söz konusu çıktıları yazmaz. Bağımsız değerlendirme için eşleşen veri kümesi, model ve rank ile `-test --checkpoint <run-directory>` kullanılır. Akışı uyarlamadan önce [`learn.py`](code/learn.py) dosyasını incele. Düzeltilen bu yollar çalıştırılarak doğrulanmadı.

## Repo haritası

| Yol | Amaç |
| --- | --- |
| [`code/process_datasets.py`](code/process_datasets.py) | Veri bölümlerini ve filtreleme yapılarını hazırlama |
| [`code/datasets.py`](code/datasets.py) | Üçlüleri yükleme, ters ilişkileri ekleme ve sıralama değerlendirmesi |
| [`code/models.py`](code/models.py) | CP, ComplEx ve RESCAL uygulamaları |
| [`code/regularizers.py`](code/regularizers.py) | DURA ve diğer düzenlileştiriciler |
| [`code/optimizers.py`](code/optimizers.py) | Minibatch optimizasyon döngüsü |
| [`code/learn.py`](code/learn.py) | Eğitim seçenekleri, değerlendirme ve çıktı kaydı |
| [`src_data/`](src_data/) | Sağlanan veri bölümleri ve bildirimleri |
| [`docs/REPRODUCING.md`](docs/REPRODUCING.md) | Korunan özgün deney komutları |
| [`result.png`](result.png) | Özgün raporlanmış sonuç görseli |

## Katkıda bulunma

Tekrarlanabilirlik raporlarına commit'i, ortamı, CUDA/PyTorch sürümlerini, veri kümesini, tam komutu ve gözlenen ölçümleri ekle. Rastgelelik tohumu, ön işleme veya hiperparametre değişikliklerini açıkla. Özgün komutları koru ve yeni deney sonuçlarını makalenin raporladığı sonuçlardan ayır. Dokümantasyon ve taşınabilirlik iyileştirmelerinde yazar atıflarını ve veri kümesi bildirimlerini koru.

## Atıf

Bu kod araştırmanda yararlı olduysa özgün makaleye atıf yap.

```
@inproceedings{NEURIPS2020_f6185f0e,
 author = {Zhang, Zhanqiu and Cai, Jianyu and Wang, Jie},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {21604--21615},
 publisher = {Curran Associates, Inc.},
 title = {Duality-Induced Regularizer for Tensor Factorization Based Knowledge Graph Completion},
 url = {https://proceedings.neurips.cc/paper/2020/file/f6185f0ef02dcaec414a3171cd01c697-Paper.pdf},
 volume = {33},
 year = {2020}
}
```

## Lisans ve katkılar

Özgün uygulama [facebookresearch/kbc](https://github.com/facebookresearch/kbc) koduna teşekkür eder. Yazarların ilişkili çalışması **Learning Hierarchy-Aware Knowledge Graph Embeddings for Link Prediction**, Zhanqiu Zhang, Jianyu Cai, Yongdong Zhang ve Jie Wang tarafından AAAI 2020 kapsamında yayımlanmıştır. [Makale](https://arxiv.org/pdf/1911.09419.pdf) · [KGE-HAKE kodu](https://github.com/MIRALab-USTC/KGE-HAKE)

Repoda proje düzeyinde yazılım lisansı dosyası bulunmuyor. Veri kümesi bildirimleri [FB237](src_data/FB237/license.docx), [WN18RR](src_data/WN18RR/license.txt) ve [YAGO3-10](src_data/YAGO3-10/license.txt) altında ayrı korunur. WN18RR bildirimi bu ağaçta bulunmayan `WN/license.txt` dosyasına yönlendirir. YAGO3-10 bildirimi kaynak dökümler için CC-BY belirtir. Bu bildirimler kod tabanının tamamına bir lisans atamaz. Bu dokümantasyon yeni bir lisans oluşturmaz.
