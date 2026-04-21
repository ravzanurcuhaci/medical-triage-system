---
title: Medical Triage Api
emoji: ⚕️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# 🏥 Medical Triage System — Yapay Zeka Destekli Tıbbi Triyaj Sistemi

Kullanıcının Türkçe yazdığı semptomları analiz ederek **olası hastalığı**, **ilgili poliklinik alanını** ve **aciliyet seviyesini** yapay zeka ile tahmin eden uçtan uca (end-to-end) bir tıbbi karar destek sistemidir.

> ⚠️ **Uyarı:** Bu sistem yalnızca semptomlara dayalı bir karar destek aracıdır. Tıbbi teşhis koymaz ve doktor yerine geçmez.

---

## ✨ Özellikler

| Özellik | Açıklama |
|---|---|
| 🇹🇷 Türkçe Girdi | Kullanıcı semptomlarını doğrudan Türkçe yazar |
| 🌐 Otomatik Çeviri | Türkçe metin Helsinki-NLP modeli ile İngilizceye çevrilir |
| 🧠 BERT Tahmini | Fine-tune edilmiş BERT modeli ile hastalık sınıflandırması |
| 🤖 RoBERTa Tahmini | Fine-tune edilmiş RoBERTa modeli ile hastalık sınıflandırması |
| ⚖️ Ensemble (Ortak Karar) | İki modelin olasılıklarının ortalaması alınarak daha güvenilir tahmin |
| 🏥 Poliklinik Yönlendirmesi | Tahmin edilen hastalığa göre ilgili tıbbi uzmanlık alanı önerisi |
| 🚦 Aciliyet Seviyesi (Triyaj) | 🔴 Acil / 🟡 Orta / 🟢 Normal olarak triyaj belirleme |
| 🗣️ Çok Dilli Çıktı | Hastalık isimleri hem İngilizce hem Türkçe olarak döner |
| 📝 Semptom Özeti | Girilen metinden öne çıkan semptomların kısa özeti |

---

## 🏗️ Mimari

```
Kullanıcı (Türkçe Metin)
        │
        ▼
   ┌─────────────┐
   │   FastAPI    │  ← POST /predict
   │   Backend    │
   └──────┬──────┘
          │
    ┌─────┼──────────────────┐
    ▼     ▼                  ▼
 Çeviri  BERT Model    RoBERTa Model
 (TR→EN) (Top-3)        (Top-3)
    │     │                  │
    │     └──────┬───────────┘
    │            ▼
    │     Ensemble (Ortalama)
    │            │
    ▼            ▼
┌────────────────────────────────┐
│         API Yanıtı             │
│  • bert_top3                   │
│  • roberta_top3                │
│  • final_predictions           │
│  • Poliklinik + Aciliyet       │
└────────────────────────────────┘
```

---

## 📂 Proje Yapısı

```
medical-triage-system/
├── Dockerfile                  # HF Spaces Docker yapılandırması
├── .dockerignore               # Docker'a dahil edilmeyecek dosyalar
├── README.md                   # Bu dosya
└── backend/
    ├── requirements.txt        # Python bağımlılıkları
    ├── upload.py               # Modelleri HF Hub'a yükleyen betik
    ├── .gitignore              # Git'e dahil edilmeyecek dosyalar
    ├── .env                    # Ortam değişkenleri (Git'e dahil değil)
    └── app/
        ├── main.py             # FastAPI uygulama giriş noktası + CORS
        ├── model_service.py    # Model yükleme, çeviri ve tahmin servisi
        ├── schemas.py          # Pydantic istek/yanıt şemaları
        ├── department_mapping.py  # 46 hastalık → Poliklinik + Aciliyet eşleştirmesi
        └── utils.py            # Metin temizleme ve semptom özeti
```

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler

- Python 3.12+
- pip

### Yerel Kurulum

```bash
# 1. Depoyu klonla
git clone https://github.com/ravzanurcuhaci/medical-triage-system.git
cd medical-triage-system/backend

# 2. Sanal ortam oluştur ve aktif et
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Bağımlılıkları kur
pip install -r requirements.txt

# 4. Sunucuyu başlat
uvicorn app.main:app --reload
```

Sunucu varsayılan olarak `http://127.0.0.1:8000` adresinde çalışır.

---

## 🌍 Dışarıya Açma (LocalTunnel)

Arkadaşlarınızın veya ekibinizin API'ye uzaktan erişmesi için LocalTunnel kullanabilirsiniz.

```bash
# İlk terminal: Sunucuyu çalıştır
source venv/bin/activate
uvicorn app.main:app --reload

# İkinci terminal: Tüneli aç
npx localtunnel --port 8000
```

Ekrana çıkan `https://xxxx-xxxx-xxx.loca.lt` linkinin sonuna `/predict` ekleyerek paylaşın:

```
https://xxxx-xxxx-xxx.loca.lt/predict
```

> **Not:** LocalTunnel geçici bir bağlantıdır. Terminali kapattığınızda link devre dışı kalır.

---

## ☁️ Canlı Yayın (Hugging Face Spaces)

Bu proje **Hugging Face Spaces** üzerinde Docker ile 7/24 çalışmaktadır. Modeller Hugging Face Hub'da ayrı depolarda barındırılır ve uygulama başlatılırken otomatik olarak indirilir.

**Canlı API Adresi:**
```
https://ravzanurcuhaci-medical-triage-api.hf.space/predict
```

### Nasıl Çalışıyor?

1. `Dockerfile` ile container oluşturulur.
2. Container ayağa kalkarken `model_service.py` devreye girer.
3. BERT ve RoBERTa modelleri Hugging Face Hub'dan (`ravzanurcuhaci/bert_symptom_model`, `ravzanurcuhaci/roberta_model`) dinamik olarak indirilir ve önbelleğe alınır.
4. FastAPI sunucusu 7860 portunda başlar.

### Kendi Space'inize Deploy Etmek

```bash
# 1. Modelleri HF Hub'a yükleyin (Google Colab'da hızlıdır)
python upload.py

# 2. Değişiklikleri commit edin
git add .
git commit -m "Deploy: HF Spaces Docker"

# 3. Space remote'unu ekleyin
git remote add space https://huggingface.co/spaces/<kullaniciadi>/<space-adi>

# 4. Push edin
git push space main
```

---

## 📡 API Kullanımı

### `POST /predict`

Türkçe semptom metni gönderip tahmin almak için kullanılır.

**İstek (Request Body):**
```json
{
    "text_tr": "İdrar yaparken yanma var ve çok sık tuvalete gidiyorum."
}
```

**Yanıt (Response):**
```json
{
    "turkish_input": "İdrar yaparken yanma var ve çok sık tuvalete gidiyorum.",
    "short_symptom_summary": "Girdide öne çıkan semptomlar: sık idrara çıkma.",
    "english_translation": "There's burning while urinating, and I go to the bathroom a lot.",
    "bert_top3": [
        {
            "label": "UTI",
            "label_tr": "İdrar Yolu Enfeksiyonu",
            "score": 66.3,
            "department": "Üroloji",
            "urgency": "🟡 Orta (Yakın Zamanda Doktora Görünün)"
        }
    ],
    "roberta_top3": [
        {
            "label": "UTI",
            "label_tr": "İdrar Yolu Enfeksiyonu",
            "score": 80.98,
            "department": "Üroloji",
            "urgency": "🟡 Orta (Yakın Zamanda Doktora Görünün)"
        }
    ],
    "final_predictions": [
        {
            "label": "UTI",
            "label_tr": "İdrar Yolu Enfeksiyonu",
            "score": 73.64,
            "department": "Üroloji",
            "urgency": "🟡 Orta (Yakın Zamanda Doktora Görünün)"
        }
    ]
}
```

### `GET /`

Sunucunun çalışıp çalışmadığını kontrol etmek için kullanılır.

```json
{
    "message": "Welcome to Medical Triage AI API! System is UP and Running. 🚀"
}
```

---

## 🧪 Test

### Postman ile Test

1. Postman uygulamasını açın.
2. Yeni bir **POST** isteği oluşturun.
3. URL olarak yerel (`http://127.0.0.1:8000/predict`) veya canlı (`https://ravzanurcuhaci-medical-triage-api.hf.space/predict`) adresi girin.
4. **Body** sekmesinde **raw** ve **JSON** seçin.
5. Aşağıdaki JSON'u yapıştırıp **Send** butonuna basın:

```json
{
    "text_tr": "3 gündür öksürüğüm var, ateşim çıktı ve boğazım ağrıyor."
}
```

### cURL ile Test

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text_tr": "Başım çok ağrıyor ve midem bulanıyor."}'
```

---

## 🤖 Kullanılan Modeller

| Model | Kaynak | Açıklama |
|---|---|---|
| BERT (Fine-tuned) | [ravzanurcuhaci/bert_symptom_model](https://huggingface.co/ravzanurcuhaci/bert_symptom_model) | 46 hastalık sınıfı üzerinde eğitilmiş |
| RoBERTa (Fine-tuned) | [ravzanurcuhaci/roberta_model](https://huggingface.co/ravzanurcuhaci/roberta_model) | 46 hastalık sınıfı üzerinde eğitilmiş |
| Helsinki-NLP | [Helsinki-NLP/opus-mt-tc-big-tr-en](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-tr-en) | Türkçe → İngilizce çeviri |

---

## 🛠️ Teknoloji Yığını

- **Backend:** FastAPI + Uvicorn
- **Derin Öğrenme:** PyTorch + Hugging Face Transformers
- **Çeviri:** Helsinki-NLP (MarianMT)
- **Model Barındırma:** Hugging Face Hub
- **Dağıtım:** Hugging Face Spaces (Docker)
- **Geliştirici Araçları:** Postman, LocalTunnel, Google Colab

---

## 🩺 Desteklenen Hastalıklar (46 Sınıf)

<details>
<summary>Tüm listeyi görmek için tıklayın</summary>

| # | İngilizce | Türkçe | Poliklinik | Aciliyet |
|---|-----------|--------|------------|----------|
| 1 | Acne | Akne (Sivilce) | Dermatoloji | 🟢 Normal |
| 2 | Acute Bronchitis | Akut Bronşit | Göğüs Hastalıkları | 🟡 Orta |
| 3 | Allergic Rhinitis | Alerjik Rinit | KBB / Alerji | 🟢 Normal |
| 4 | Anemia | Anemi (Kansızlık) | Dahiliye / Hematoloji | 🟡 Orta |
| 5 | Angina | Anjina (Göğüs Ağrısı) | Kardiyoloji | 🔴 Acil |
| 6 | Appendicitis | Apendisit | Genel Cerrahi | 🔴 Acil |
| 7 | Arthritis | Artrit (Eklem İltihabı) | Romatoloji | 🟢 Normal |
| 8 | Bronchial Asthma | Astım | Göğüs Hastalıkları | 🟡 Orta |
| 9 | COPD Exacerbation | KOAH Alevlenmesi | Göğüs Hastalıkları | 🔴 Acil |
| 10 | Cellulitis | Selülit (Deri Enfeksiyonu) | Dermatoloji | 🟡 Orta |
| 11 | Cervical spondylosis | Boyun Kireçlenmesi | Beyin Cerrahisi | 🟢 Normal |
| 12 | Chicken pox | Suçiçeği | Enfeksiyon | 🟡 Orta |
| 13 | Common Cold | Soğuk Algınlığı | KBB / Dahiliye | 🟢 Normal |
| 14 | Dehydration | Dehidrasyon | Dahiliye | 🟡 Orta |
| 15 | Dengue | Dang Humması | Enfeksiyon | 🔴 Acil |
| 16 | Diabetes | Diyabet (Şeker) | Endokrinoloji | 🟡 Orta |
| 17 | Dimorphic Hemorrhoids | Hemoroit (Basur) | Genel Cerrahi | 🟢 Normal |
| 18 | Drug Reaction | İlaç Reaksiyonu | Alerji | 🔴 Acil |
| 19 | Fungal Infection | Mantar Enfeksiyonu | Dermatoloji | 🟢 Normal |
| 20 | GERD | Reflü | Gastroenteroloji | 🟢 Normal |
| 21 | Gastritis | Gastrit | Gastroenteroloji | 🟢 Normal |
| 22 | Gastroenteritis | Gastroenterit | Enfeksiyon / Dahiliye | 🟡 Orta |
| 23 | Hypertension | Hipertansiyon | Kardiyoloji | 🟡 Orta |
| 24 | IBS | İBS | Gastroenteroloji | 🟢 Normal |
| 25 | Impetigo | İmpetigo | Dermatoloji | 🟢 Normal |
| 26 | Jaundice | Sarılık | Gastroenteroloji | 🟡 Orta |
| 27 | Kidney Stone | Böbrek Taşı | Üroloji | 🟡 Orta |
| 28 | Malaria | Sıtma | Enfeksiyon | 🔴 Acil |
| 29 | Migraine | Migren | Nöroloji | 🟡 Orta |
| 30 | Palpitations | Çarpıntı | Kardiyoloji | 🟡 Orta |
| 31 | Panic Attack | Panik Atak | Psikiyatri | 🟡 Orta |
| 32 | Peptic Ulcer | Ülser | Gastroenteroloji | 🟡 Orta |
| 33 | Pneumonia | Zatürre | Göğüs Hastalıkları | 🔴 Acil |
| 34 | Psoriasis | Sedef Hastalığı | Dermatoloji | 🟢 Normal |
| 35 | Pyelonephritis | Böbrek İltihabı | Üroloji / Nefroloji | 🔴 Acil |
| 36 | Sinusitis | Sinüzit | KBB | 🟢 Normal |
| 37 | TIA-like | Mini İnme | Nöroloji | 🔴 Acil |
| 38 | Tension Headache | Gerilim Tipi Baş Ağrısı | Nöroloji | 🟢 Normal |
| 39 | Typhoid | Tifo | Enfeksiyon | 🔴 Acil |
| 40 | UTI | İdrar Yolu Enfeksiyonu | Üroloji | 🟡 Orta |
| 41 | Upper Respiratory Infection | Üst Solunum Yolu Enfeksiyonu | KBB / Dahiliye | 🟢 Normal |
| 42 | Urticaria | Ürtiker (Kurdeşen) | Dermatoloji | 🟡 Orta |
| 43 | Varicose Veins | Varis | Kalp Damar Cerrahisi | 🟢 Normal |
| 44 | Vertigo | Vertigo (Baş Dönmesi) | KBB / Nöroloji | 🟡 Orta |
| 45 | Viral Infection | Viral Enfeksiyon | Enfeksiyon / Dahiliye | 🟢 Normal |

</details>

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.

---

## 👩‍💻 Geliştirici

**Ravza Nur Cuhacı**
- GitHub: [@ravzanurcuhaci](https://github.com/ravzanurcuhaci)
- Hugging Face: [@ravzanurcuhaci](https://huggingface.co/ravzanurcuhaci)
