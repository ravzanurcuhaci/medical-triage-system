# Medical Triage System (Semptom Tabanlı Karar Destek Sistemi)

Bu proje, kullanıcıların yazdığı semptomlara göre olası hastalıkları tahmin eden bir yapay zeka sistemidir. Sistem, sınıflandırma (classification) ve benzer vaka bulma (retrieval) yöntemlerini birlikte kullanır.

## 🚀 Özellikler

* Türkçe veya İngilizce semptom girişi
* Otomatik çeviri (Türkçe ↔ İngilizce)
* TF-IDF + Logistic Regression ile sınıflandırma
* Sentence Transformers ile benzer vaka bulma
* En olası hastalıkların (Top-k) listelenmesi
* Benzer geçmiş vakaların gösterilmesi

## ⚙️ Nasıl Kullanılır?

1. Notebook’u Google Colab üzerinde açın
2. Tüm hücreleri çalıştırın (Run All)
3. Semptomlarınızı yazın
4. Sistem size:

   * Olası hastalıkları
   * Benzer vakaları
   * Kısa bir yorum çıktısı verecektir

## 🧪 Örnek Girdi

"I have had cough, sore throat and fever for 3 days"

## 📊 Çıktı

* Olası hastalıklar (olasılık skorları ile)
* Benzer semptomlara sahip vakalar
* Model yorumu

## ⚠️ Uyarı

Bu sistem yalnızca semptomlara dayalı bir karar destek sistemidir.
Tıbbi teşhis koymaz ve doktor yerine geçmez.

## 🎓 Amaç

Bu proje, yapay zeka tekniklerinin sağlık alanında nasıl kullanılabileceğini göstermek amacıyla geliştirilmiştir.
