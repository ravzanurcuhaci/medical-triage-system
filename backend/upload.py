import os
from huggingface_hub import HfApi, whoami

def main():
    api = HfApi()
    try:
        user = whoami()["name"]
    except Exception:
        print("HATA: Hugging Face hesabına henüz giriş yapılmamış!")
        print("Lütfen terminalde 'hf auth login --force' komutunu çalıştırın ve yeni 'Write' şifrenizi yapıştırın.")
        return

    print(f"🚀 Hoş geldin {user}! Modeller senin Hugging Face hesabına yükleniyor...")
    
    try:
        # 1. BERT Modeli Repo Oluşturma ve Yükleme
        bert_repo = f"{user}/bert_symptom_model"
        print(f"📁 1/2: '{bert_repo}' deposu oluşturuluyor...")
        api.create_repo(repo_id=bert_repo, exist_ok=True, private=False)
        print("⏳ BERT ağırlıkları internete yükleniyor (Bu işlem dosya boyutundan dolayı 1-2 dakika sürebilir)...")
        api.upload_folder(
            folder_path="models/bert_symptom_model",
            repo_id=bert_repo,
        )
        print("✅ BERT yüklemesi tamamlandı!\n")

        # 2. RoBERTa Modeli Repo Oluşturma ve Yükleme
        roberta_repo = f"{user}/roberta_model"
        print(f"📁 2/2: '{roberta_repo}' deposu oluşturuluyor...")
        api.create_repo(repo_id=roberta_repo, exist_ok=True, private=False)
        print("⏳ RoBERTa ağırlıkları internete yükleniyor...")
        api.upload_folder(
            folder_path="models/roberta_model",
            repo_id=roberta_repo,
        )
        print("✅ RoBERTa yüklemesi tamamlandı!\n")
        
        # Ek olarak label_map de BERT repo una yükleyelim, çünkü lazım olabilir.
        if os.path.exists("models/label_map.csv"):
            api.upload_file(
                path_or_fileobj="models/label_map.csv",
                path_in_repo="label_map.csv",
                repo_id=bert_repo
            )

        print("-" * 50)
        print("🎉 MUHTEŞEM! Bütün modellerin başarıyla Hugging Face Hub'a aktarıldı!")
        print("Artık asistanına 'bitti' diye haber vererek bir sonraki aşamaya (Dockerfile) geçebilirsin.")
    except Exception as e:
        print(f"\n❌ Yükleme sırasında bir hata oluştu:\n{e}")
        print("Şifreni 'Write' yerine yanlışlıkla tekrar 'Read' almış olabilir misin? Kontrol et lütfen.")

if __name__ == "__main__":
    main()
