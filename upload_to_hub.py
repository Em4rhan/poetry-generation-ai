# upload_to_hub.py

from huggingface_hub import HfApi, upload_folder, create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# --- Yapılandırma ---
# Kullanıcı adın ve yüklemek istediğin modelin adı (Hugging Face Hub'da bu isimle bir repo oluşturulacak)
# Örnek: HF_USERNAME = "emirhan" (Hugging Face kullanıcı adın)
#        MODEL_HUB_NAME = "distilgpt2-poetry-finetuned" (Modelin Hub'daki adı)

HF_USERNAME = "Emirhan05"  # <<<--- BURAYI DOLDUR
MODEL_HUB_NAME = "distilgpt2-poetry-generation-ai"    # <<<--- İSTEDİĞİN BİR İSİM VER (benzersiz olmalı)

# Eğitilmiş modelinin bulunduğu klasörün yolu
# En son eğittiğin modelin klasör adını kullan (örn: fine_tuned_poetry_model_v3 veya v4)
LOCAL_MODEL_PATH = "./fine_tuned_poetry_model_v3" # <<<--- BURAYI KONTROL ET/GÜNCELLE

# Repoyu public mi private mı yapmak istediğin
IS_PRIVATE_REPO = False # True yaparsan sadece sen görürsün

# Commit mesajı
COMMIT_MESSAGE = "Upload fine-tuned poetry generation model and tokenizer"

def upload_model_and_tokenizer():
    # 1. Hugging Face API'sine bağlan
    api = HfApi()

    # 2. Model repo adını oluştur (kullanıcı_adı/model_adı)
    repo_id = f"{HF_USERNAME}/{MODEL_HUB_NAME}"
    print(f"Hugging Face Hub'a yüklenecek repo ID: {repo_id}")

    # 3. Repoyu oluştur (eğer zaten yoksa)
    try:
        create_repo(repo_id, private=IS_PRIVATE_REPO, exist_ok=True) # exist_ok=True zaten varsa hata vermez
        print(f"'{repo_id}' reposu başarıyla oluşturuldu veya zaten mevcut.")
    except Exception as e:
        print(f"HATA: Repo oluşturulurken bir sorun oluştu: {e}")
        print("Lütfen Hugging Face'e giriş yaptığınızdan ve kullanıcı adınızın doğru olduğundan emin olun.")
        return

    # 4. Model ve tokenizer dosyalarını yükle
    # `upload_folder` fonksiyonu, belirtilen klasördeki tüm dosyaları repo'ya yükler.
    # Bu, config.json, model.safetensors (veya pytorch_model.bin), tokenizer.json vb. dosyaları içerir.
    try:
        print(f"'{LOCAL_MODEL_PATH}' klasöründeki dosyalar '{repo_id}' reposuna yükleniyor...")
        
        # Önce model ve tokenizer'ı yükleyelim ki doğru şekilde yüklendiklerinden emin olalım
        # Bu adım aslında gerekmeyebilir eğer `LOCAL_MODEL_PATH` zaten doğru dosyaları içeriyorsa,
        # ama bazen Trainer farklı dosyalar kaydedebiliyor. Emin olmak için modeli ve tokenizer'ı
        # `from_pretrained` ile yükleyip `save_pretrained` ile geçici bir klasöre kaydedip
        # o klasörü yüklemek daha garantili olabilir.
        # Şimdilik direkt klasörü yüklemeyi deneyelim.

        # `ignore_patterns` ile yüklenmemesi gereken dosyaları belirtebilirsin
        # Örneğin, Trainer'ın oluşturduğu optimizer state gibi büyük dosyalar.
        # `fine_tuned_poetry_model_v3` klasöründe zaten sadece gerekli dosyalar olmalı.
        
        # Eğer `training_args.bin` gibi dosyaların gitmesini istemiyorsan:
        ignore_patterns = ["*.ckpt", "optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin", "rng_state.pth"]
        # "all_results.json", "eval_final_v3_results.json", "train_final_v3_results.json" da eklenebilir.
        # Şimdilik hepsini yollayalım, sonra Hub'dan silebilirsin.

        upload_folder(
            folder_path=LOCAL_MODEL_PATH,
            repo_id=repo_id,
            repo_type="model", # Bu bir model reposu
            commit_message=COMMIT_MESSAGE,
            # ignore_patterns=ignore_patterns # Opsiyonel
        )
        print(f"Dosyalar başarıyla '{repo_id}' reposuna yüklendi!")
        print(f"Modelinize şuradan erişebilirsiniz: https://huggingface.co/{repo_id}")

    except Exception as e:
        print(f"HATA: Dosyalar yüklenirken bir sorun oluştu: {e}")
        print(f"Lütfen '{LOCAL_MODEL_PATH}' klasörünün var olduğundan ve doğru dosyaları içerdiğinden emin olun.")

if __name__ == "__main__":
    if HF_USERNAME == "SENIN_HUGGINGFACE_KULLANICI_ADIN":
        print("Lütfen `upload_to_hub.py` dosyasında `HF_USERNAME` değişkenini kendi Hugging Face kullanıcı adınızla güncelleyin.")
    else:
        # Giriş yapılıp yapılmadığını kontrol etmek için basit bir deneme (opsiyonel)
        try:
            api = HfApi()
            user_info = api.whoami()
            print(f"Hugging Face'e '{user_info['name']}' olarak giriş yapılmış.")
            upload_model_and_tokenizer()
        except Exception as e:
            print(f"Hugging Face'e giriş yapılamadı veya bir hata oluştu: {e}")
            print("Lütfen `huggingface-cli login` komutu ile giriş yaptığınızdan emin olun.")