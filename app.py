# app.py

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Model ve Tokenizer Yükleme Fonksiyonu ---
# Bu fonksiyon, modeli ve tokenizer'ı sadece bir kez yüklemek için kullanılabilir (opsiyonel optimizasyon)
# Ancak basitlik için doğrudan ana fonksiyonda da yüklenebilir.

MODEL_PATH = "C:/Users/Emirhan/Desktop/poetry_generation_project/fine_tuned_poetry_model_v3"  # Eğitilmiş modelin kaydedildiği klasör
DEVICE = 0 if torch.cuda.is_available() else -1 # GPU varsa kullan, yoksa CPU (-1)

print(f"Kullanılacak cihaz: {'GPU' if DEVICE == 0 else 'CPU'}")

# Pipeline'ı global olarak yükleyebiliriz veya her çağrıda yeniden oluşturabiliriz.
# Uygulama boyunca aynı kalacaksa global yüklemek daha verimli olabilir.
try:
    print(f"Model ve tokenizer yükleniyor: {MODEL_PATH}...")
    text_generator_pipeline = pipeline(
        "text-generation",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        device=DEVICE
    )
    print("Pipeline başarıyla yüklendi.")
except Exception as e:
    print(f"HATA: Pipeline yüklenirken sorun oluştu: {e}")
    print("Lütfen model yolunun doğru olduğundan ve model dosyalarının tam olduğundan emin olun.")
    text_generator_pipeline = None # Hata durumunda None ata

# --- Şiir Üretme Fonksiyonu (Gradio Arayüzü İçin) ---
def generate_poem_interface(
    prompt_text,
    max_len=150,
    num_poems=1, # Arayüzde genellikle tek bir çıktı göstermek daha iyidir
    temperature=0.8,
    top_k=40,
    top_p=0.92,
    no_repeat_ngram_size=3
):
    if text_generator_pipeline is None:
        return "HATA: Model yüklenemedi. Lütfen konsol loglarını kontrol edin."

    if not prompt_text or not prompt_text.strip():
        return "Lütfen bir başlangıç dizesi veya tema girin."

    try:
        print(f"Şiir üretiliyor... Prompt: '{prompt_text}'")
        # pad_token_id'yi al
        pad_token_id_to_use = text_generator_pipeline.tokenizer.eos_token_id if \
                              hasattr(text_generator_pipeline, 'tokenizer') and \
                              text_generator_pipeline.tokenizer.eos_token_id is not None else 50256

        generated_outputs = text_generator_pipeline(
            prompt_text,
            max_length=int(max_len), # Gradio'dan gelen değer string olabilir, int'e çevir
            num_return_sequences=int(num_poems),
            do_sample=True,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
            pad_token_id=pad_token_id_to_use,
            truncation=True # Truncation uyarısını önlemek için
        )
        
        # Arayüz için genellikle tek bir şiir döndürürüz, ama birden fazla istenirse birleştirilebilir
        poems_output = ""
        for i, output in enumerate(generated_outputs):
            if 'generated_text' in output:
                poems_output += f"--- ŞİİR {i+1} ---\n"
                poems_output += output['generated_text']
                poems_output += "\n------------------\n"
            else:
                poems_output += f"HATA: Şiir {i+1} üretilirken 'generated_text' anahtarı bulunamadı.\n"
        
        print("Şiir üretimi tamamlandı.")
        return poems_output.strip()

    except Exception as e:
        print(f"HATA: Şiir üretimi sırasında bir sorun oluştu: {e}")
        return f"Bir hata oluştu: {str(e)}"

# --- Gradio Arayüzünü Oluşturma ---
# Arayüz elemanları ve açıklamaları
title = "📜 Poetry Producer Gen Ai 📜"
description = """
**ÖNEMLİ NOT:** Model İngilizce şiirler üzerine eğitildiği için, lütfen başlangıç dizesini/temayı **İngilizce** olarak giriniz. 
Farklı dillerde anlamlı sonuçlar alınamayabilir.
Aşağıdaki kutucuğa bir başlangıç dizesi veya tema girin ve "Şiir Üret" butonuna tıklayın.
Üretim parametrelerini kaydırıcılar ve kutucuklarla ayarlayabilirsiniz.
"""
article = "<p style='text-align: center;'>Emirhan Gürbüz | İngilizce Şiir Üretme"

# Gradio arayüzünü tanımla
iface = gr.Interface(
    fn=generate_poem_interface,
    inputs=[
        gr.Textbox(lines=2, placeholder="Örn: i love your eyes...", label="Başlangıç Dizesi / Tema"),
        gr.Slider(minimum=50, maximum=300, value=150, step=10, label="Maksimum Şiir Uzunluğu (token)"),
        gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Üretilecek Şiir Sayısı"), # Genellikle 1 daha iyi arayüz için
        gr.Slider(minimum=0.1, maximum=1.5, value=0.8, step=0.05, label="Sıcaklık (Temperature - Yaratıcılık)"),
        gr.Slider(minimum=1, maximum=100, value=40, step=1, label="Top-K Örnekleme"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.92, step=0.01, label="Top-P (Nucleus) Örnekleme"),
        gr.Slider(minimum=0, maximum=5, value=3, step=1, label="Tekrar Eden N-gram Engelleme Boyutu (0=Kapalı)")
    ],
    outputs=gr.Textbox(label="Üretilen Şiir(ler)", lines=15),
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(), # Farklı temalar deneyebilirsin:Default(), Glass(), Monarch(), etc.
    allow_flagging="never" # Kullanıcıların çıktıları işaretlemesini engelle
)

# Arayüzü başlat
if __name__ == "__main__":
    if text_generator_pipeline is not None:
        print("Gradio arayüzü başlatılıyor...")
        iface.launch(share=False) # share=True ile herkese açık bir link oluşturur (dikkatli kullan)
    else:
        print("Model yüklenemediği için Gradio arayüzü başlatılamıyor.")