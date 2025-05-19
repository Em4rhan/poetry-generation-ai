# generate_poetry.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Konfigürasyon ---
MODEL_PATH = "C:/Users/Emirhan/Desktop/poetry_generation_project/fine_tuned_poetry_model"  # Eğitilmiş modelin kaydedildiği klasör
DEVICE = 0 if torch.cuda.is_available() else -1 # GPU varsa kullan, yoksa CPU (-1)

def generate_poems(prompt_text, num_poems=3, max_len=150, temperature=0.8, top_k=50, top_p=0.95, no_repeat_ngram_size=2):
    print(f"Loading model and tokenizer from {MODEL_PATH}...")
    try:
        # Hugging Face pipeline ile metin üretimi
        text_generator_pipeline = pipeline(
            "text-generation",
            model=MODEL_PATH,
            tokenizer=MODEL_PATH, 
            device=DEVICE
        )
        print("Pipeline loaded successfully.")
    except Exception as e_pipeline:
        print(f"Error loading pipeline: {e_pipeline}")
        print("Attempting to load model and tokenizer manually for generation...")
        try:
            tokenizer_manual = AutoTokenizer.from_pretrained(MODEL_PATH)
            model_manual = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
            if DEVICE == 0 and torch.cuda.is_available():
                model_manual.to("cuda")
            elif DEVICE != -1 : # Belirli bir GPU ID'si varsa (örn: 1, 2..)
                model_manual.to(f"cuda:{DEVICE}")
            else: # CPU
                model_manual.to("cpu")
            
            print("Model and tokenizer loaded manually.")
            # Pipeline'ı manuel olarak simüle et
            def manual_text_generator(prompt, **kwargs):
                # pipeline'ın kabul ettiği argümanları model.generate'e çevir
                gen_kwargs = {k: v for k, v in kwargs.items() if k in ["max_length", "num_return_sequences", "do_sample", "temperature", "top_k", "top_p", "no_repeat_ngram_size", "pad_token_id"]}
                
                input_ids = tokenizer_manual.encode(prompt, return_tensors='pt').to(model_manual.device)
                outputs = model_manual.generate(input_ids, **gen_kwargs)
                
                return [{"generated_text": tokenizer_manual.decode(output_seq, skip_special_tokens=True)} for output_seq in outputs]

            text_generator_pipeline = manual_text_generator
            # tokenizer'ı da pipeline gibi davranması için bir özelliğe ata
            text_generator_pipeline.tokenizer = tokenizer_manual


        except Exception as e_manual:
            print(f"Fatal error loading model/tokenizer manually: {e_manual}")
            return

    print(f"\nGenerating {num_poems} poems with prompt: '{prompt_text}'")
    
    # pad_token_id'yi al (pipeline kullanılıyorsa tokenizer'dan, değilse manuel yüklenenden)
    pad_token_id_to_use = text_generator_pipeline.tokenizer.eos_token_id if hasattr(text_generator_pipeline, 'tokenizer') and text_generator_pipeline.tokenizer.eos_token_id is not None else 50256 # Varsayılan EOS

    generated_outputs = text_generator_pipeline(
        prompt_text,
        max_length=max_len, 
        num_return_sequences=num_poems,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=pad_token_id_to_use 
    )

    for i, output in enumerate(generated_outputs):
        print(f"\n--- POEM {i+1} ---")
        # Çıktının 'generated_text' anahtarını içerdiğinden emin ol
        if 'generated_text' in output:
            print(output['generated_text'])
        else:
            print("Error: 'generated_text' key not found in output.")
            print("Full output:", output) # Hata ayıklama için
        print("------------------")

if __name__ == "__main__":
    if DEVICE == 0:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU for generation.")

    user_prompt = input("Enter a starting line or theme for your poem (e.g., 'The moon shines bright'): ")
    if not user_prompt.strip():
        user_prompt = "The wind whispers through the trees" 
        print(f"No prompt entered, using default: '{user_prompt}'")

    generate_poems(
        prompt_text=user_prompt,
        num_poems=3,      
        max_len=120, # Şiirlerin çok uzamasını engellemek için prompt dahil toplam token
        temperature=0.8, 
        top_k=40,         
        top_p=0.92,
        no_repeat_ngram_size=3 # 3 kelimelik tekrarları engelle
    )