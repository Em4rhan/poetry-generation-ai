import matplotlib.pyplot as plt
import ast # String'i sözlüğe güvenli çevirmek için

# Sağladığınız log verisi
log_data_string = """
{'loss': 4.9533, 'grad_norm': 5.659243106842041, 'learning_rate': 2.9911591355599216e-05, 'epoch': 0.16}
{'loss': 3.9644, 'grad_norm': 4.852148532867432, 'learning_rate': 2.9420432220039293e-05, 'epoch': 0.32}
{'loss': 3.8583, 'grad_norm': 3.740621328353882, 'learning_rate': 2.892927308447937e-05, 'epoch': 0.48}
{'loss': 3.7742, 'grad_norm': 2.9501211643218994, 'learning_rate': 2.843811394891945e-05, 'epoch': 0.65}
{'loss': 3.7819, 'grad_norm': 2.8760874271392822, 'learning_rate': 2.7946954813359528e-05, 'epoch': 0.81}
{'loss': 3.7342, 'grad_norm': 3.338052988052368, 'learning_rate': 2.745579567779961e-05, 'epoch': 0.97}
{'loss': 3.6534, 'grad_norm': 3.0119457244873047, 'learning_rate': 2.6964636542239686e-05, 'epoch': 1.13}
{'loss': 3.6427, 'grad_norm': 3.0855653285980225, 'learning_rate': 2.6473477406679766e-05, 'epoch': 1.29}
{'loss': 3.6458, 'grad_norm': 3.40236234664917, 'learning_rate': 2.5982318271119843e-05, 'epoch': 1.45}
{'loss': 3.6389, 'grad_norm': 2.9317679405212402, 'learning_rate': 2.549115913555992e-05, 'epoch': 1.62}
{'loss': 3.6261, 'grad_norm': 2.9503016471862793, 'learning_rate': 2.5e-05, 'epoch': 1.78}
{'loss': 3.6518, 'grad_norm': 3.232667922973633, 'learning_rate': 2.450884086444008e-05, 'epoch': 1.94}
{'loss': 3.5742, 'grad_norm': 2.895582675933838, 'learning_rate': 2.401768172888016e-05, 'epoch': 2.1}
{'loss': 3.5617, 'grad_norm': 2.588717460632324, 'learning_rate': 2.3526522593320236e-05, 'epoch': 2.26}
{'loss': 3.5734, 'grad_norm': 3.141781806945801, 'learning_rate': 2.3035363457760313e-05, 'epoch': 2.42}
{'loss': 3.5353, 'grad_norm': 2.788400411605835, 'learning_rate': 2.2544204322200394e-05, 'epoch': 2.58}
{'loss': 3.5805, 'grad_norm': 2.898452043533325, 'learning_rate': 2.205304518664047e-05, 'epoch': 2.75}
{'loss': 3.5457, 'grad_norm': 3.3502120971679688, 'learning_rate': 2.156188605108055e-05, 'epoch': 2.91}
{'loss': 3.5242, 'grad_norm': 2.666952610015869, 'learning_rate': 2.107072691552063e-05, 'epoch': 3.07}
{'loss': 3.5024, 'grad_norm': 3.5603537559509277, 'learning_rate': 2.0579567779960706e-05, 'epoch': 3.23}
{'loss': 3.4931, 'grad_norm': 3.048949956893921, 'learning_rate': 2.0088408644400787e-05, 'epoch': 3.39}
{'loss': 3.5014, 'grad_norm': 2.7299108505249023, 'learning_rate': 1.9597249508840864e-05, 'epoch': 3.55}
{'loss': 3.506, 'grad_norm': 2.7508530616760254, 'learning_rate': 1.9106090373280944e-05, 'epoch': 3.71}
{'loss': 3.4723, 'grad_norm': 3.4603986740112305, 'learning_rate': 1.861493123772102e-05, 'epoch': 3.88}
{'loss': 3.5147, 'grad_norm': 2.4097940921783447, 'learning_rate': 1.812573673870334e-05, 'epoch': 4.04}
{'loss': 3.4553, 'grad_norm': 2.7181122303009033, 'learning_rate': 1.763457760314342e-05, 'epoch': 4.2}
{'loss': 3.4674, 'grad_norm': 3.0553739070892334, 'learning_rate': 1.7143418467583497e-05, 'epoch': 4.36}
{'loss': 3.4258, 'grad_norm': 3.18560528755188, 'learning_rate': 1.6652259332023577e-05, 'epoch': 4.52}
{'loss': 3.4492, 'grad_norm': 2.8891565799713135, 'learning_rate': 1.6161100196463654e-05, 'epoch': 4.68}
{'loss': 3.4584, 'grad_norm': 2.9687325954437256, 'learning_rate': 1.566994106090373e-05, 'epoch': 4.85}
{'loss': 3.448, 'grad_norm': 2.773918628692627, 'learning_rate': 1.517878192534381e-05, 'epoch': 5.01}
{'loss': 3.4016, 'grad_norm': 3.4436490535736084, 'learning_rate': 1.4687622789783891e-05, 'epoch': 5.17}
{'loss': 3.3948, 'grad_norm': 3.0547211170196533, 'learning_rate': 1.4196463654223968e-05, 'epoch': 5.33}
{'loss': 3.4397, 'grad_norm': 2.986570119857788, 'learning_rate': 1.3705304518664049e-05, 'epoch': 5.49}
{'loss': 3.4236, 'grad_norm': 2.713714122772217, 'learning_rate': 1.3216110019646364e-05, 'epoch': 5.65}
{'loss': 3.4144, 'grad_norm': 2.842848777770996, 'learning_rate': 1.2724950884086445e-05, 'epoch': 5.81}
{'loss': 3.4381, 'grad_norm': 2.7679033279418945, 'learning_rate': 1.2233791748526522e-05, 'epoch': 5.98}
{'loss': 3.4043, 'grad_norm': 2.8565585613250732, 'learning_rate': 1.1742632612966603e-05, 'epoch': 6.14}
{'loss': 3.3748, 'grad_norm': 2.908310651779175, 'learning_rate': 1.125147347740668e-05, 'epoch': 6.3}
{'loss': 3.4025, 'grad_norm': 2.80248761177063, 'learning_rate': 1.0760314341846759e-05, 'epoch': 6.46}
{'loss': 3.3645, 'grad_norm': 3.232140302658081, 'learning_rate': 1.0269155206286838e-05, 'epoch': 6.62}
{'loss': 3.4071, 'grad_norm': 3.14410138130188, 'learning_rate': 9.777996070726915e-06, 'epoch': 6.78}
{'loss': 3.3829, 'grad_norm': 2.9384396076202393, 'learning_rate': 9.286836935166996e-06, 'epoch': 6.94}
{'loss': 3.3806, 'grad_norm': 3.0571210384368896, 'learning_rate': 8.797642436149313e-06, 'epoch': 7.11}
{'loss': 3.3619, 'grad_norm': 3.17140531539917, 'learning_rate': 8.308447937131632e-06, 'epoch': 7.27}
{'loss': 3.3747, 'grad_norm': 3.050002336502075, 'learning_rate': 7.817288801571709e-06, 'epoch': 7.43}
{'loss': 3.3589, 'grad_norm': 3.0495858192443848, 'learning_rate': 7.326129666011788e-06, 'epoch': 7.59}
{'loss': 3.3675, 'grad_norm': 3.1047475337982178, 'learning_rate': 6.834970530451867e-06, 'epoch': 7.75}
{'loss': 3.3589, 'grad_norm': 2.7679784297943115, 'learning_rate': 6.343811394891946e-06, 'epoch': 7.91}
{'loss': 3.3572, 'grad_norm': 3.082465648651123, 'learning_rate': 5.852652259332023e-06, 'epoch': 8.07}
{'loss': 3.3672, 'grad_norm': 2.916391134262085, 'learning_rate': 5.361493123772102e-06, 'epoch': 8.24}
{'loss': 3.3315, 'grad_norm': 3.354583740234375, 'learning_rate': 4.870333988212181e-06, 'epoch': 8.4}
{'loss': 3.332, 'grad_norm': 2.7348413467407227, 'learning_rate': 4.37917485265226e-06, 'epoch': 8.56}
{'loss': 3.3768, 'grad_norm': 2.9915056228637695, 'learning_rate': 3.8880157170923385e-06, 'epoch': 8.72}
{'loss': 3.332, 'grad_norm': 2.9030961990356445, 'learning_rate': 3.3988212180746563e-06, 'epoch': 8.88}
{'loss': 3.3518, 'grad_norm': 2.974374532699585, 'learning_rate': 2.9076620825147347e-06, 'epoch': 9.04}
{'loss': 3.3187, 'grad_norm': 2.726005792617798, 'learning_rate': 2.4165029469548136e-06, 'epoch': 9.21}
{'loss': 3.3403, 'grad_norm': 3.7003660202026367, 'learning_rate': 1.9253438113948916e-06, 'epoch': 9.37}
{'loss': 3.3529, 'grad_norm': 2.964127540588379, 'learning_rate': 1.4341846758349705e-06, 'epoch': 9.53}
{'loss': 3.3263, 'grad_norm': 2.735247850418091, 'learning_rate': 9.430255402750491e-07, 'epoch': 9.69}
{'loss': 3.3707, 'grad_norm': 2.5593960285186768, 'learning_rate': 4.518664047151277e-07, 'epoch': 9.85}
"""

# Veriyi satırlara böl ve her satırı sözlüğe çevir
log_entries = []
for line in log_data_string.strip().split('\n'):
    try:
        # ast.literal_eval, string'i Python sözlüğüne güvenli bir şekilde dönüştürür
        entry = ast.literal_eval(line.strip())
        # Sadece ilgili anahtarları içeren sözlükleri alalım
        if all(key in entry for key in ['loss', 'grad_norm', 'learning_rate', 'epoch']):
            log_entries.append(entry)
    except (SyntaxError, ValueError) as e:
        print(f"Uyarı: Satır '{line}' işlenemedi: {e}")
        # Hatalı veya ilgisiz satırları atla (örneğin, ilerleme çubuğu veya özet satırları)

# Veri yoksa çık
if not log_entries:
    print("Grafik çizmek için geçerli log verisi bulunamadı.")
    exit()

# Grafikler için listeleri oluştur
epochs = [entry['epoch'] for entry in log_entries]
losses = [entry['loss'] for entry in log_entries]
grad_norms = [entry['grad_norm'] for entry in log_entries]
learning_rates = [entry['learning_rate'] for entry in log_entries]

# Grafikleri çiz
fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True) # 3 satır, 1 sütun, x eksenini paylaş

# Loss grafiği
axs[0].plot(epochs, losses, marker='o', linestyle='-', color='r', label='Loss')
axs[0].set_ylabel('Loss Değeri')
axs[0].set_title('Eğitim Kaybı (Loss) vs. Epoch')
axs[0].legend()
axs[0].grid(True)

# Gradient Norm grafiği
axs[1].plot(epochs, grad_norms, marker='s', linestyle='--', color='g', label='Gradient Norm')
axs[1].set_ylabel('Gradient Norm Değeri')
axs[1].set_title('Gradient Norm vs. Epoch')
axs[1].legend()
axs[1].grid(True)

# Learning Rate grafiği
axs[2].plot(epochs, learning_rates, marker='^', linestyle=':', color='b', label='Learning Rate')
axs[2].set_ylabel('Öğrenme Oranı (Learning Rate)')
axs[2].set_title('Öğrenme Oranı vs. Epoch')
axs[2].set_xlabel('Epoch')
axs[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Bilimsel gösterim
axs[2].legend()
axs[2].grid(True)

# Grafiklerin düzgün görünmesini sağla
plt.tight_layout()

# Grafikleri göster
plt.show()

# İsteğe bağlı: Veriyi CSV dosyasına yazma
# import pandas as pd
# if log_entries:
#     df = pd.DataFrame(log_entries)
#     df.to_csv('training_logs.csv', index=False)
#     print("Veri 'training_logs.csv' dosyasına kaydedildi.")