# AzuraForge: Basit Ses Üretim Uygulaması

Bu proje, AzuraForge platformu için **üretken bir uygulama eklentisidir**. Basit bir ses veriseti üzerinde, bir sonraki ses dalgası örneğini tahmin ederek ham ses üretmeyi amaçlayan bir pipeline içerir.

Bu, platformun üretken yeteneklerini sergileyen ilk "kavram kanıtlama" (Proof of Concept) projesidir ve WaveNet gibi daha karmaşık modellere zemin hazırlar.

## 🏛️ Ekosistemdeki Yeri

Bu eklenti, AzuraForge ekosisteminin modüler ve genişletilebilir yapısının bir örneğidir. Projenin genel mimarisini, vizyonunu ve geliştirme rehberini anlamak için lütfen ana **[AzuraForge Platform Dokümantasyonuna](https://github.com/AzuraForge/platform/tree/main/docs)** başvurun.

## 🛠️ İzole Geliştirme ve Hızlı Test

Bu eklenti, tüm AzuraForge platformunu (`Docker`) çalıştırmadan, tamamen bağımsız olarak test edilebilir.

### Gereksinimler
1.  Ana platformun **[Geliştirme Rehberi](https://github.com/AzuraForge/platform/blob/main/docs/DEVELOPMENT_GUIDE.md)**'ne göre Python sanal ortamınızın kurulu ve aktif olduğundan emin olun.
2.  Bu reponun kök dizininde olduğunuzundan emin olun.

### Testi Çalıştırma (CPU üzerinde)
Aşağıdaki komut, pipeline'ı varsayılan ayarlarla çalıştıracaktır:
```bash
python tools/run_isolated.py
```

### Testi Çalıştırma (GPU üzerinde)
Eğer sisteminizde uyumlu bir NVIDIA GPU ve CUDA kurulu ise, eğitimi GPU üzerinde çalıştırmak için `AZURAFORGE_DEVICE` ortam değişkenini ayarlayabilirsiniz. Bu, eğitim sürecini önemli ölçüde hızlandıracaktır.

**Windows (PowerShell):**
```powershell
$env:AZURAFORGE_DEVICE="gpu"; python tools/run_isolated.py
```

**Linux / macOS / WSL:**
```bash
AZURAFORGE_DEVICE=gpu python tools/run_isolated.py
```
