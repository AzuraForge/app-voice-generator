import logging
import json
import sys
import os
import argparse

# Proje kök dizinini Python yoluna ekle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from azuraforge_learner import Callback
from azuraforge_voicegen.pipeline import VoiceGeneratorPipeline, get_default_config

class MockProgressCallback(Callback):
    """Redis yerine ilerlemeyi konsola yazdıran sahte callback."""
    def on_epoch_end(self, event):
        payload = event.payload
        epoch = payload.get('epoch', '?')
        total_epochs = payload.get('total_epochs', '?')
        loss = payload.get('loss', float('nan'))
        print(f"  [DEBUG] Epoch {epoch}/{total_epochs} -> Loss: {loss:.6f}")

def run_isolated_test(args):
    """Pipeline'ı izole bir şekilde çalıştıran ana fonksiyon."""
    print("--- 🎙️ AzuraForge İzole Ses Üretim Pipeline Testi Başlatılıyor ---")

    default_config = get_default_config()
    print("✅ Varsayılan konfigürasyon yüklendi.")

    # Komut satırı argümanları ile konfigürasyonu override et
    final_config = default_config.copy()
    if args.epochs:
        final_config['training_params']['epochs'] = args.epochs
    if args.lr:
        final_config['training_params']['lr'] = args.lr
    if args.hidden_size:
        final_config['model_params']['hidden_size'] = args.hidden_size

    print("\n🔧 Çalıştırılacak Nihai Konfigürasyon:")
    print(json.dumps(final_config, indent=2))

    # .tmp dizinini projenin kökünde oluştur
    project_base_dir = os.path.dirname(os.path.abspath(__file__))
    final_config['experiment_dir'] = os.path.join(project_base_dir, '..', '.tmp', 'isolated_run')

    pipeline = VoiceGeneratorPipeline(final_config)
    mock_callback = MockProgressCallback()

    print("\n🚀 Pipeline çalıştırılıyor...")
    
    try:
        results = pipeline.run(callbacks=[mock_callback])

        print("\n--- ✅ Test Başarıyla Tamamlandı ---")
        print("Sonuç Metrikleri:")
        final_loss = results.get('history', {}).get('loss', [])[-1]
        print(f"  - Son Kayıp (Final Loss): {final_loss:.6f}")
        
        generated_path = results.get('generated_audio_path')
        if generated_path and os.path.exists(generated_path):
            print(f"\n🔊 Üretilen ses dosyası şuraya kaydedildi: {os.path.relpath(generated_path)}")
        else:
            print("\n⚠️ Üretilen ses dosyası bulunamadı.")

    except Exception:
        print(f"\n--- ❌ Test Sırasında Hata Oluştu ---")
        logging.error("Pipeline hatası:", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="AzuraForge Voice Generator - Isolated Runner")
    parser.add_argument("--epochs", type=int, help="Eğitim için epoch sayısı.")
    parser.add_argument("--lr", type=float, help="Öğrenme oranı (learning rate).")
    parser.add_argument("--hidden-size", type=int, help="LSTM gizli katman boyutu.")
    
    args = parser.parse_args()
    run_isolated_test(args)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    main()