import numpy as np
import yaml
from importlib import resources
from scipy.io import wavfile
from scipy import signal # Yeniden örnekleme için
from typing import Tuple, Optional, Any, Dict
from pydantic import BaseModel

from azuraforge_learner import Sequential, Embedding, LSTM, Linear, AudioGenerationPipeline

def get_default_config():
    with resources.open_text("azuraforge_voicegen.config", "default_config.yml") as f:
        return yaml.safe_load(f)

def mu_law_encode(audio, quantization_channels):
    mu = float(quantization_channels - 1)
    # Ses verisini [-1, 1] aralığına getir
    audio_float = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    encoded = np.sign(audio_float) * np.log1p(mu * np.abs(audio_float)) / np.log1p(mu)
    return ((encoded + 1) / 2 * mu + 0.5).astype(np.int64)

class VoiceGeneratorPipeline(AudioGenerationPipeline):
    """
    Basit bir .wav dosyasından bir sonraki ses örneğini tahmin etmeyi öğrenir.
    """
    def get_config_model(self) -> Optional[type[BaseModel]]:
        return None

    def _load_data(self) -> np.ndarray:
        """Paket içindeki örnek ses dosyasını yükler, yeniden örnekler ve ön işler."""
        self.logger.info("Loading sample audio data from package...")
        target_sr = self.config.get("data_sourcing", {}).get("sample_rate", 8000)
        
        try:
            with resources.path("azuraforge_voicegen.data", "sample.wav") as wav_path:
                original_sr, waveform = wavfile.read(wav_path)
                self.logger.info(f"Loaded audio with original sample rate: {original_sr} and shape: {waveform.shape}")
        except (FileNotFoundError, ValueError):
            self.logger.warning("sample.wav not found or invalid. Generating random noise as fallback.")
            original_sr, waveform = target_sr, (np.random.rand(target_sr * 5) * 60000 - 30000).astype(np.int16)

        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1).astype(np.int16)

        # === YENİ: YENİDEN ÖRNEKLEME ADIMI ===
        if original_sr != target_sr:
            self.logger.info(f"Resampling waveform from {original_sr}Hz to {target_sr}Hz...")
            num_samples = int(len(waveform) * float(target_sr) / original_sr)
            waveform = signal.resample(waveform, num_samples).astype(np.int16)
            self.logger.info(f"Resampled waveform shape: {waveform.shape}")
        # === BİTTİ ===

        quantization_channels = 2 ** self.config.get("data_sourcing", {}).get("quantization_bits", 8)
        encoded_waveform = mu_law_encode(waveform, quantization_channels)
        self.logger.info(f"Waveform quantized to {quantization_channels} channels. Final data length: {len(encoded_waveform)}")
        
        return encoded_waveform

    def _create_model(self, vocab_size: int) -> Sequential:
        self.logger.info(f"Creating a generative model with vocab_size: {vocab_size}")
        model_params = self.config.get("model_params", {})
        embedding_dim = model_params.get("embedding_dim", 128)
        hidden_size = model_params.get("hidden_size", 256)

        model = Sequential(
            Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim),
            LSTM(input_size=embedding_dim, hidden_size=hidden_size, return_sequences=True), 
            Linear(hidden_size, vocab_size)
        )
        return model