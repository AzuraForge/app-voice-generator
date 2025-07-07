# app-voice-generator/src/azuraforge_voicegen/config_schema.py
from pydantic import BaseModel, Field
from typing import Literal, Optional

class DataSourcingConfig(BaseModel):
    data_source: Literal["package_data"] = "package_data" # Şu an sadece package_data destekleniyor
    quantization_bits: int = Field(8, ge=1, le=16, description="Number of bits for mu-law quantization (e.g., 8 for 256 channels).")
    sample_rate: int = Field(8000, gt=0, description="Sample rate of the audio data.") # Yeni eklendi

class ModelParamsConfig(BaseModel):
    sequence_length: int = Field(256, gt=0, description="How many past audio samples the model looks at.")
    embedding_dim: int = Field(128, gt=0)
    hidden_size: int = Field(256, gt=0)

class TrainingParamsConfig(BaseModel):
    epochs: int = Field(20, gt=0)
    lr: float = Field(0.001, gt=0)

class VoiceGeneratorConfig(BaseModel):
    pipeline_name: Literal['voice_generator']
    data_sourcing: DataSourcingConfig
    model_params: ModelParamsConfig
    training_params: TrainingParamsConfig

    class Config:
        extra = 'forbid'