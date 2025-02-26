import os
import librosa
import numpy
import pyarrow

from dataclasses import dataclass, field

from .base import Base

@dataclass
class AudioData(Base):
    # data_id: str                = field(default=None)                   # String
    audio: numpy.ndarray        = field(default=None, compare=False)    # mono audio, numpy array dtype=float32 with range [-1, 1]
    """
    mono, numpy array dtype=float32, range [-1, 1]
    """
    sample_rate: int            = field(default=None, compare=False)    # sample rate of audio
    duration: float             = field(default=None, compare=False)    # ms

    def __post_init__(self):
        if isinstance(self.audio, (numpy.ndarray, type(None))):
            return
        elif isinstance(self.audio, bytes):
            self.audio = numpy.frombuffer(self.audio)
        elif isinstance(self.audio, str) and os.path.isfile(self.audio):
            try:
                audio, sr = librosa.load(self.audio, sr=self.sample_rate)
                audio = (audio * numpy.iinfo(numpy.int16).max).astype(numpy.int16)
                self.audio = audio
                if sr != self.sample_rate:
                    self.sample_rate = sr
                self.duration = self.audio.shape[-1] / self.sample_rate * 1000
            except Exception as e:
                raise NotImplementedError((f"Can't load audio from {self.audio}", e))
        else:
            raise NotImplementedError(f"Unsupported type {type(self.audio)}, only can read from bytes, string_path_to_audio")

    @staticmethod
    def get_example():
        return AudioData(
            # data_id="example_data",
            audio=numpy.random.randint(0, 1000, size=(1000, )).astype(numpy.int16),
            sample_rate=48000,
            duration=10.1,
        )

    @staticmethod
    def get_schema():
        data_schema = pyarrow.schema([
            pyarrow.field("data_id", pyarrow.string(), nullable=False),
            pyarrow.field("audio", pyarrow.list_(pyarrow.int16())),
            pyarrow.field("sample_rate", pyarrow.int32()),
            pyarrow.field("duration", pyarrow.float32()),
        ])
        return data_schema