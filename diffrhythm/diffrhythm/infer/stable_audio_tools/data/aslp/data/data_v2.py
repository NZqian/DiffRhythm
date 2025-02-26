
import pyarrow

from dataclasses import dataclass, field

from .base import Base
from .limitations import SentencePatternChoices, RhetoricChoices

@dataclass
class Data(Base):
    # Acoustic Label
    acoustic_speaking_rate_phone: str       = field(default=None, compare=False)
    acoustic_speaking_rate_utterance: str   = field(default=None, compare=False)
    acoustic_dynamic_range: str             = field(default=None, compare=False)
    acoustic_rms: str                       = field(default=None, compare=False)
    acoustic_rms_diff: str                  = field(default=None, compare=False)
    acoustic_mean_rolloff: str              = field(default=None, compare=False)
    acoustic_utterance_pitch_mean: str      = field(default=None, compare=False)
    acoustic_description: str               = field(default=None, compare=False)

    # Semantic Label
    semantic_sentence_pattern: SentencePatternChoices   = field(default=None, compare=False)
    semantic_rhetoric: RhetoricChoices                  = field(default=None, compare=False)
    semantic_style: str                                 = field(default=None, compare=False)
    semantic_emotion: str                               = field(default=None, compare=False)

    def __post_init__(self):
        self.semantic_sentence_pattern = SentencePatternChoices.check(self.__to_lower_attr(self.semantic_sentence_pattern))
        self.semantic_rhetoric = RhetoricChoices.check(self.__to_lower_attr(self.semantic_rhetoric))

    def __to_lower_attr(self, attr) -> str:
        if isinstance(attr, str):
            attr = attr.lower()
        return attr
    
    @staticmethod
    def get_example():
        return Data(
            data_id="aslp_example",
            acoustic_speaking_rate_phone="example",
            acoustic_speaking_rate_utterance="example",
            acoustic_dynamic_range="example",
            acoustic_rms="example",
            acoustic_rms_diff="example",
            acoustic_mean_rolloff="example",
            acoustic_utterance_pitch_mean="example",
            acoustic_description="example",
            semantic_sentence_pattern=SentencePatternChoices.Declarative,
            semantic_rhetoric=RhetoricChoices.Normal,
            semantic_style="example",
            semantic_emotion="example",
        )

    @staticmethod
    def get_schema():
        data_schema = pyarrow.schema([
            pyarrow.field("data_id", pyarrow.string(), nullable=False),
            pyarrow.field("acoustic_speaking_rate_phone", pyarrow.string()),
            pyarrow.field("acoustic_speaking_rate_utterance", pyarrow.string()),
            pyarrow.field("acoustic_dynamic_range", pyarrow.string()),
            pyarrow.field("acoustic_rms", pyarrow.string()),
            pyarrow.field("acoustic_rms_diff", pyarrow.string()),
            pyarrow.field("acoustic_mean_rolloff", pyarrow.string()),
            pyarrow.field("acoustic_utterance_pitch_mean", pyarrow.string()),
            pyarrow.field("acoustic_description", pyarrow.string()),
            pyarrow.field("semantic_sentence_pattern", pyarrow.string()),
            pyarrow.field("semantic_rhetoric", pyarrow.string()),
            pyarrow.field("semantic_style", pyarrow.string()),
            pyarrow.field("semantic_emotion", pyarrow.string()),
        ])
        return data_schema