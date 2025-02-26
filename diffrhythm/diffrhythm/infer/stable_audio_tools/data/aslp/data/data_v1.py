
import pyarrow

from dataclasses import dataclass, field

from .base import Base
from .limitations import LanguageChoices, EmotionChoices, AgeChoices, StyleChoices, GenderChoices

@dataclass
class Data(Base):
    # Base Label
    source_filename: str        = field(default=None, compare=False)    # 
    text: str                   = field(default=None, compare=False)    # String
    """
    text
    """
    text_with_aed: str          = field(default=None, compare=False)    # String
    """
    text with AED tag
    """
    phones: str                 = field(default=None, compare=False)    # Phoneme list splited by space
    """
    phoneme list splited by space
    """
    mos: float                      = field(default=None, compare=False)    # MOS score
    force_alignment: list           = field(default=None, compare=False)    # 
    confidence: float               = field(default=None, compare=False)    # float
    speaker: str                    = field(default=None, compare=False)    # String
    language: LanguageChoices       = field(default=None, compare=False)    # LanguageChoices, Storage as string
    emotion: EmotionChoices         = field(default=None, compare=False)    # EmotionChoices, Storage as string
    age: AgeChoices                 = field(default=None, compare=False)    # AgeChoices, Storage as string
    style: StyleChoices             = field(default=None, compare=False)    # StyleChoices, Storage as string
    gender: GenderChoices           = field(default=None, compare=False)    # GenderChoices, Storage as string
    prev_dialog_data_id: str        = field(default=None, compare=False)    # String 

    def __post_init__(self):
        self.language = LanguageChoices.check(self.__to_lower_attr(self.language))
        self.emotion = EmotionChoices.check(self.__to_lower_attr(self.emotion))
        self.age = AgeChoices.check(self.__to_lower_attr(self.age))
        self.style = StyleChoices.check(self.__to_lower_attr(self.style))
        self.gender = GenderChoices.check(self.__to_lower_attr(self.gender))

    def __to_lower_attr(self, attr) -> str:
        if isinstance(attr, str):
            attr = attr.lower()
        return attr
    
    @staticmethod
    def get_example():
        return Data(
            data_id="aslp_example",
            source_filename="aslp_example_data_source_wav.wav",
            text="aslp_example_data_12345",
            text_with_aed="aslp_example_data_12345",
            phones="aslp example data 12345",
            mos=3.3,
            force_alignment=[1,2,3],
            confidence=0.88,
            speaker="example_speaker",
            language=LanguageChoices.ZH,
            emotion=EmotionChoices.Neutral,
            age=AgeChoices.Middle,
            style=StyleChoices.Conversation,
            gender=GenderChoices.Male,
            prev_dialog_data_id="aslp_example"
        )

    @staticmethod
    def get_schema():
        data_schema = pyarrow.schema([
            pyarrow.field("data_id", pyarrow.string(), nullable=False),
            pyarrow.field("source_filename", pyarrow.string()),
            pyarrow.field("text", pyarrow.string()),
            pyarrow.field("text_with_aed", pyarrow.string()),
            pyarrow.field("phones", pyarrow.string()),
            pyarrow.field("mos", pyarrow.float32()),
            pyarrow.field("force_alignment", pyarrow.list_(pyarrow.int16())),
            pyarrow.field("confidence", pyarrow.float32()),
            pyarrow.field("speaker", pyarrow.string()),
            pyarrow.field("language", pyarrow.string()),
            pyarrow.field("emotion", pyarrow.string()),
            pyarrow.field("age", pyarrow.string()),
            pyarrow.field("style", pyarrow.string()),
            pyarrow.field("gender", pyarrow.string()),
            pyarrow.field("prev_dialog_data_id", pyarrow.string())
        ])
        return data_schema