import pyarrow
from dataclasses import dataclass, field
from .base import Base

@dataclass
class TextData(Base):
    data: str = field(default=None, compare=False)

    def __post_init__(self):
        if self.data is None:
            return
        assert isinstance(self.data, str), "data 应该是单个字符串"

    @staticmethod
    def get_example():
        return TextData(
            data_id="text_id",
            data="这是一个示例文本。",
        )

    @staticmethod
    def get_schema():
        data_schema = pyarrow.schema([
            pyarrow.field("data_id", pyarrow.string(), nullable=False),
            pyarrow.field("data", pyarrow.string()),  # 改为单个字符串
        ])
        return data_schema
    
    def to_dict(self, only_not_none: bool=False, str_wrap: bool=False):
        data_id = f"'{self.data_id}'" if str_wrap else self.data_id
        new_dict = {
            "data_id": data_id,
            "data": self.data,
        }
        return new_dict
