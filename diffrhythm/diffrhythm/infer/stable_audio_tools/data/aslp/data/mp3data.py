import os
import pyarrow
import numpy
from dataclasses import dataclass, field

from .base import Base

@dataclass
class Mp3Data(Base):
    # data_id: str                  = field(default=None)                   # String
    name: str                     = field(default=None, compare=False)    # 音乐名称
    mp3_binary: bytes             = field(default=None, compare=False)    # MP3 文件的二进制数据

    def __post_init__(self):
        if isinstance(self.mp3_binary, (bytes, type(None))):
            # 如果是字节数据或 None，则不做额外处理
            return
        elif isinstance(self.mp3_binary, str) and os.path.isfile(self.mp3_binary):
            # 如果 mp3_binary 是字符串类型的文件路径，则读取该文件的二进制内容
            try:
                with open(self.mp3_binary, "rb") as f:
                    self.mp3_binary = f.read()
            except Exception as e:
                raise NotImplementedError(f"无法从文件 {self.mp3_binary} 加载 MP3 数据: {e}")
        else:
            # 处理不支持的类型
            raise NotImplementedError(f"不支持的类型 {type(self.mp3_binary)}，只能从字节数据或文件路径读取")

    @staticmethod
    def get_example():
        # 返回一个 Mp3Data 的示例对象
        return Mp3Data(
            data_id="example_mp3_data",
            name="example_song",
            mp3_binary=b"example_mp3_binary_data",
        )

    @staticmethod
    def get_schema():
        # 定义数据的 schema，用于数据集创建
        data_schema = pyarrow.schema([
            pyarrow.field("data_id", pyarrow.string(), nullable=False),
            pyarrow.field("name", pyarrow.string(), nullable=False),
            pyarrow.field("mp3_binary", pyarrow.binary(), nullable=True),
        ])
        return data_schema
