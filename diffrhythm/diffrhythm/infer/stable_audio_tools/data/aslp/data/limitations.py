from enum import Enum, auto
from math import inf
import logging

MAX_MOS = 5
MIN_MOS = -5
MIN_DURATION = 0
MAX_DURATION = inf

class AutoNameValue(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values) -> str:
        return name.lower()
    
    @classmethod
    def check(cls: Enum, obj: object) -> Enum:
        """
        检查并转换为对应枚举类
        """
        if isinstance(obj, cls):
            return obj
        try:
            item = cls(obj)
        except ValueError:
            if obj is None or obj == "None":
                item = cls(None)
            else:
                logging.warning(f"{cls.__name__} has not attribute with {obj}! Create NotExistEnum({obj})")
                item = NotExistEnum.create_not_exist_enum(obj, obj)
        return item
    
class NotExistEnum(AutoNameValue):
    @staticmethod
    def create_not_exist_enum(name: str, value: object):
        """
        根据不存在的值动态创建NotExistEnum
        """
        return AutoNameValue("NotExistEnum", {name: value})
    
    @staticmethod
    def is_same_type(cls) -> bool:
        """
        由于NotExistEnum是动态生成的类，无法使用isinstance来断言类。
        """
        return type(cls) == type(NotExistEnum)

class AgeChoices(AutoNameValue):
    """
    年龄类别
    """
    Empty = None

    Child = auto()      # 儿童
    Younger = auto()    # 青年
    Middle = auto()     # 中年
    Older = auto()      # 老年


class EmotionChoices(AutoNameValue):
    """
    情感类别
    """
    Empty = None
    Neutral = auto()
    Happy = auto()
    Surprise = auto()
    Angry = auto()
    Sad = auto()
    Fear = auto()
    Disgust = auto()
    Other = auto()


class StyleChoices(AutoNameValue):
    """
    语音风格类别
    """
    Empty = None
    Reading = auto()                # 朗读风格
    Storytelling = auto()           # 评书风格
    MartialNovel = auto()           # 武侠小说风格
    broadcast= auto()               # 电台播报风格
    CustomerService = auto()        # 客服风格
    Poetic = auto()                 # 诗歌风格
    Fairytal = auto()               # 童话故事风格
    Conversation = auto()           # 闲聊对话风格
    GameCharacter = auto()          # 游戏角色风格
    Humorous = auto()               # 幽默风格
    Thriller = auto()               # 惊悚风格
    Prose = auto()                  # 鸡汤散文风格
    ScienceEncyclopedia = auto()    # 科普百科风格
    Explain = auto()                # 讲解介绍风格
    Others = auto()                 # 其他风格


class LanguageChoices(AutoNameValue):
    """
    语种类别
    """
    Empty = None
    ZH = auto()
    EN = auto()
    
class GenderChoices(AutoNameValue):
    """
    性别类别
    """
    Empty = None
    Male = auto()
    Female = auto()


class SentencePatternChoices(AutoNameValue):
    """
    句式类别
    """
    Empty = None
    Declarative = auto()    # 陈述
    Question = auto()       # 疑问
    Imperative = auto()     # 祈使
    Exclamatory = auto()    # 感叹


class RhetoricChoices(AutoNameValue):
    """
    修辞手法类别
    """
    Empty = None
    Normal = auto()
    Exaggerate = auto()             # 夸张
    Opposition = auto()             # 对立
    Onomatopoeia = auto()           # 拟声
    Simile = auto()                 # 明喻
    Personification = auto()        # 拟人
    References = auto()             # 引语
    Irony = auto()                  # 反讽
    Question = auto()               # 反问
    Repeat = auto()                 # 重复
    Transition = auto()             # 过渡
