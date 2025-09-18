import enum
from ast import Not
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from matplotlib import category

LANG_CODES = ["eng_Latn", "tur_Latn", "pes_Arab", "zho_Hans", "ita_Latn"]


class LANGS(enum.Enum):
    eng_Latn = "eng_Latn"
    tur_Latn = "tur_Latn"
    pes_Arab = "pes_Arab"
    zho_Hans = "zho_Hans"
    ita_Latn = "ita_Latn"


@dataclass
class Perturbation:
    name: str
    category: str
    available_languages: List[LANGS] = field(default_factory=list)
    description: Optional[str] = None
    automatable: bool = False
    func: Optional[Callable] = None

    def __call__(
        self, word: str, lang: Optional[LANGS] = None, *args: Any, **kwds: Any
    ) -> str:
        if self.func:
            return self.func(word, lang, *args, **kwds)
        raise NotImplementedError(
            f"Must be implemented by subclasses of {self.__class__.__name__}"
        )


@dataclass
class PerturbationCategory:
    name: str
    description: Optional[str] = None
    perturbations: List[Perturbation] = field(default_factory=list)
