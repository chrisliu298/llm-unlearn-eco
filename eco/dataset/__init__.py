from .arc import ARCChallenge, ARCEasy
from .bbc_news import BBCNews
from .boolq import BoolQ
from .commonsenseqa import CommonsenseQA
from .hellaswag import HellaSwag
from .hp_book import HPBook
from .mmlu import MMLU
from .mmlu_subset import (
    MMLUEconometrics,
    MMLUEconomics,
    MMLUJurisprudence,
    MMLULaw,
    MMLUMath,
    MMLUPhysics,
    MMLUSubset,
    MMLUWithoutEconomicsEconometrics,
    MMLUWithoutLawJurisprudence,
    MMLUWithoutPhysicsMath,
)
from .openbookqa import OpenBookQA
from .piqa import PIQA
from .social_i_qa import SocialIQA
from .tofu import TOFU, TOFUPerturbed
from .truthfulqa import TruthfulQA
from .winogrande import Winogrande
from .wmdp import WMDP, WMDPBio, WMDPChem, WMDPCyber

dataset_classes = {
    c.name: c
    for c in [
        ARCChallenge,
        ARCEasy,
        CommonsenseQA,
        HellaSwag,
        MMLU,
        OpenBookQA,
        TOFU,
        TOFUPerturbed,
        TruthfulQA,
        Winogrande,
        PIQA,
        BoolQ,
        SocialIQA,
        WMDP,
        WMDPBio,
        WMDPChem,
        WMDPCyber,
        HPBook,
        BBCNews,
        MMLUEconomics,
        MMLULaw,
        MMLUPhysics,
        MMLUWithoutEconomicsEconometrics,
        MMLUWithoutLawJurisprudence,
        MMLUWithoutPhysicsMath,
        MMLUEconometrics,
        MMLUJurisprudence,
        MMLUMath,
        MMLUSubset,
    ]
}
