from .answer_prob import AnswerProb
from .bertscore import BERTScore
from .bleu import BLEU
from .bleurt import BLEURT
from .choice_by_top_logit import ChoiceByTopLogit
from .choice_by_top_prob import ChoiceByTopProb
from .exact_match import ExactMatch
from .meteor import METEOR
from .normalized_answer_prob import NormalizedAnswerProb
from .perplexity import Perplexity
from .rouge import ROUGE
from .rouge_recall import ROUGERecall
from .sacrebleu import SacreBLEU
from .truth_ratio import TruthRatio
from .unique_token_ratio import UniqueTokenRatio

evaluator_classes = {
    c.name: c
    for c in [
        AnswerProb,
        BERTScore,
        BLEU,
        BLEURT,
        ChoiceByTopLogit,
        ChoiceByTopProb,
        ExactMatch,
        METEOR,
        NormalizedAnswerProb,
        Perplexity,
        ROUGE,
        ROUGERecall,
        SacreBLEU,
        TruthRatio,
        UniqueTokenRatio,
    ]
}
