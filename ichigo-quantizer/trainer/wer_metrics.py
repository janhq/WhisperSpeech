import jiwer
import pandas as pd
from whisper_normalizer.english import EnglishTextNormalizer

engnorm = EnglishTextNormalizer()


def whisper_normalize(x):
    if type(x) == list:
        return [engnorm(y) for y in x]
    else:
        return engnorm(x)


default_transform = jiwer.transforms.Compose(
    [
        jiwer.transforms.ToLowerCase(),
        jiwer.transforms.ExpandCommonEnglishContractions(),
        whisper_normalize,
        jiwer.transforms.RemoveMultipleSpaces(),
        jiwer.transforms.Strip(),
        jiwer.transforms.RemovePunctuation(),
        jiwer.transforms.ReduceToListOfListOfWords(),
    ]
)


class DfBuilder:
    def __init__(self):
        self.data = {}

    def push(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.data:
                self.data[k] = [v]
            else:
                self.data[k].append(v)

    def df(self):
        return pd.DataFrame(self.data)


class WERStats(DfBuilder):
    def __init__(self, transform=default_transform):
        super().__init__()
        self.reference_transform = transform
        self.hypothesis_transform = transform

    def push_sample(self, snd, gt_text, text, idx=None):
        if snd is not None:
            self.push(secs=snd.shape[-1] / 16000)
        diff = jiwer.process_words(
            gt_text,
            text,
            reference_transform=self.reference_transform,
            hypothesis_transform=self.hypothesis_transform,
        )
        self.push(
            idx=idx,
            gt_text=gt_text,
            text=text,
            wer=diff.wer,
            mer=diff.mer,
            wil=diff.wil,
            wip=diff.wip,
        )
        return diff


def compute_wer(pred_ids, target_ids, tokenizer=None):
    """
    Compute Word Error Rate between predicted and target token sequences.

    Args:
        pred_ids: Predicted token IDs
        target_ids: Target token IDs
        tokenizer: Optional tokenizer for decoding. If None, uses default Whisper tokenizer

    Returns:
        float: Word Error Rate score
    """
    if tokenizer is None:
        import whisper

        tokenizer = whisper.tokenizer.get_tokenizer(True, language="vi")

    # Decode token sequences to text
    pred_text = tokenizer.decode(pred_ids.tolist())
    target_text = tokenizer.decode(target_ids.tolist())

    # Calculate WER using jiwer
    stats = WERStats()
    stats.push_sample(None, target_text, pred_text)
    return stats.df()["wer"].mean()
