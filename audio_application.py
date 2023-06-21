from datasets import load_dataset
from datasets import Audio


# 1.Audio classification with a pipeline
minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

from transformers import pipeline

classifier = pipeline(
    "audio-classification",
    model="anton-l/xtreme_s_xlsr_300m_minds14",
)
example = minds[0]
classifier(example["audio"]["array"])

id2label = minds.features["intent_class"].int2str
id2label(example["intent_class"])


# 2.Automatic speech recognition with a pipeline
