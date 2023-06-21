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
from transformers import pipeline

asr = pipeline("automatic-speech-recognition")
example = minds[0]
asr(example["audio"]["array"])
example["english_transcription"]

from datasets import load_dataset
from datasets import Audio

minds = load_dataset("PolyAI/minds14", name="de-DE", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

example = minds[0]
example["transcription"]

from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="maxidl/wav2vec2-large-xlsr-german")
asr(example["audio"]["array"])