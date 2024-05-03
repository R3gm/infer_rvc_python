# GPT-SoVITS-FastInference

A streamlined Python wrapper for fast inference with RVC.
This is designed solely for inference purposes.

## Introduction

description

## Getting Started

### Installation
```
pip install infer_rvc_python
```

# Usage

## Initialize the base class

```
from infer_rvc_python import BaseLoader

converter = BaseLoader(only_cpu=False, hubert_path=None, rmvpe_path=None)
```

## Define a tag and select the model along with other parameters.

```
converter.apply_conf(
        tag="yoimiya",
        file_model="model.pth",
        pitch_algo="rmvpe+",
        pitch_lvl=0,
        file_index="model.index",
        index_influence=0.66,
        respiration_median_filtering=3,
        envelope_ratio=0.25,
        consonant_breath_protection=0.33
    )
```

## Select the audio or audios you want to convert.

```
# audio_files = ["audio.wav", "haha.mp3"]
audio_files = "myaudio.mp3"

# speakers_list = ["sunshine", "yoimiya"]
speakers_list = "yoimiya"
```

## Perform inference

```
result = converter(
    audio_files,
    speakers_list,
    overwrite=False,
    parallel_workers=4
)
```
The `result` is a list with the paths of the converted files.

## Unload models
```
converter.unload_models()
```

# License
This project is licensed under the MIT License.

# Disclaimer
This software is provided for educational and research purposes only. The authors and contributors of this project do not endorse or encourage any misuse or unethical use of this software. Any use of this software for purposes other than those intended is solely at the user's own risk. The authors and contributors shall not be held responsible for any damages or liabilities arising from the use of this software inappropriately.
