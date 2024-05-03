# RVC-Python-FastInference

A streamlined Python wrapper for fast inference with RVC.
Specifically designed for inference tasks.

## Introduction

This streamlined wrapper offers an efficient solution for integrating RVC into your Python projects, focusing primarily on rapid inference. Whether you're working on voice conversion applications or related projects, this tool simplifies the process while maintaining performance.

## Key Features
- Preloaded Models: Accelerate inference by loading models into memory beforehand, minimizing latency during runtime.
- Batch Processing: Enhance efficiency by enabling batch processing, allowing for simultaneous conversion of multiple inputs, further optimizing throughput.
- Support for Array Input and Output: Facilitate seamless integration with existing data pipelines by accepting and returning arrays, enhancing compatibility across various platforms and frameworks.

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

# Preloading model (Reduces inference time)

The initial execution will preload the model for the tag. Subsequent calls to inference with the same tag will benefit from preloaded components, thereby reducing inference time.
```
result_array, sample_rate = converter.generate_from_cache(
    audio_data="myaudiofile_path.wav",
    tag="yoimiya",
)
```

The param audio_data can be a path or a tuple with (array_data, sampling_rate)

```
# array_data = np.array([-22, -22, -15, ..., 0, 0, 0], dtype=np.int16)
# source_sample_rate = 16000
data = (array_data, source_sample_rate)
result_array, sample_rate = converter.generate_from_cache(
    audio_data=data,
    tag="yoimiya",
)
```
The result in both cases will be (array, sample_rate), which you can save or play in a notebook

```
# Save
import soundfile as sf

sf.write(
    file="output_file.wav",
    samplerate=sample_rate,
    data=result_array
)
```

```
# Play; need to install ipython
from IPython.display import Audio

Audio(result_array, rate=sample_rate)
```
When settings or the tag are altered, the model requires reloading. To maintain multiple preloaded models, you can instantiate another BaseLoader object.
```
second_converter = BaseLoader()
```
# Credits
- RVC-Project
- FFMPEG

# License
This project is licensed under the MIT License.

# Disclaimer
This software is provided for educational and research purposes only. The authors and contributors of this project do not endorse or encourage any misuse or unethical use of this software. Any use of this software for purposes other than those intended is solely at the user's own risk. The authors and contributors shall not be held responsible for any damages or liabilities arising from the use of this software inappropriately.
