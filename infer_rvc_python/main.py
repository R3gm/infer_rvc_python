from infer_rvc_python.lib.log_config import logger
import torch
import gc
import numpy as np
import os
import warnings
import threading
from tqdm import tqdm
from infer_rvc_python.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer_rvc_python.lib.audio import load_audio
import soundfile as sf
from scipy import signal
from time import time as ttime
import faiss
from infer_rvc_python.root_pipe import VC, change_rms, bh, ah
import librosa
from urllib.parse import urlparse
import copy

warnings.filterwarnings("ignore")


class Config:
    def __init__(self, only_cpu=False):
        self.device = "cuda:0"
        self.is_half = True
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        (
            self.x_pad,
            self.x_query,
            self.x_center,
            self.x_max
        ) = self.device_config(only_cpu)

    def device_config(self, only_cpu) -> tuple:
        if torch.cuda.is_available() and not only_cpu:
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                logger.info(
                    "16/10 Series GPUs and P40 excel "
                    "in single-precision tasks."
                )
                self.is_half = False
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
        elif torch.backends.mps.is_available() and not only_cpu:
            logger.info("Supported N-card not found, using MPS for inference")
            self.device = "mps"
        else:
            logger.info("No supported N-card found, using CPU for inference")
            self.device = "cpu"
            self.is_half = False

        if self.n_cpu == 0:
            self.n_cpu = os.cpu_count()

        if self.is_half:
            # 6GB VRAM configuration
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5GB VRAM configuration
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        logger.info(
            f"Config: Device is {self.device}, "
            f"half precision is {self.is_half}"
        )

        return x_pad, x_query, x_center, x_max


BASE_DOWNLOAD_LINK = "https://huggingface.co/r3gm/sonitranslate_voice_models/resolve/main/"
BASE_MODELS = [
    "hubert_base.pt",
    "rmvpe.pt"
]
BASE_DIR = "."


def load_file_from_url(
    url: str,
    model_dir: str,
    file_name: str | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> str:
    """Download a file from `url` into `model_dir`,
    using the file present if possible.

    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))

    # Overwrite
    if os.path.exists(cached_file):
        if overwrite or os.path.getsize(cached_file) == 0:
            os.remove(cached_file)

    # Download
    if not os.path.exists(cached_file):
        logger.info(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file

        download_url_to_file(url, cached_file, progress=progress)
    else:
        logger.debug(cached_file)

    return cached_file


def friendly_name(file: str):
    if file.startswith("http"):
        file = urlparse(file).path

    file = os.path.basename(file)
    model_name, extension = os.path.splitext(file)
    return model_name, extension


def download_manager(
    url: str,
    path: str,
    extension: str = "",
    overwrite: bool = False,
    progress: bool = True,
):
    url = url.strip()

    name, ext = friendly_name(url)
    name += ext if not extension else f".{extension}"

    if url.startswith("http"):
        filename = load_file_from_url(
            url=url,
            model_dir=path,
            file_name=name,
            overwrite=overwrite,
            progress=progress,
        )
    else:
        filename = path

    return filename


def load_hu_bert(config, hubert_path=None):
    from fairseq import checkpoint_utils

    if hubert_path is None:
        hubert_path = ""
    if not os.path.exists(hubert_path):
        for id_model in BASE_MODELS:
            download_manager(
                os.path.join(BASE_DOWNLOAD_LINK, id_model), BASE_DIR
            )
        hubert_path = "hubert_base.pt"

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [hubert_path],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

    return hubert_model


def load_trained_model(model_path, config):

    if not model_path:
        raise ValueError("No model found")

    logger.info("Loading %s" % model_path)
    cpt = torch.load(model_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    if if_f0 == 0:
        # protect to 0.5 need?
        pass

    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(
                *cpt["config"], is_half=config.is_half
            )
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(
                *cpt["config"], is_half=config.is_half
            )
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q

    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().to(config.device)

    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()

    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]

    return n_spk, tgt_sr, net_g, vc, cpt, version


class BaseLoader:
    def __init__(self, only_cpu=False, hubert_path=None, rmvpe_path=None):
        self.model_config = {}
        self.config = None
        self.cache_model = {}
        self.only_cpu = only_cpu
        self.hubert_path = hubert_path
        self.rmvpe_path = rmvpe_path

    def apply_conf(
        self,
        tag="base_model",
        file_model="",
        pitch_algo="pm",
        pitch_lvl=0,
        file_index="",
        index_influence=0.66,
        respiration_median_filtering=3,
        envelope_ratio=0.25,
        consonant_breath_protection=0.33,
        resample_sr=0,
        file_pitch_algo="",
    ):

        if not file_model:
            raise ValueError("Model not found")

        if file_index is None:
            file_index = ""

        if file_pitch_algo is None:
            file_pitch_algo = ""

        if not self.config:
            self.config = Config(self.only_cpu)
            self.hu_bert_model = None
            self.model_pitch_estimator = None

        self.model_config[tag] = {
            "file_model": file_model,
            "pitch_algo": pitch_algo,
            "pitch_lvl": pitch_lvl,  # no decimal
            "file_index": file_index,
            "index_influence": index_influence,
            "respiration_median_filtering": respiration_median_filtering,
            "envelope_ratio": envelope_ratio,
            "consonant_breath_protection": consonant_breath_protection,
            "resample_sr": resample_sr,
            "file_pitch_algo": file_pitch_algo,
        }
        return f"CONFIGURATION APPLIED FOR {tag}: {file_model}"

    def infer(
        self,
        task_id,
        params,
        # load model
        n_spk,
        tgt_sr,
        net_g,
        pipe,
        cpt,
        version,
        if_f0,
        # load index
        index_rate,
        index,
        big_npy,
        # load f0 file
        inp_f0,
        # audio file
        input_audio_path,
        overwrite,
        type_output,
    ):

        f0_method = params["pitch_algo"]
        f0_up_key = params["pitch_lvl"]
        filter_radius = params["respiration_median_filtering"]
        resample_sr = params["resample_sr"]
        rms_mix_rate = params["envelope_ratio"]
        protect = params["consonant_breath_protection"]
        base_sr = 16000

        if isinstance(input_audio_path, tuple):
            if f0_method == "harvest":
                raise ValueError("Harvest not support from array")
            audio = input_audio_path[0]
            source_sr = input_audio_path[1]
            if source_sr != base_sr:
                audio = librosa.resample(
                    audio.astype(np.float32),
                    orig_sr=source_sr,
                    target_sr=base_sr
                )
            audio = audio.astype(np.float32).flatten()
        elif not os.path.exists(input_audio_path):
            raise ValueError(
                "The audio file was not found or is not "
                f"a valid file: {input_audio_path}"
            )
        else:
            audio = load_audio(input_audio_path, base_sr)

        f0_up_key = int(f0_up_key)

        # Normalize audio
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max

        times = [0, 0, 0]

        # filters audio signal, pads it, computes sliding window sums,
        # and extracts optimized time indices
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(
            audio, (pipe.window // 2, pipe.window // 2), mode="reflect"
        )
        opt_ts = []
        if audio_pad.shape[0] > pipe.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(pipe.window):
                audio_sum += audio_pad[i:i - pipe.window]
            for t in range(pipe.t_center, audio.shape[0], pipe.t_center):
                opt_ts.append(
                    t
                    - pipe.t_query
                    + np.where(
                        np.abs(audio_sum[t - pipe.t_query: t + pipe.t_query])
                        == np.abs(audio_sum[t - pipe.t_query: t + pipe.t_query]).min()
                    )[0][0]
                )

        s = 0
        audio_opt = []
        t = None
        t1 = ttime()

        sid_value = 0
        sid = torch.tensor(sid_value, device=pipe.device).unsqueeze(0).long()

        # Pads audio symmetrically, calculates length divided by window size.
        audio_pad = np.pad(audio, (pipe.t_pad, pipe.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // pipe.window

        # Estimates pitch from audio signal
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = pipe.get_f0(
                input_audio_path,
                audio_pad,
                p_len,
                f0_up_key,
                f0_method,
                filter_radius,
                inp_f0,
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if pipe.device == "mps":
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(
                pitch, device=pipe.device
            ).unsqueeze(0).long()
            pitchf = torch.tensor(
                pitchf, device=pipe.device
            ).unsqueeze(0).float()

        t2 = ttime()
        times[1] += t2 - t1
        for t in opt_ts:
            t = t // pipe.window * pipe.window
            if if_f0 == 1:
                pitch_slice = pitch[
                    :, s // pipe.window: (t + pipe.t_pad2) // pipe.window
                ]
                pitchf_slice = pitchf[
                    :, s // pipe.window: (t + pipe.t_pad2) // pipe.window
                ]
            else:
                pitch_slice = None
                pitchf_slice = None

            audio_slice = audio_pad[s:t + pipe.t_pad2 + pipe.window]
            audio_opt.append(
                pipe.vc(
                    self.hu_bert_model,
                    net_g,
                    sid,
                    audio_slice,
                    pitch_slice,
                    pitchf_slice,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[pipe.t_pad_tgt:-pipe.t_pad_tgt]
            )
            s = t

        pitch_end_slice = pitch[
            :, t // pipe.window:
        ] if t is not None else pitch
        pitchf_end_slice = pitchf[
            :, t // pipe.window:
        ] if t is not None else pitchf

        audio_opt.append(
            pipe.vc(
                self.hu_bert_model,
                net_g,
                sid,
                audio_pad[t:],
                pitch_end_slice,
                pitchf_end_slice,
                times,
                index,
                big_npy,
                index_rate,
                version,
                protect,
            )[pipe.t_pad_tgt:-pipe.t_pad_tgt]
        )

        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate != 1:
            audio_opt = change_rms(
                audio, 16000, audio_opt, tgt_sr, rms_mix_rate
            )
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if tgt_sr != resample_sr >= 16000:
            final_sr = resample_sr
        else:
            final_sr = tgt_sr

        """
        "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            times[0],
            times[1],
            times[2],
        ), (final_sr, audio_opt)

        """

        if type_output == "array":
            return audio_opt, final_sr

        if overwrite:
            output_audio_path = input_audio_path  # Overwrite
        else:
            basename = os.path.basename(input_audio_path)
            dirname = os.path.dirname(input_audio_path)

            new_basename = basename.split(
                '.')[0] + "_edited." + basename.split('.')[-1]
            new_path = os.path.join(dirname, new_basename)

            output_audio_path = new_path

        # Save file
        if type_output:
            output_audio_path = os.path.splitext(
                output_audio_path
            )[0]+f".{type_output}"

        try:
            sf.write(
                file=output_audio_path,
                samplerate=final_sr,
                data=audio_opt
            )
        except Exception as e:
            logger.error(e)
            logger.error("Error saving file, trying with WAV format")
            output_audio_path = os.path.splitext(output_audio_path)[0]+".wav"
            sf.write(
                file=output_audio_path,
                samplerate=final_sr,
                data=audio_opt
            )

        logger.info(str(output_audio_path))

        self.model_config[task_id]["result"].append(output_audio_path)
        self.output_list.append(output_audio_path)

    def run_threads(self, threads):
        # Start threads
        for thread in threads:
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        gc.collect()
        torch.cuda.empty_cache()

    def unload_models(self):
        self.hu_bert_model = None
        self.model_pitch_estimator = None
        self.model_vc = {}
        self.cache_model = {}
        gc.collect()
        torch.cuda.empty_cache()

    def __call__(
        self,
        audio_files=[],
        tag_list=[],
        overwrite=False,
        parallel_workers=1,
        type_output=None,  # ["mp3", "wav", "ogg", "flac"]
    ):
        logger.info(f"Parallel workers: {str(parallel_workers)}")

        self.output_list = []

        if not self.model_config:
            raise ValueError("No model has been configured for inference")

        if isinstance(audio_files, str):
            audio_files = [audio_files]
        if isinstance(tag_list, str):
            tag_list = [tag_list]

        if not audio_files:
            raise ValueError("No audio found to convert")
        if not tag_list:
            tag_list = [list(self.model_config.keys())[-1]] * len(audio_files)

        if len(audio_files) > len(tag_list):
            logger.info("Extend tag list to match audio files")
            extend_number = len(audio_files) - len(tag_list)
            tag_list.extend([tag_list[0]] * extend_number)

        if len(audio_files) < len(tag_list):
            logger.info("Cut list tags")
            tag_list = tag_list[:len(audio_files)]

        tag_file_pairs = list(zip(tag_list, audio_files))
        sorted_tag_file = sorted(tag_file_pairs, key=lambda x: x[0])

        # Base params
        if not self.hu_bert_model:
            self.hu_bert_model = load_hu_bert(self.config, self.hubert_path)

        cache_params = None
        threads = []
        progress_bar = tqdm(total=len(tag_list), desc="Progress")
        for i, (id_tag, input_audio_path) in enumerate(sorted_tag_file):

            if id_tag not in self.model_config.keys():
                logger.info(
                    f"No configured model for {id_tag} with {input_audio_path}"
                )
                continue

            if (
                len(threads) >= parallel_workers
                or cache_params != id_tag
                and cache_params is not None
            ):

                self.run_threads(threads)
                progress_bar.update(len(threads))

                threads = []

            if cache_params != id_tag:

                self.model_config[id_tag]["result"] = []

                # Unload previous
                (
                    n_spk,
                    tgt_sr,
                    net_g,
                    pipe,
                    cpt,
                    version,
                    if_f0,
                    index_rate,
                    index,
                    big_npy,
                    inp_f0,
                ) = [None] * 11
                gc.collect()
                torch.cuda.empty_cache()

                # Model params
                params = self.model_config[id_tag]

                model_path = params["file_model"]
                f0_method = params["pitch_algo"]
                file_index = params["file_index"]
                index_rate = params["index_influence"]
                f0_file = params["file_pitch_algo"]

                # Load model
                (
                    n_spk,
                    tgt_sr,
                    net_g,
                    pipe,
                    cpt,
                    version
                ) = load_trained_model(model_path, self.config)
                if_f0 = cpt.get("f0", 1)  # pitch data

                # Load index
                if os.path.exists(file_index) and index_rate != 0:
                    try:
                        index = faiss.read_index(file_index)
                        big_npy = index.reconstruct_n(0, index.ntotal)
                    except Exception as error:
                        logger.error(f"Index: {str(error)}")
                        index_rate = 0
                        index = big_npy = None
                else:
                    logger.warning("File index not found")
                    index_rate = 0
                    index = big_npy = None

                # Load f0 file
                inp_f0 = None
                if os.path.exists(f0_file):
                    try:
                        with open(f0_file, "r") as f:
                            lines = f.read().strip("\n").split("\n")
                        inp_f0 = []
                        for line in lines:
                            inp_f0.append([float(i) for i in line.split(",")])
                        inp_f0 = np.array(inp_f0, dtype="float32")
                    except Exception as error:
                        logger.error(f"f0 file: {str(error)}")

                if "rmvpe" in f0_method:
                    if not self.model_pitch_estimator:
                        from infer_rvc_python.lib.rmvpe import RMVPE

                        logger.info("Loading vocal pitch estimator model")
                        if self.rmvpe_path is None:
                            self.rmvpe_path = ""
                        rm_local_path = "rmvpe.pt"
                        if os.path.exists(self.rmvpe_path):
                            rm_local_path = self.rmvpe_path
                        self.model_pitch_estimator = RMVPE(
                            rm_local_path,
                            is_half=self.config.is_half,
                            device=self.config.device
                        )

                    pipe.model_rmvpe = self.model_pitch_estimator

                cache_params = id_tag

            # self.infer(
            #     id_tag,
            #     params,
            #     # load model
            #     n_spk,
            #     tgt_sr,
            #     net_g,
            #     pipe,
            #     cpt,
            #     version,
            #     if_f0,
            #     # load index
            #     index_rate,
            #     index,
            #     big_npy,
            #     # load f0 file
            #     inp_f0,
            #     # output file
            #     input_audio_path,
            #     overwrite,
            #     type_output,
            # )

            thread = threading.Thread(
                target=self.infer,
                args=(
                    id_tag,
                    params,
                    # loaded model
                    n_spk,
                    tgt_sr,
                    net_g,
                    pipe,
                    cpt,
                    version,
                    if_f0,
                    # loaded index
                    index_rate,
                    index,
                    big_npy,
                    # loaded f0 file
                    inp_f0,
                    # audio file
                    input_audio_path,
                    overwrite,
                    type_output,
                )
            )

            threads.append(thread)

        # Run last
        if threads:
            self.run_threads(threads)

        progress_bar.update(len(threads))
        progress_bar.close()

        final_result = []
        valid_tags = set(tag_list)
        for tag in valid_tags:
            if (
                tag in self.model_config.keys()
                and "result" in self.model_config[tag].keys()
            ):
                final_result.extend(self.model_config[tag]["result"])

        return final_result

    def generate_from_cache(
        self,
        audio_data=None,  # str or tuple (<array data>,<int sampling rate>)
        tag=None,
        reload=False,
    ):

        if not self.model_config:
            raise ValueError("No model has been configured for inference")

        if not audio_data:
            raise ValueError(
                "An audio file or tuple with "
                "(<numpy data audio>,<sampling rate>) is needed"
            )

        # Base params
        if not self.hu_bert_model:
            self.hu_bert_model = load_hu_bert(self.config, self.hubert_path)

        if tag not in self.model_config.keys():
            raise ValueError(
                f"No configured model for {tag}"
            )

        now_data = self.model_config[tag]
        now_data["tag"] = tag

        if self.cache_model != now_data and not reload:

            # Unload previous
            self.model_vc = {}
            gc.collect()
            torch.cuda.empty_cache()

            model_path = now_data["file_model"]
            f0_method = now_data["pitch_algo"]
            file_index = now_data["file_index"]
            index_rate = now_data["index_influence"]
            f0_file = now_data["file_pitch_algo"]

            # Load model
            (
                self.model_vc["n_spk"],
                self.model_vc["tgt_sr"],
                self.model_vc["net_g"],
                self.model_vc["pipe"],
                self.model_vc["cpt"],
                self.model_vc["version"]
            ) = load_trained_model(model_path, self.config)
            self.model_vc["if_f0"] = self.model_vc["cpt"].get("f0", 1)

            # Load index
            if os.path.exists(file_index) and index_rate != 0:
                try:
                    index = faiss.read_index(file_index)
                    big_npy = index.reconstruct_n(0, index.ntotal)
                except Exception as error:
                    logger.error(f"Index: {str(error)}")
                    index_rate = 0
                    index = big_npy = None
            else:
                logger.warning("File index not found")
                index_rate = 0
                index = big_npy = None

            self.model_vc["index_rate"] = index_rate
            self.model_vc["index"] = index
            self.model_vc["big_npy"] = big_npy

            # Load f0 file
            inp_f0 = None
            if os.path.exists(f0_file):
                try:
                    with open(f0_file, "r") as f:
                        lines = f.read().strip("\n").split("\n")
                    inp_f0 = []
                    for line in lines:
                        inp_f0.append([float(i) for i in line.split(",")])
                    inp_f0 = np.array(inp_f0, dtype="float32")
                except Exception as error:
                    logger.error(f"f0 file: {str(error)}")

            self.model_vc["inp_f0"] = inp_f0

            if "rmvpe" in f0_method:
                if not self.model_pitch_estimator:
                    from infer_rvc_python.lib.rmvpe import RMVPE

                    logger.info("Loading vocal pitch estimator model")
                    if self.rmvpe_path is None:
                        self.rmvpe_path = ""
                    rm_local_path = "rmvpe.pt"
                    if os.path.exists(self.rmvpe_path):
                        rm_local_path = self.rmvpe_path
                    self.model_pitch_estimator = RMVPE(
                        rm_local_path,
                        is_half=self.config.is_half,
                        device=self.config.device
                    )

                self.model_vc["pipe"].model_rmvpe = self.model_pitch_estimator

            self.cache_model = copy.deepcopy(now_data)

        return self.infer(
            tag,
            now_data,
            # load model
            self.model_vc["n_spk"],
            self.model_vc["tgt_sr"],
            self.model_vc["net_g"],
            self.model_vc["pipe"],
            self.model_vc["cpt"],
            self.model_vc["version"],
            self.model_vc["if_f0"],
            # load index
            self.model_vc["index_rate"],
            self.model_vc["index"],
            self.model_vc["big_npy"],
            # load f0 file
            self.model_vc["inp_f0"],
            # output file
            audio_data,
            False,
            "array",
        )
