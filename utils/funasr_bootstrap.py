from __future__ import annotations

import importlib
import inspect


def bootstrap_funasr_runtime() -> None:
    _patch_inspect()
    _patch_torch_jit()

    char_tokenizer = importlib.import_module("funasr.tokenizer.char_tokenizer")
    importlib.import_module("funasr.frontends.wav_frontend")
    importlib.import_module("funasr.models.paraformer.cif_predictor")
    importlib.import_module("funasr.models.paraformer.decoder")
    importlib.import_module("funasr.models.sanm.encoder")
    importlib.import_module("funasr.models.specaug.specaug")
    importlib.import_module("funasr.models.paraformer.model")
    importlib.import_module("funasr.models.ct_transformer.model")
    importlib.import_module("funasr.models.fsmn_vad_streaming.model")

    if "load_seg_dict" not in char_tokenizer.CharTokenizer.__init__.__globals__:
        def load_seg_dict(seg_dict_file: str) -> dict[str, str]:
            seg_dict: dict[str, str] = {}
            with open(seg_dict_file, "r", encoding="utf8") as file:
                for line in file:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    seg_dict[parts[0]] = " ".join(parts[1:])
            return seg_dict

        char_tokenizer.load_seg_dict = load_seg_dict
        char_tokenizer.CharTokenizer.__init__.__globals__["load_seg_dict"] = load_seg_dict

    if "seg_tokenize" not in char_tokenizer.CharTokenizer.text2tokens.__globals__:
        def seg_tokenize(txt: list[str], seg_dict: dict[str, str]) -> list[str]:
            import re

            pattern = re.compile(r"([\u4E00-\u9FA5A-Za-z0-9])")
            out_txt = ""
            for word in txt:
                word = word.lower()
                if word in seg_dict:
                    out_txt += seg_dict[word] + " "
                else:
                    if pattern.match(word):
                        for char in word:
                            out_txt += seg_dict.get(char, "<unk>") + " "
                    else:
                        out_txt += "<unk> "
            return out_txt.strip().split()

        char_tokenizer.seg_tokenize = seg_tokenize
        char_tokenizer.CharTokenizer.text2tokens.__globals__["seg_tokenize"] = seg_tokenize


def _patch_inspect() -> None:
    if getattr(inspect, "_atc_safe_patch", False):
        return

    original_getfile = inspect.getfile
    original_getsourcelines = inspect.getsourcelines

    def safe_getfile(target: object) -> str:
        try:
            return original_getfile(target)
        except Exception:
            return f"<frozen:{getattr(target, '__module__', 'unknown')}>"

    def safe_getsourcelines(target: object) -> tuple[list[str], int]:
        try:
            return original_getsourcelines(target)
        except Exception:
            return ([], 0)

    inspect.getfile = safe_getfile
    inspect.getsourcelines = safe_getsourcelines
    inspect._atc_safe_patch = True


def _patch_torch_jit() -> None:
    import torch

    if getattr(torch.jit, "_atc_safe_patch", False):
        return

    original_script = torch.jit.script

    def safe_script(obj, *args, **kwargs):
        try:
            return original_script(obj, *args, **kwargs)
        except RuntimeError as exc:
            message = str(exc)
            if "Expected a single top-level function" in message or "could not get source code" in message:
                return obj
            raise

    torch.jit.script = safe_script
    torch.jit._atc_safe_patch = True
