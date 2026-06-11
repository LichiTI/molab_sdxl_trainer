from __future__ import annotations

import gzip
from dataclasses import dataclass
import html
import logging
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

try:
    import ftfy  # type: ignore
except Exception:  # pragma: no cover
    ftfy = None

try:
    import regex as re
except Exception:  # pragma: no cover
    import re  # type: ignore

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).with_name("assets")
SDXL_BASE_CONFIG_PATH = ASSETS_DIR / "sd_xl_base.yaml"
OPENCLIP_BPE_PATH = ASSETS_DIR / "bpe_simple_vocab_16e6.txt.gz"
DEFAULT_CONTEXT_LENGTH = 77
DEFAULT_BETA_START = 0.00085
DEFAULT_BETA_END = 0.012
DEFAULT_TIMESTEPS = 1000


class TokenizerOutput(SimpleNamespace):
    input_ids: torch.LongTensor


@lru_cache(maxsize=1)
def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    extra = 0
    for value in range(2**8):
        if value not in bs:
            bs.append(value)
            cs.append(2**8 + extra)
            extra += 1
    return dict(zip(bs, (chr(value) for value in cs)))


@lru_cache(maxsize=1)
def _bpe_merges() -> tuple[tuple[str, str], ...]:
    with gzip.open(OPENCLIP_BPE_PATH, "rt", encoding="utf-8") as handle:
        merges = handle.read().splitlines()
    merges = merges[1 : 49152 - 256 - 2 + 1]
    return tuple(tuple(line.split()) for line in merges if line)


def _basic_clean(text: str) -> str:
    if ftfy is not None:
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def _whitespace_clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _get_pairs(word: Sequence[str]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    previous = word[0]
    for current in word[1:]:
        pairs.add((previous, current))
        previous = current
    return pairs


class _OpenClipSimpleTokenizer:
    def __init__(self, bpe_path: Path):
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {value: key for key, value in self.byte_encoder.items()}
        self.merges = _bpe_merges()
        vocab = list(self.byte_encoder.values())
        vocab.extend(token + "</w>" for token in list(self.byte_encoder.values()))
        vocab.extend("".join(pair) for pair in self.merges)
        vocab.extend(["<start_of_text>", "<end_of_text>"])
        self.encoder = {token: index for index, token in enumerate(vocab)}
        self.decoder = {index: token for token, index in self.encoder.items()}
        self.bpe_ranks = {merge: rank for rank, merge in enumerate(self.merges)}
        self.cache = {"<start_of_text>": "<start_of_text>", "<end_of_text>": "<end_of_text>"}
        self.pattern = re.compile(
            r"<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
            re.IGNORECASE,
        )
        self.vocab_size = len(self.encoder)
        self.bos_token_id = self.encoder["<start_of_text>"]
        self.eos_token_id = self.encoder["<end_of_text>"]
        self.pad_token_id = 0

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                    new_word.extend(word[index:next_index])
                    index = next_index
                except ValueError:
                    new_word.extend(word[index:])
                    break
                if index < len(word) - 1 and word[index] == first and word[index + 1] == second:
                    new_word.append(first + second)
                    index += 2
                else:
                    new_word.append(word[index])
                    index += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text: str) -> list[int]:
        cleaned = _whitespace_clean(_basic_clean(text)).lower()
        bpe_tokens: list[int] = []
        for token in re.findall(self.pattern, cleaned):
            token = "".join(self.byte_encoder[byte] for byte in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[piece] for piece in self.bpe(token).split(" "))
        return bpe_tokens


class OpenClipTokenizerAdapter:
    def __init__(self, context_length: int = DEFAULT_CONTEXT_LENGTH):
        self.model_max_length = context_length
        self._tokenizer = _OpenClipSimpleTokenizer(OPENCLIP_BPE_PATH)
        self.bos_token_id = self._tokenizer.bos_token_id
        self.eos_token_id = self._tokenizer.eos_token_id
        self.pad_token_id = self._tokenizer.pad_token_id
        self.vocab_size = self._tokenizer.vocab_size
        self.added_tokens_encoder: dict[str, int] = {}

    def __len__(self) -> int:
        return self.vocab_size

    def tokenize(self, text: str, **_: Any) -> list[str]:
        return [str(token_id) for token_id in self._tokenizer.encode(str(text or ""))]

    def __call__(
        self,
        texts: str | Sequence[str],
        padding: str = "max_length",
        max_length: int | None = None,
        truncation: bool = True,
        return_tensors: str = "pt",
        **_: Any,
    ) -> TokenizerOutput:
        if return_tensors != "pt":
            raise ValueError("OpenClipTokenizerAdapter only supports return_tensors=\"pt\"")
        if isinstance(texts, str):
            texts = [texts]
        encoded: list[list[int]] = []
        for text in texts:
            token_ids = [self.bos_token_id] + self._tokenizer.encode(text) + [self.eos_token_id]
            encoded.append(token_ids)
        if padding == "longest" and max_length is None:
            context_length = max((len(token_ids) for token_ids in encoded), default=0)
        elif padding == "max_length":
            context_length = max_length or self.model_max_length
        else:
            raise ValueError("OpenClipTokenizerAdapter only supports padding=\"max_length\" or \"longest\"")
        input_ids = torch.zeros((len(texts), context_length), dtype=torch.long)
        for row, token_ids in enumerate(encoded):
            if len(token_ids) > context_length:
                if not truncation:
                    raise ValueError(f"Prompt is too long for context length {context_length}")
                token_ids = token_ids[:context_length]
                token_ids[-1] = self.eos_token_id
            input_ids[row, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
        return TokenizerOutput(input_ids=input_ids)


def _require_assets() -> None:
    missing = [str(path) for path in (SDXL_BASE_CONFIG_PATH, OPENCLIP_BPE_PATH) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing bundled SDXL single-file loader assets: "
            + ", ".join(missing)
            + ". The OpenCLIP BPE vocabulary is required when loading SDXL .safetensors/.ckpt files directly. "
            + "Please update from a build that includes backend/core/lulynx_trainer/assets/."
        )


def _max_block_index(checkpoint: dict[str, torch.Tensor], prefix: str) -> int:
    max_index = -1
    for key in checkpoint:
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        block_index_text = suffix.split(".", 1)[0]
        if block_index_text.isdigit():
            max_index = max(max_index, int(block_index_text))
    if max_index < 0:
        raise KeyError(f"Could not infer layer count from checkpoint prefix: {prefix}")
    return max_index


def _build_clip_l_config(checkpoint: dict[str, torch.Tensor]) -> CLIPTextConfig:
    prefix = "conditioner.embedders.0.transformer."
    token_embedding = checkpoint[prefix + "text_model.embeddings.token_embedding.weight"]
    position_embedding = checkpoint[prefix + "text_model.embeddings.position_embedding.weight"]
    fc1_weight = checkpoint[prefix + "text_model.encoder.layers.0.mlp.fc1.weight"]
    hidden_size = position_embedding.shape[1]
    num_hidden_layers = _max_block_index(checkpoint, prefix + "text_model.encoder.layers.") + 1
    return CLIPTextConfig(
        vocab_size=token_embedding.shape[0],
        hidden_size=hidden_size,
        intermediate_size=fc1_weight.shape[0],
        projection_dim=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=max(1, hidden_size // 64),
        max_position_embeddings=position_embedding.shape[0],
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        bos_token_id=49406,
        eos_token_id=49407,
        pad_token_id=0,
    )


def _build_openclip_bigg_config(checkpoint: dict[str, torch.Tensor]) -> CLIPTextConfig:
    prefix = "conditioner.embedders.1.model."
    token_embedding = checkpoint[prefix + "token_embedding.weight"]
    position_embedding = checkpoint[prefix + "positional_embedding"]
    fc1_weight = checkpoint[prefix + "transformer.resblocks.0.mlp.c_fc.weight"]
    projection = checkpoint[prefix + "text_projection"]
    hidden_size = token_embedding.shape[1]
    num_hidden_layers = _max_block_index(checkpoint, prefix + "transformer.resblocks.") + 1
    return CLIPTextConfig(
        vocab_size=token_embedding.shape[0],
        hidden_size=hidden_size,
        intermediate_size=fc1_weight.shape[0],
        projection_dim=projection.shape[0],
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=max(1, hidden_size // 64),
        max_position_embeddings=position_embedding.shape[0],
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        bos_token_id=49406,
        eos_token_id=49407,
        pad_token_id=0,
    )


def _build_training_scheduler() -> Any:
    from diffusers import DDPMScheduler

    return DDPMScheduler(
        num_train_timesteps=DEFAULT_TIMESTEPS,
        beta_start=DEFAULT_BETA_START,
        beta_end=DEFAULT_BETA_END,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
        clip_sample=False,
    )


def _cast_pipe_components(pipe: Any, torch_dtype: torch.dtype) -> None:
    for module_name in ("unet", "text_encoder", "text_encoder_2", "vae"):
        module = getattr(pipe, module_name, None)
        if module is not None:
            module.to(dtype=torch_dtype)




@dataclass
class SingleFileSDXLComponents:
    unet: Any
    text_encoder_1: Any
    text_encoder_2: Any
    vae: Any
    tokenizer_1: Any
    tokenizer_2: Any
    noise_scheduler: Any


class SDXLSingleFileLoader:
    def __init__(self, dtype: torch.dtype = torch.float16):
        self.dtype = dtype

    def load(self, checkpoint_path: str | Path) -> SingleFileSDXLComponents:
        components = load_sdxl_single_file_components(checkpoint_path, torch_dtype=self.dtype)
        return SingleFileSDXLComponents(**components)


def _infer_model_type_fallback(checkpoint: dict[str, torch.Tensor]) -> str:
    """Best-effort fallback for older diffusers builds without single_file_utils."""
    if (
        "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight" in checkpoint
        and "conditioner.embedders.1.model.text_projection" in checkpoint
    ):
        return "xl_base"
    if "model.diffusion_model.input_blocks.0.0.weight" in checkpoint:
        return "v1"
    return "unknown"


def _infer_diffusers_model_type_compat(checkpoint: dict[str, torch.Tensor]) -> str:
    try:
        from diffusers.loaders.single_file_utils import infer_diffusers_model_type

        return infer_diffusers_model_type(checkpoint)
    except Exception:
        return _infer_model_type_fallback(checkpoint)


def load_sdxl_single_file_components(checkpoint_path: str | Path, torch_dtype: torch.dtype = torch.float16) -> dict[str, Any]:
    _require_assets()
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix.lower() not in {".safetensors", ".ckpt", ".pt"}:
        raise ValueError(f"Unsupported single-file checkpoint format: {checkpoint_path}")

    from diffusers import StableDiffusionXLPipeline
    from diffusers.models.modeling_utils import load_state_dict
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

    logger.info("Loading SDXL single-file checkpoint with offline native loader: %s", checkpoint_path)

    checkpoint = load_state_dict(str(checkpoint_path))
    while isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model_type = _infer_diffusers_model_type_compat(checkpoint)
    if model_type != "xl_base":
        raise RuntimeError(f"single_file_loader currently supports SDXL base checkpoints only, got: {model_type}")

    tokenizer_1 = OpenClipTokenizerAdapter()
    tokenizer_2 = OpenClipTokenizerAdapter()
    text_encoder_1 = CLIPTextModel(_build_clip_l_config(checkpoint))
    text_encoder_2 = CLIPTextModelWithProjection(_build_openclip_bigg_config(checkpoint))

    pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path_or_dict=str(checkpoint_path),
        original_config_file=str(SDXL_BASE_CONFIG_PATH),
        pipeline_class=StableDiffusionXLPipeline,
        from_safetensors=checkpoint_path.suffix.lower() == ".safetensors",
        device="cpu",
        local_files_only=True,
        load_safety_checker=False,
        tokenizer=tokenizer_1,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder_1,
        text_encoder_2=text_encoder_2,
    )

    _cast_pipe_components(pipe, torch_dtype)

    return {
        "unet": pipe.unet,
        "text_encoder_1": pipe.text_encoder,
        "text_encoder_2": pipe.text_encoder_2,
        "vae": pipe.vae,
        "tokenizer_1": tokenizer_1,
        "tokenizer_2": tokenizer_2,
        "noise_scheduler": _build_training_scheduler(),
    }
