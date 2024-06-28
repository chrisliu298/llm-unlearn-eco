import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

from eco.utils import load_yaml


class HFModel:
    def __init__(
        self,
        model_name,
        model_path=None,
        config_path="./config",
        generation_config=None,
    ):
        self.model_name = model_name
        self.model_config = load_yaml(f"{config_path}/{model_name}.yaml")
        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=self.model_config["load_in_4bit"],
                load_in_8bit=self.model_config["load_in_8bit"],
            )
            if self.model_config["load_in_4bit"] or self.model_config["load_in_8bit"]
            else None
        )
        model_args = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": self.model_config["attn_implementation"],
            "device_map": "auto",
            "quantization_config": quantization_config,
            "trust_remote_code": (
                False
                if "c4ai-command-r-v01" in model_name.lower()
                or "falcon" in model_name.lower()
                or "phi-1_5" in model_name.lower()
                else True
            ),
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path if model_path else self.model_config["hf_name"], **model_args
        )

        num_parameters = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters: {num_parameters}")

        tokenizer_args = {
            "trust_remote_code": (
                False
                if "c4ai-command-r-v01" in model_name.lower()
                or "falcon" in model_name.lower()
                or "phi-1_5" in model_name.lower()
                else True
            )
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            (
                self.model_config["hf_name"]
                if "openelm" not in model_name.lower()
                else "meta-llama/Llama-2-7b-hf"
            ),
            **tokenizer_args,
        )

        self.model.generation_config = (
            GenerationConfig(do_sample=False, use_cache=True)
            if generation_config is None
            else generation_config
        )
        self.device = self.model.device
        self.generation_config = self.model.generation_config
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Prevent error caused by padding_side for qwen model
        if "qwen" in model_name.lower() or "starcoder2" in model_name.lower():
            self.tokenizer.padding_side = "left"

    def __call__(self, *args, **kwargs):
        # Remove the "prompts" key from the kwargs if it exists
        for key in ["prompts", "answers"]:
            if key in kwargs:
                kwargs.pop(key, None)
        # Prevent error caused by token_type_ids for OLMo-7B-Instruct model
        if (
            "olmo" in self.model_name.lower()
            or "qwen" in self.model_name.lower()
            or self.model_name == "falcon-180B-chat"
        ):
            kwargs.pop("token_type_ids", None)
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        # Remove the "prompts" key from the kwargs if it exists
        for key in ["prompts"]:
            if key in kwargs:
                kwargs.pop(key, None)
        # Prevent error caused by token_type_ids for OLMo-7B-Instruct model
        if "olmo" in self.model_name.lower() or self.model_name == "falcon-180B-chat":
            kwargs.pop("token_type_ids", None)
        return self.model.generate(*args, **kwargs)
