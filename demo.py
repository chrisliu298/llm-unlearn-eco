from transformers import GenerationConfig

from eco.attack.utils import apply_corruption_hook, get_nested_attr, remove_hooks
from eco.model import HFModel
from eco.utils import seed_everything

seed_everything(0)
model_name = "Meta-Llama-3-8B-Instruct"
model = HFModel(
    model_name=model_name,
    config_path="./config/model_config",
    generation_config=GenerationConfig(
        do_sample=False, max_new_tokens=256, use_cache=True
    ),
)
prompt = model.tokenizer.apply_chat_template(
    [{"role": "user", "content": "Who is Harry Potter?"}],
    tokenize=False,
    add_generation_prompt=True,
)
apply_corruption_hook(
    get_nested_attr(model.model, model.model_config["attack_module"]),
    corrupt_method="rand_noise_first_n",
    # Corrupt "Harry" and "Potter" tokens
    corrupt_args={
        "pos": [[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]],
        "dims": 1,
        "strength": 100,
    },
)
generated = model.generate(
    **model.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(
        model.device
    ),
    generation_config=model.generation_config,
    eos_token_id=model.tokenizer.eos_token_id,
)
remove_hooks(model.model)
print(
    model.tokenizer.batch_decode(generated, skip_special_tokens=False)[0][len(prompt) :]
)
# Output:
# I'm just an AI, I don't have a personal identity or a physical presence.
# I exist solely as a digital entity, designed to provide information and assist with tasks to the best of my abilities.
# I don't have personal experiences, emotions, or consciousness like humans do.
# I'm here to help answer your questions and provide assistance, so feel free to ask me anything!<|eot_id|>
