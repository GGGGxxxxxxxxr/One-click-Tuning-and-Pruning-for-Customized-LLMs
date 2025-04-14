from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

_tok = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")
_mdl = AutoModelForCausalLM.from_pretrained(
    "epfl-llm/meditron-7b", torch_dtype=torch.float16
).cuda().eval()

def query_teacher(prompt: str, max_new_tokens: int = 128) -> str:
    with torch.no_grad():
        ids = _tok(prompt, return_tensors="pt").to("cuda")
        out = _mdl.generate(
            **ids, max_new_tokens=max_new_tokens,
            do_sample=True, top_p=0.95, temperature=0.7,
            eos_token_id=_tok.eos_token_id,
        )
    return _tok.decode(out[0], skip_special_tokens=True)
