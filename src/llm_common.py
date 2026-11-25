"""Shared LLM helpers: load a (4-bit) chat model and get label probabilities."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_llm(name: str, four_bit: bool = True):
    kw = {}
    if four_bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        kw["device_map"] = "auto"
    elif torch.cuda.is_available():
        kw["torch_dtype"] = torch.float16
        kw["device_map"] = "auto"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
    return tok, model


def chat_prompt(tok, user_msg: str, system: str = "You are a careful assistant.") -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    try:   # Qwen3: switch off the long "thinking" mode; other templates ignore the flag
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except Exception:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def classify_msg(text: str, labels: list) -> str:
    return (f"Text (Egyptian Arabic / Arabizi / English, possibly mixed):\n{text}\n\n"
            f"What is the sentiment of this text? Answer with exactly one word from: "
            f"{', '.join(labels)}.")


class LabelScorer:
    """P(label | text) from the next-token probabilities of each label's first token."""

    def __init__(self, tok, model, labels):
        self.tok, self.model, self.labels = tok, model, labels
        self.first = [tok.encode(l, add_special_tokens=False)[0] for l in labels]
        assert len(set(self.first)) == len(labels), "label words share a first token; rename labels"

    @torch.no_grad()
    def probs(self, text: str) -> list:
        prompt = chat_prompt(self.tok, classify_msg(text, self.labels))
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        logits = self.model(**ids).logits[0, -1]
        return torch.softmax(logits[self.first].float(), -1).tolist()
