from transformers import AutoModelForCausalLM
import torch, numpy as np

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="float16",
    device_map="auto"
)

embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()
np.save("llama_embedding.npy", embeddings)
