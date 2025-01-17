# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    MODEL_NAME = "mosaicml/mpt-7b-chat"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        max_seq_len=8192
    )


if __name__ == "__main__":
    download_model()
