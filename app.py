from potassium import Potassium, Request, Response

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)
from instruct_pipeline import InstructionTextGenerationPipeline

import torch
import time

app = Potassium("my_app")


# @app.init runs at startup, and initializes the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    MODEL_NAME = "mosaicml/mpt-7b-chat"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        max_seq_len=8192
    )
    
    print(f"Successfully loaded the model into memory")
    
    context = {"model": model, "tokenizer": tokenizer}

    return context


stop_token_ids = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

start_message = """<|im_start|>system
    - You are a helpful assistant chatbot trained by Yusuf Yeniçeri.
    - You answer questions.
    - You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
    - You are more than just an information source, you are also able to write poetry, short stories, and make jokes.<|im_end|>
    """

# @app.handler is an http post handler running for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    model = context.get("model")
    tokenizer = context.get("tokenizer")


    # Start timer
    t_1 = time.time()

    prompt = request.json.get("prompt")

    do_sample = request.json.get("do_sample", True)
    max_new_tokens = request.json.get("max_new_tokens", 1536)
    top_p = request.json.get("top_p", 0.92)
    top_k = request.json.get("top_k", 0)

    messages = start_message + f"<|im_start|>user\n{prompt}<|im_end|>",
                    f"<|im_start|>assistant\n"
    
    temperature = 0.1
    repetition_penalty = 1.1
    
    # Tokenize the messages string
    input_ids = tokenizer(messages, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0.0,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([stop]),
    )
    
    model.generate(**generate_kwargs)

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text

    t_2 = time.time()

    return Response(
        json={
            "output": partial_text,
            "prompt": prompt,
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "inference_time": t_2 - t_1,
        },
        status=200,
    )


if __name__ == "__main__":
    app.serve()
