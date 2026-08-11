# autocodegen.py
from __future__ import annotations


try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
except ImportError:  # optional dependency: pip install transformers
    GPT2Tokenizer = None
    GPT2LMHeadModel = None

def generate_code_from_description(problem_desc):
    """
    This function generates quantum code from the given problem description
    using an AI model (like GPT).
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    inputs = tokenizer.encode(problem_desc, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code
