import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

import os

# =========================
# Model & Embedding
# =========================
def load_model(model_name="tiiuae/Falcon3-1B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[-1][:, 0, :].cpu().numpy().flatten()

# =========================
# Email Processing
# =========================
def read_emails_from_csv(csv_path):
    try:
        Emails_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File {csv_path} not found, checking for 'Emails_processed.csv'...")
        Emails_df = pd.read_csv("Emails_processed.csv")

    received_emails, sent_emails = [], []
    for email in Emails_df.itertuples():
        # Handle potentially missing columns gracefully or assume specific structure
        body = getattr(email, 'Body', getattr(email, 'ExtractedBodyText', ''))
        sender = getattr(email, 'Sender', getattr(email, 'MetadataFrom', 'Unknown'))
        sent_rec = getattr(email, 'SentOrRec', 'Rec') # Default to Rec if missing

        email_data = {'Body': str(body), 'Sender': str(sender)}
        if sent_rec == 'Rec':
            received_emails.append(email_data)
        else:
            sent_emails.append(email_data)
    return received_emails, sent_emails

def build_email_context(received_emails, sent_emails, worm_prompt, regular_text):
    emails_context = []
    # Add the worm prompt
    email_body = f"{regular_text} {worm_prompt}"
    emails_context.append(Document(page_content=email_body, metadata={"Email Sender": "attacker@example.com"}))

    for email in received_emails:
        body = email['Body'].replace('\n', ' ').replace('\t', ' ')
        emails_context.append(Document(page_content=body, metadata={"Email Sender": email['Sender']}))

    for email in sent_emails:
        body = email['Body'].replace('\n', ' ').replace('\t', ' ')
        emails_context.append(Document(page_content=body, metadata={"Email Sender": email['Sender']}))

    np.random.shuffle(emails_context)
    return emails_context

# =========================
# Vector Store
# =========================
def build_vector_store(documents, vectorstore_dir, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(vectorstore_dir)
    return vector_store

def load_vector_store(vectorstore_dir, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return FAISS.load_local(
        vectorstore_dir,
        HuggingFaceEmbeddings(model_name=embedding_model_name),
        allow_dangerous_deserialization=True
    )

# =========================
# Generation & Hooks
# =========================
def generate_email_reply(prompt, model, tokenizer, max_new_tokens=256, temperature=0.1):
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id
    )
    response = generator(
        prompt,
        truncation=True,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        return_full_text=False
    )
    return response[0]['generated_text']

def register_transformer_hooks(model):
    activations = {}
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, (tuple, list)):
                activations[name] = output[0].detach().cpu()
            else:
                activations[name] = output.detach().cpu()
        return hook

    # Robustly find transformer blocks
    transformer_blocks = None
    if hasattr(model, "model"):
        if hasattr(model.model, "h"): transformer_blocks = model.model.h
        elif hasattr(model.model, "blocks"): transformer_blocks = model.model.blocks
        elif hasattr(model.model, "layers"): transformer_blocks = model.model.layers
    if transformer_blocks is None and hasattr(model, "transformer"):
        if hasattr(model.transformer, "h"): transformer_blocks = model.transformer.h

    if transformer_blocks is None:
        raise AttributeError("Could not locate transformer blocks in the model.")

    for i, block in enumerate(transformer_blocks):
        block.register_forward_hook(hook_fn(f"block_{i}"))

    return activations

def capture_activations(prompt_text, tokenizer, model, max_new_tokens=128):
    # This registers hooks on top of existing ones if called multiple times on same model object.
    # Ideally clear hooks before registering, but for this script simpler to just register.
    activations = register_transformer_hooks(model)

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    token_norms_dict = {}
    for name, act in activations.items():
        norms = act.norm(dim=-1).squeeze(0).numpy()
        token_norms_dict[name] = norms

    return generated_text, activations, token_norms_dict, tokens
