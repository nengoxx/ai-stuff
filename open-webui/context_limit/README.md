# Enable context limit for external APIs (pre 0.5, updated version in routers folder)

## Editing the open-webui/apps/openai/main.py file

You will need to install tiktoken in the venv, then add the following code:

```python
import tiktoken

def count_tokens(messages, model):
    encoding = tiktoken.get_encoding("gpt2")
    #encoding = tiktoken.encoding_for_model(model) #TODO:extract external model name??
    total_tokens = 0

    for message in messages:
        total_tokens += len(encoding.encode(message['role']))
        total_tokens += len(encoding.encode(message['content']))

    return total_tokens

def trim_context(messages, model, max_tokens):

    #Leave the system role.
    preserved_messages = [msg for msg in messages if msg['role'] not in ['user', 'assistant']]
    trimmable_messages = [msg for msg in messages if msg['role'] in ['user', 'assistant']]

    while count_tokens(preserved_messages + trimmable_messages, model) > max_tokens:
        trimmable_messages.pop(0)

    return preserved_messages + trimmable_messages

```

Then inside the /chat/completions route:

```python
max_tokens = api_config.get("max_context_tokens", 12288)  #Hardcoded max tokens.
if "messages" in payload:
  payload["messages"] = trim_context(payload["messages"], payload["model"], max_tokens)
```

This basically hard-codes the maximum tokens sent to the API (except for the system prompt). This will let you have more agency over the context size in long chats. For APIs like Groq's or for models that hallucinate over long context.
