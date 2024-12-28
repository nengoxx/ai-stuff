# Improvements & fixes

Only add the code between ###################################################################

Added below the imports:

```python
###############################################################################

import tiktoken  # Install this for token counting

# Function to calculate the total tokens in the conversation
def count_tokens(messages, model):
    encoding = tiktoken.get_encoding("gpt2")
    #encoding = tiktoken.encoding_for_model(model)
    total_tokens = 0

    for message in messages:
        total_tokens += len(encoding.encode(message['role']))
        total_tokens += len(encoding.encode(message['content']))

    return total_tokens

def trim_context(messages, model, max_tokens):
    """
    Only trims the chat history.
    """
    preserved_messages = [msg for msg in messages if msg['role'] not in ['user', 'assistant']]
    trimmable_messages = [msg for msg in messages if msg['role'] in ['user', 'assistant']]

    # Trim the trimmable messages if they exceed the token limit
    while count_tokens(preserved_messages + trimmable_messages, model) > max_tokens:
        trimmable_messages.pop(0)  # Remove the oldest message with "user" or "assistant" role

    return preserved_messages + trimmable_messages

###############################################################################
```

Added inside /chat/completions route, at line 598:

```python
    # Add user info to the payload if the model is a pipeline
    if "pipeline" in model and model.get("pipeline"):
        payload["user"] = {
            "name": user.name,
            "id": user.id,
            "email": user.email,
            "role": user.role,
        }

    url = app.state.config.OPENAI_API_BASE_URLS[idx]
    key = app.state.config.OPENAI_API_KEYS[idx]

###############################################################################
    
    #print(json.dumps(api_config, indent=2))

    max_tokens = api_config.get("max_context_tokens", 8192)  # Hardcoded limit
    if "messages" in payload:
        payload["messages"] = trim_context(payload["messages"], payload["model"], max_tokens)

    # with open("messages_payload.json", "w") as file:
    #     json.dump(payload["messages"], file, indent=2)

###############################################################################


    # Fix: O1 does not support the "max_tokens" parameter, Modify "max_tokens" to "max_completion_tokens"
    is_o1 = payload["model"].lower().startswith("o1-")
    # Change max_completion_tokens to max_tokens (Backward compatible)
    if "api.openai.com" not in url and not is_o1:
        if "max_completion_tokens" in payload:
            # Remove "max_completion_tokens" from the payload
            payload["max_tokens"] = payload["max_completion_tokens"]
            del payload["max_completion_tokens"]
    else:
        if is_o1 and "max_tokens" in payload:
            payload["max_completion_tokens"] = payload["max_tokens"]
            del payload["max_tokens"]
        if "max_tokens" in payload and "max_completion_tokens" in payload:
            del payload["max_tokens"]
```
