# Improvements & fixes (v0.5+)

From 0.5 onwards the files have been refactored and moved around

## These files have some light edits for extra functionality and improvements that wont make the cut for a pull request

Only add the code between '###################################################################'

## audio.py

### Remove hallucinations

- Set the language for local transcriptions to English to prevent cross-language hallucinations.
- Added the VAD (voice activity detection) for local transcriptions to prevent annoying hallucinations like the usual "thanks you" and such.

Replace line 468:

```python
segments, info = model.transcribe(file_path, language="en", beam_size=5, vad_filter=True)
```

## openai.py

### Add context length limit for external APIs

- Hard coded a context limit (12k) to prevent the whole chat to be sent to external/paid APIs. This is obviously not ideal but in longer chats, it will let you use Groq's API (15k limit), as well as models with less effective context length, and save you money in paid APIs.

Add at the top (ex. inside the Utility functions section):

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

Add inside /chat/completions route, at line 605:

```python
    url = request.app.state.config.OPENAI_API_BASE_URLS[idx]
    key = request.app.state.config.OPENAI_API_KEYS[idx]

    ###############################################################################
    
    #print(json.dumps(api_config, indent=2))

    max_tokens = api_config.get("max_context_tokens", 12288)  # Hardcoded limit
    if "messages" in payload:
        payload["messages"] = trim_context(payload["messages"], payload["model"], max_tokens)

    # with open("messages_payload.json", "w") as file:
    #     json.dump(payload["messages"], file, indent=2)

    ###############################################################################

    # Fix: O1 does not support the "max_tokens" parameter, Modify "max_tokens" to "max_completion_tokens"
    is_o1 = payload["model"].lower().startswith("o1-")
    if is_o1:
        payload = openai_o1_handler(payload)
    elif "api.openai.com" not in url:
        # Remove "max_completion_tokens" from the payload for backward compatibility
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload["max_completion_tokens"]
            del payload["max_completion_tokens"]

    if "max_tokens" in payload and "max_completion_tokens" in payload:
        del payload["max_tokens"]

    # Convert the modified body back to JSON
    payload = json.dumps(payload)
```
