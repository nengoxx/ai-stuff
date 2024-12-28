# Improvements & fixes

Update: from 0.5 onwards the files have been refactored and moved around, this is the old version of the code. The updated code for 0.5 is in [open-webui/routers](../routers/README.md)

## These files have some light edits for extra functionality and improvements that wont make the cut for a pull request

### /audio/main.py

- Set the language for local transcriptions to English to prevent cross-language hallucinations.
- Added the VAD (voice activity detection) for local transcriptions to prevent annoying hallucinations like the usual "thanks you" and such.

### openai/main.py

- Hard coded a context limit (8k) to prevent the whole chat to be sent to external/paid APIs. This is obviously not ideal but in longer chats, it will let you use Groq's API (15k limit), as well as models with less effective context length, and save you money in paid APIs.
