## Start script
```bash	
.\koboldcpp.exe --model .\Lyra4-Gutenberg-12B.i1-Q6_K.gguf --usecublas --contextsize 8192 --flashattention --host 0.0.0.0 --gpulayers 99 --usemlock --quantkv 1 --blasbatchsize 2048
```

## Comments
Good adherence to prompt.

It refers to example dialog or scenarios if not stated as 'examples'.

Follows instructions better, over user's inputs, e.g. It doesnt necessarily answer the user if the system prompt suggests an eerie/mysterious character, not even explicitly stating it.

Also takes in the shortcomings of the instructions/examples, e.g. punctuation errors, etc.

The dialog sounds more realistic, less descriptive.

Responses are very varied, sometimes it can ramble on a bit too much.

