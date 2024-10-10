## Specific settings & tips

### 12GB:
It's a 22B parameter model, but fits at a decent quantization.

**IQ3_XS** - 8k context / 16k context with --quantkv 2
**IQ3_M** - 8k context with --quantkv 1

```bash
.\koboldcpp.exe --model .\Mistral-Small-Instruct-2409-IQ3_M.gguf --usecublas --contextsize 8192 --host 0.0.0.0 --gpulayers 99 --flashattention --usemlock --quantkv 1 --blasbatchsize 2048
```