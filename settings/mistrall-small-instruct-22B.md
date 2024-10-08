## Specific settings & tips

### 12GB:
**IQ3_XS** - 8k context 
**IQ3_M** - 8k context with --quantkv 1

```bash
.\koboldcpp.exe --model .\Mistral-Small-Instruct-2409-IQ3_M.gguf --usecublas --contextsize 8192 --host 0.0.0.0 --gpulayers 99 --flashattention --quantkv 1
```