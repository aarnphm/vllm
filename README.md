A stripped down version of vLLM that only contains core functionalities.

This branch removes features such as accelerator support, LoRA, prompt adapters, CPU support, Ray

Features and models reserved on this branch on top of core functionalities:
- GPU executor
- Spec decode
- BlockManager V2

Only support serving Llama models


> [!NOTE]
> This is for educational purpose only.

Request for OpenAI endpoints:

```json
{
  "messages": [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"}
  ],
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "temperature": 0.87,
  "n": 2
}
```
