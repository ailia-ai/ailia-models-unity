# Large Language Model (ailia SDK for Unity)

This folder demonstrates running quantized GGUF LLMs locally. The scene is `LargeLanguageModel/AiliaLargeLanguageModelSample.unity`, controller `LargeLanguageModel/AiliaLargeLanguageModelSample.cs`.

## Supported Models

- Gemma 2 2B (IT, Q4_K_M)
- Gemma 3 1B (IT, Q4_K_M)
- Gemma 3 4B (IT, Q4_K_M)
- Gemma 3n E2B (IT, Q4_K_M)
- Gemma 3n E4B (IT, Q4_K_M)
- Gemma 4 E2B (IT, Q4_K_M)
- Gemma 4 E4B (IT, Q4_K_M)
- Llama 3.2 3B Instruct (Q4_K_M)

## Script Behavior

- Downloads a `.gguf` model to `Application.temporaryCachePath` and opens it with `AiliaLLMModel` using `n_ctx` (8192 by default).
- Maintains chat history (`AiliaLLMChatMessage`), sets a system prompt, and streams tokens via `Generate()` while updating UI.
- Handles context reset when `ContextFull()` is true.

## Reusing in Your Own Code

- Initialize:
  - `var llm = new AiliaLLMModel(); llm.Create();`
  - `llm.Open(pathToGguf, 8192);`
- Chat:
  - Maintain `List<AiliaLLMChatMessage>` for system/user/assistant turns.
  - `llm.SetPrompt(messages);` then loop `llm.Generate(ref done)` and append `llm.GetDeltaText()`.

## Source Links

- Controller: `./AiliaLargeLanguageModelSample.cs`
