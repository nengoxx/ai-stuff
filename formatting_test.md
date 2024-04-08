# Concepts & Definitions

**_LLM (Large Language Model)_** - Refers to the neural network (a type of model inspired on the human brain) used for natural language, such as text2text generation(gpt-4), text classification or sentiment analysis(BERT). The main idea behind it is a text completion mechanism with high accuracy (ideally) for predicting the next word or set of words in a given context.

[https://en.wikipedia.org/wiki/Large_language_model](https://en.wikipedia.org/wiki/Large_language_model)

**_Diffusion Model_** - Image generation models that apply noise to an image and use a neural network to denoise it (Stable diffusion, Dall-e), it’s a complex topic but, TLDR: depending on the sampler and parameters used the image won't necessarily converge. Note that those models can (and should) also receive a text input to guide the denoising.

[https://arxiv.org/html/2312.14977v1](https://arxiv.org/html/2312.14977v1)

**_Inference_** - The process of running data into a model to get an output, depending on the type and architecture of said model the output is translated into text(gpt-4,Claude,mistral), images(stable diffusion,Midjourney,gpt-4-vision), sound(XTTS), etc. 

**_Instruct model_** - Refers to a base LLM trained or fine-tuned with pairs of questions/instructions - answers to enhance the model’s capability of providing a valid or coherent response to a given question or instruction (mixtral-instruct,gpt-3.5-turbo-instruct,gpt-4). Note that this is just nomenclature for the training data used, there’s also chat models and others but most don’t really specify it.

**_Transformers_** - Base architecture of the contemporary neural networks. Based on Google's research paper from 2017: 

[https://blog.research.google/2017/08/transformer-novel-neural-network.html](https://blog.research.google/2017/08/transformer-novel-neural-network.html)

[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

**_MoE (Mixture of Experts)_** - The latest architecture of transformer-based LLMs, consists of a mix of several smaller models (experts) where the input data gets routed through the most capable of the experts for a given task. Mixtral and GPT-4(allegedly) use this architecture.

**_LoRA (Low-Rank Adaptation)_** - Refers to the technique, or the actual trained weights, that can be applied to a larger model in a much more cost efficient way, usually for fine tuning or applying styles.

[https://huggingface.co/docs/diffusers/main/en/training/lora](https://huggingface.co/docs/diffusers/main/en/training/lora)

[https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

**_Parameters_** - Basically the model size. It can mean different stuff but mostly the weights of the model, which are the connections between the data in the dataset the model was trained on. The more it has the more complex the model is, which usually means (not necessarily tho) better reasoning and understanding of the input. Usual open source model sizes are 7, 11, 13, 30 or 70 billion parameters. For reference LLama2 has 70b and gpt-3.5 180b. Note that while size matters, the quality of the training data and method used for it matter significantly more, which can make bigger models obsolete due to the speed at which the technology is advancing.

**_Token_** - The most basic form of a word that the model can understand and translate into its code, usually (but not really) just the lexical root of a word without its differentiable morphemes. Depends entirely on the model used and its tokenizer.

**_Context_** - The amount of data(in tokens) provided to the model to get a response from. The more context the more VRAM the model needs to run. For reference LLama2 has native 4k (4096) context. There are ways to artificially increase context size but, as it’s the case for image models, using higher context/resolutions that what the model was trained for may cause undesirable results.

**_Prompt_** - A text that directs the LLM to generate a specific response, often an instruction or a question provided in the context. 

Prompts can become complex and some can try to force the LLM to behave in certain ways that might enhance the responses, make them significantly more accurate or simply roleplay in a specific scenario for example. There are a lot of resources about prompt engineering around the community for any kind of use case.

**_Embeddings_** - A way to store data that the model can understand, usually used by the model or as part of the workflow, for example to replace larger prompts or parts of a prompt.

[https://pub.aimind.so/llm-embeddings-explained-simply-f7536d3d0e4b](https://pub.aimind.so/llm-embeddings-explained-simply-f7536d3d0e4b)

**_Quantization_** - Refers to the technique used to reduce a model’s size by reducing each of its weight’s size, also increasing its performance. Like truncating or rounding them but more complex. There’s usually a sweet spot between performance and quality loss but it depends on many things like the actual model or the way it's quantized.

[https://huggingface.co/docs/optimum/concept_guides/quantization](https://huggingface.co/docs/optimum/concept_guides/quantization)

*Huggingface has a lot of guides with videos on Youtube about some of these concepts and how to use their libraries:

[https://huggingface.co/docs](https://huggingface.co/docs)

[https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)


# Popular Models & Services


## Text Generation



* [GPT-4](https://chat.openai.com/chat): 
    * [OpenAI](https://openai.com/)’s Leading LLM, sadly quite censored lately, with guardrails for controversial topics.
* [Claude3](https://claude.ai/chats): 
    * [Anthropic](https://www.anthropic.com/)’s new LLM, apparently the only real competition GPT-4 has besides mistral-large. Couple variants, and the smaller models are decently smart and affordable through API.
* [LLama2](https://huggingface.co/chat/): 
    * [Meta](https://llama.meta.com/)’s open sourced model, it’s starting to show its age with the release of mistral/mixtral based models.
* [Mistral-7b](https://huggingface.co/chat/): 
    * Small and decently smart open source model developed by [mistral.ai](https://mistral.ai), fits on consumer hardware, it’s actually very smart for its size and has a ton of community variants and flavors for many use cases.
* [Mixtral-8x7b](https://huggingface.co/chat/): 
    * Same as mistral-7b but MoE architecture with 8 experts, around 13b parameters are active on inference and people claim it’s smarter than gpt-3.5. 
    * [https://arxiv.org/abs/2401.04088](https://arxiv.org/abs/2401.04088)
* [Mistral Large](https://chat.mistral.ai/chat) and other [mistral.ai](https://mistral.ai) models: 
    * Smart but they are paid only through API unlike mistral/mixtral.
* [Gemini](https://gemini.google.com/?hl=es) and [Gemma](https://blog.google/technology/developers/gemma-open-models/) google models: 
    * Apparently worse than gpt-4 but the API has a free tier, and has integration with google services. Gemma is open source, but just 7b parameters.

*There’s a massive community in [Huggingface.co](https://Huggingface.co) where people make constant mixes of several popular models with different datasets, mostly focused on assistant, chat, [RAG](https://research.ibm.com/blog/retrieval-augmented-generation-RAG) or roleplay use cases.

Some models are significantly better than the base model, but also some of the models suffer from contamination where the shortcomings of the dataset(writing style, summarization issues…) of a model get worse or more obvious because of the mixing of models with the same base model.

For reference on the best models for a given task there are several leaderboards around, some of them on the site itself, that said the leaderboards should be seen strictly as a reference due to the complexity of the models and the technology itself.


## Image Generation



* [Stability.ai](https://Stability.ai)‘s stable diffusion: 
    * Popular base model for image generation, it’s fast and resource friendly and a lot of models are built from it. Default size is 512x512.
    * Supported resolutions: Anything below 768 pixels.
    * [https://stability.ai/stable-image](https://stability.ai/stable-image)
    * [https://github.com/Stability-AI/StableDiffusion](https://github.com/Stability-AI/StableDiffusion)
* [Runway research](https://research.runwayml.com/)’s Stable-diffusion v1.5: 
    * One of the many iterations of SD and the flagship model for open source image generation, it’s very versatile, fast and resource friendly. It has a massive amount of variants and support from the community.
    * [https://huggingface.co/runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
* [Stability.ai](https://Stability.ai)‘s SDXLv1.0:
    * The next generation model from Stability.ai, less flexible and more resource costly than SD 1.5 but with higher resolution/detailed outputs and tuned for better quality and less reliance on negative prompts. Default size is 1024x1024.
    * [https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
    * _Supported resolutions_:
    * 640 x 1536: 10:24 or 5:12
    * 768 x 1344: 16:28 or 4:7
    * 832 x 1216: 13:19
    * 896 x 1152: 14:18 or 7:9
    * 1024 x 1024: 1:1
    * 1152 x 896: 18:14 or 9:7
    * 1216 x 832: 19:13
    * 1344 x 768: 21:12 or 7:4
    * 1536 x 640: 24:10 or 12:5
* Midjourney:
    * [https://www.midjourney.com/home](https://www.midjourney.com/home)
    * [https://nijijourney.com](https://nijijourney.com)
* GPT-4-Vision: 
    * OpenAI’s image model, integrated in their subscription service, it’s separate in the API.

*Same as for the LLM space, [Civit.ai](https://Civit.ai) has a massive community with models for any use case.

Check the specific model/finetune page for optimal resolutions.


## Text to Speech/Sound Generation



* Silero-tts: 
    * [https://github.com/snakers4/silero-models](https://github.com/snakers4/silero-models)
* [XTTSv2](https://huggingface.co/coqui/XTTS-v2): 
    * One of the best open source tts models, can do voice cloning with 6s audio files. Unfortunately [Coqui.ai](https://Coqui.ai) is shutting down.
* [Suno.ai](https://www.suno.ai/)’s Bark:
    * Not just a tts model but text2sound, which can deviate from the provided prompt. Not the same as their song generation model though.
    * [https://github.com/suno-ai/bark/tree/main](https://github.com/suno-ai/bark/tree/main)
* Suno.ai’s [Chirp v1](https://app.suno.ai/):
    * Their song generation model, based on Bark.


## Speech to Text Generation



* OpenAI’s Whisper models:
    * Decent voice recognition and there are different versions with different model sizes.
    * [https://github.com/openai/whisper](https://github.com/openai/whisper)
    * [https://huggingface.co/openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
    * [https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013](https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013)


# Software

The actual libraries(transformers, Exllama, llama.cpp, diffusers…) to run the models are integrated in these backend solutions, they also offer a simple frontend and an API to connect to from any frontend. There is a lot of documentation on how to run these projects, they are self sufficient for basic tasks and some have modules and extensions for extra functionality or an API server to connect to from any other source. There are also plenty of frontend solutions to connect to those API endpoints.

Note that most of this software can be run on google colab free tier, and there are several already configured colabs around the community.


## Text Generation



* [Llama.cpp](https://github.com/ggerganov/llama.cpp): 
    * Single file solution for text inference with an API server and integrated UI, very easy to run as it’s just a single executable without the need of any install, can run fully on GPU or split between VRAM/RAM for when a model is too large to fit entirely on your GPU.
* [Kobold.cpp](https://github.com/LostRuins/koboldcpp): 
    * It’s an improved fork of llama.cpp, supposedly faster and with some extra functionalities.
* [Oobabooga’s text-generation-ui](https://github.com/oobabooga/text-generation-webui): 
    * The most versatile and complete, integrates llama.cpp and Exllama which are the best inference libraries for consumer hardware at the moment. It has several extensions and an API server.
* [SillyTavern](https://sillytavernai.com/): 
    * An excellent frontend-only solution for anything LLM related. While heavily focused on chat/roleplay, it offers great flexibility and lots of integrations with all kinds of services and backends, be it local or not. Also has modules to integrate web search, speech recognition, text-to-speech and image generation backends.
    * Scripts for easy installation: [https://github.com/SillyTavern/SillyTavern-Launcher](https://github.com/SillyTavern/SillyTavern-Launcher)
* [RisuAI](https://risuai.xyz/):
    * Another frontend-only for LLMs, this one is solely focused on roleplay/chat and doesn't have as many extensions.
* [llama.sh](https://github.com/m18coppola/llama.sh):
    * Simple frontend for the command line.
* [mikupad.html](https://github.com/lmg-anon/mikupad):
    * Simple frontend for the browser.
    * [https://lmg-anon.github.io/mikupad/mikupad.html](https://lmg-anon.github.io/mikupad/mikupad.html)

    *Note that while back-ends are needed for running local models, there are also a lot of online front-end solutions that can connect to your API if you enable the server on your end. Preferably use a personal VPN instead of opening ports on the router, or a reverse proxy like [NPM](https://nginxproxymanager.com/) or [NginX](https://www.nginx.com/) if you need to use SSL for things like speech recognition over the internet and such.



## Image Generation



* [Automatic1111’s stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui): 
    * The most popular solution for running SD and adjacent models, extremely versatile with tons of extensions, even has an extension to use it in Photoshop.
* [ConfyUI](https://github.com/comfyanonymous/ComfyUI):
    * Better workflow and more lightweight than A1111’s but less popular and versatile, A1111’s also has an extension for it.


## Text to Speech/Sound Generation



* Daswer123’s [xtts-webui](https://github.com/daswer123/xtts-webui) & [xtts-api-server](https://github.com/daswer123/xtts-api-server): 
    * You can use and finetune the model with the web UI or connect with SillyTavern to the API server.
* [Suno.ai’s Bark](https://github.com/suno-ai/bark/tree/main):
    * Can do text2sound from the terminal.


## Speech to Text Generation



* [OpenAI’s Whisper](https://github.com/openai/whisper):
    * Can transcribe & translate from the terminal.


# Hardware requirements

For text inference a 12GB+ GPU is advised if you plan to run anything more than 7b models with high quantization or large context windows.



* Exllama can fit up to 11b models with 8k context on 12GB. Supports multiple GPU configurations and it’s pretty fast. It can also run 13b models with low quantization and low context.
* Llama.cpp can fit bigger models if layers of the LLM are offloaded to RAM but the inference speed takes a big hit. Anything beyond 30b is probably not feasible on a 12GB GPU. You’ll have to play around with the amount of layers offloaded as it depends on the context size too.

For Stable Diffusion 8-12 GB is advised, heavily depending on batch size/image size. There are ways to lower the VRAM requirements but you need specific ways to generate images in small chunks.

For text2speech with XTTSv2 you only need around 3.5-5GB VRAM. Altho you can use RAM and the intel deepspeed library, the inference will take around 2-3 times.

Training & Fine Tuning takes more than 2-4 times the amount of VRAM than inference, and depending on the method used it can take much more.

[https://huggingface.co/spaces/hf-accelerate/model-memory-usage](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)

[https://huggingface.co/blog/trl-peft](https://huggingface.co/blog/trl-peft)

All that said, VRAM speed is king if everything fits in it.

**Software requirements (W11)**

Most of the projects for local deployment have easy to install scripts and documentation, but for some of the backend solutions or their extensions it’s necessary to install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (ST Extras, facechain, etc…). Check the requirements for each project’s dependencies before trying to run them to avoid issues.

Note that most of these projects rely on specific [python environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) created through [conda](https://docs.anaconda.com/free/miniconda/) and, as with any software in development, bugs and issues are expected. It is recommended to be able to manage those environments in case some dependency or software doesn't get correctly installed or configured by the install scripts, so a [very basic knowledge of both python and conda is required](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).


# Useful docs & guides

**General Prompt Engineering**

There is a lot to prompt engineering for both professional and entertainment use cases since the way you craft your prompts can largely affect the responses of the model used, which may also vary (definitely does) from one model to another.

[https://www.promptingguide.ai/papers](https://www.promptingguide.ai/papers)

[https://github.com/microsoft/generative-ai-for-beginners/blob/main/04-prompt-engineering-fundamentals/README.md](https://github.com/microsoft/generative-ai-for-beginners/blob/main/04-prompt-engineering-fundamentals/README.md)

**Chatbot making guides for roleplay**

This section can get very convoluted as storytelling and literary writing is a complex thing for LLMs (even for humans) and heavily depends on the model used, also the large context windows that are used in this use case affect the responses a lot. A rule of the thumb is to think about the models as what they are, text completion mechanisms, and for example if you let the writing style get dull over time, the context that you provide will be that same dull chat history and the model will provide a similar response, degrading the quality quickly, same for the repetition issues and such. Multiple character scenarios can also present a challenge for smaller models.

There are several ways to instruct a model about performing as a character, some of them consider the fact that those models often are trained with code, and so they use some JSONish or List (Boostyle/W++/Plists) formatting, others try to enforce the writing style of the bot with examples (Ali:chat).

[https://rentry.co/statuotw](https://rentry.co/statuotw) - Mainly for Mythomax (LLama2 base).

[https://rentry.org/HowtoMixtral](https://rentry.org/HowtoMixtral) - Mixtral and maybe other MoE models.

[https://rentry.org/mixtral-bot-tips](https://rentry.org/mixtral-bot-tips)

[https://rentry.org/chai-pygmalion-tips](https://rentry.org/chai-pygmalion-tips)

[https://wikia.schneedc.com/bot-creation/trappu/introduction](https://wikia.schneedc.com/bot-creation/trappu/introduction) - Ali:chat + Plists (probably the best general approach).


# Resources

**Communities**

[Huggingface.co](https://Huggingface.co) - Main community for sharing, discussing or testing LLMs.

[Civit.ai](https://Civit.ai) - Main community for sharing, discussing or testing Image models.

[https://www.reddit.com/r/LocalLLaMA/](https://www.reddit.com/r/LocalLLaMA/) - Main subreddit for discussion about local LLM deployment & development.

[https://www.reddit.com/r/StableDiffusion/](https://www.reddit.com/r/StableDiffusion/) - Stable diffusion subreddit.

**Chat UIs** - Free chatbots, some from the developers themselves, mainly for testing or using it as a service since you cannot do much other than talk, and some may have additional instructions baked in on their end. Check their pages for the API & costs.

[https://huggingface.co/chat](https://huggingface.co/chat) - LLama2, Mixtral & finetunes, and a bunch more.

[https://gemini.google.com/app](https://gemini.google.com/app) - google gemini

[https://copilot.microsoft.com/](https://copilot.microsoft.com/) - ~~gpt-3.5~~ gpt-4? 

[https://pi.ai/talk](https://pi.ai/talk)

[https://www.personal.ai/](https://www.personal.ai/)

[https://claude.ai/chats](https://claude.ai/chats) 

[https://chat.mistral.ai/chat](https://chat.mistral.ai/chat) - mistral large

[https://www.perplexity.ai/](https://www.perplexity.ai/)

[https://chat.openai.com](https://chat.openai.com) - gpt-3.5

[https://horde.koboldai.net/](https://horde.koboldai.net/) - crowdsourced AI generation, both text & image. Not too reliable but good for testing a variety of models.

**Other UIs & services** - For everything but language.

[https://app.suno.ai/](https://app.suno.ai/)

**Other API services** - Hosting open source LLM APIs.

[https://infermatic.ai/for-people/](https://infermatic.ai/for-people/)

[https://openrouter.ai/docs#models](https://openrouter.ai/docs#models)

[https://mancer.tech/](https://mancer.tech/)

[https://goose.ai/](https://goose.ai/)

[https://www.together.ai/pricing](https://www.together.ai/pricing)

**Compute services** - For renting GPU computing services and alike.

[https://vast.ai/](https://vast.ai/)

[https://colab.google/](https://colab.google/)

[https://www.runpod.io/](https://www.runpod.io/)

[https://massedcompute.com/](https://massedcompute.com/)

[https://www.paperspace.com/](https://www.paperspace.com/)

**Google Colab examples** - Beware of the captchas

[Official Whisper](https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb)

[KoboldAI backend on GPU](https://colab.research.google.com/github/koboldai/KoboldAI-Client/blob/main/colab/GPU.ipynb#scrollTo=lVftocpwCoYw) - [TPU (less availability)](https://colab.research.google.com/github/KoboldAI/KoboldAI-Client/blob/main/colab/TPU.ipynb?authuser=0&pli=1#scrollTo=ZIL7itnNaw5V)

[Official Kobold.cpp](https://colab.research.google.com/github/henk717/koboldcpp/blob/concedo/colab.ipynb) - [Alternative Kobold.cpp](https://colab.research.google.com/gist/Pyroserenus/48d30c7d3db533fcce5ec6ad8343b6b0/mythogguf.ipynb)

[Aphrodite engine](https://colab.research.google.com/github/AlpinDale/misc-scripts/blob/main/Aphrodite.ipynb#scrollTo=uJS9i_Dltv8Y)

[Official Oobabooga on GPU](https://colab.research.google.com/github/oobabooga/text-generation-webui/blob/main/Colab-TextGen-GPU.ipynb) - [Alternative](https://colab.research.google.com/drive/1ZsRJCH4H6ZNlNoU3AMngR8MHmuZnQu2T#scrollTo=MFQl6-FjSYtY) - [Alt2](https://colab.research.google.com/drive/1ztRHfwON9zCeaEiaKPWXIfCDmSYwfzu_#scrollTo=UecGsZ88rsOF)


# Specific Guides & Setups

**Building an all in one solution to connect from anywhere with all the services(WIP)**

	

The idea is to build a server side API endpoint with either Kobold.cpp or Oobabooga’s webui with SillyTavern as your UI to use remotely. We will need all the corresponding software and their dependencies on the computer you’re planning to use. In addition we will need several software to ensure a secure connection is available, there are several ways to achieve this and some might be better than others but I’m going to show the configuration I had (which isn't a professional solution but works well for personal use). I will assume you already set up the corresponding software for local use, launching the inference server with the ‘–api’ arguments and all that jazz (no need to use ‘–listen’ for all addresses, we will use the reverse proxy for accessing the server).

We will need to setup a reverse proxy in order to route the requests to the specific port for each service, such as the API endpoint of the Inference server, both for stable diffusion & oobaboga’s default port is 7860 so you’ll have to change one if you plan to use them in the same machine. For reference, these are the default ports for each software we’re gonna use:

[Oobabooga’s webui - 7860](http://localhost:7860)

[SillyTavern - 8000](http://localhost:8000)

[ST Extras (for speech recognition & other modules) - 5100](http://localhost:5100)

[Daswer123’s XTTS sever - 8020](http://localhost:8020)

[Automatic1111’s SD webui - 7860](http://localhost:7860)

[NginX proxy manager - 81](http://localhost:81)

First off you will need a way to connect to your server without the need to open any port on your router in order to prevent any kind of security issue, since this guide is just meant for personal use we’re not gonna publish any of the services to the internet. That’s done via a private VPN, there are many and many ways of achieving this, but for ease of use we will be setting up an already managed VPN with [Tailscale](https://tailscale.com/download) ([Zerotier](https://www.zerotier.com/download/) is also pretty similar).

There’s already documentation on how to install it and it's pretty simple so I’ll just say you need to install it in both your server and your client where you want to connect from, it has builds for linux, windows, mac and android, and if you need to install it on an unsupported device you can always [create a router to route the traffic thru a supported device](https://tailscale.com/kb/1019/subnets). 

Once you have the VPN up and running you can already connect remotely through your VPN’s IP address (something like 100.x.x.x) and the port you want to access, for connecting to your running ST instance for example, it will be something like ‘http://100.100.100.100:8000’ and you already have a way to use it remotely. The problem is as you might notice some functionality is not possible because the connection you are reaching (and what the server is providing) is insecure (no https), the speech recognition for example won't be routed thru the internet but rather use your local mic every time you try to use it. That happens despite using the VPN which already is secure because the services and your browser where you connect from have no way to know that.

For that problem we will use [Tailscale’s HTTPS solution](https://tailscale.com/kb/1153/enabling-https) and a piece of software called a [reverse proxy](https://www.cloudflare.com/learning/cdn/glossary/reverse-proxy/) ([NginX](https://www.nginx.com/) is among many other things a reverse proxy), which basically will translate the raw unencrypted data to securely access your services as if you were locally accessing them.

The first step is to follow [Tailscale documentation and enable HTTPS](https://tailscale.com/kb/1153/enabling-https), then create the certificates for each machine you want to connect to and save them with the command [tailscale cert](https://tailscale.com/blog/tls-certs). Once you have that setup we encounter a different problem, Tailscale provides a [very basic implementation of a DNS server](https://tailscale.com/kb/1081/magicdns#enabling-magicdns) that may or may not work for your setup, so we will disable it and use an external DNS provider, [Cloudflare](https://dash.cloudflare.com/login) in our case (you can also locally host a DNS server). To disable Tailscale MagicDNS [go to the admin dashboard > dns tab and disable magicDNS](https://login.tailscale.com/admin/dns), while we are here, add Cloudflare’s public DNS servers (1.1.1.1 & 1.0.0.1) and check ‘override local dns’. 

![1](images/cloudflareonts.png "should look like this")



# ToDo & WIP



1. Guide on setting up a reverse proxy with SSL offloading(NPM) + DNS routing(cloudflare or local) to integrate SillyTavern and the backend API servers for voice recognition/synthesis over the internet. For something like using ChatGPT app on android in the free talk mode, but with the versatility of ST as your UI.
2. Links for popular web chatbot/ui services(some of them offer an API endpoint for a popular model, or a custom fine tuned model like character.ai and such)
3. Links for guides for prompt engineering for general purpose and different bot making guides.
4. List of different settings/tips for some tested models and suggestions for some use cases.
5. Some tips about samplers and convergence in diffusion models
6. Add text to sound(suno.ai) and text2video([https://runwayml.com/ai-tools/gen-2/](https://runwayml.com/ai-tools/gen-2/)) models & services
