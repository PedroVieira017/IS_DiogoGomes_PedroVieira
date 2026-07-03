def load_model(args):
    model_path = args.model_path.lower()

    if args.model_path in ["doubao-vision-pro-32k", "doubao-vision-lite-32k", "doubao-1.5-vision-pro-32k"]:
        from models.doubao_api import DoubaoAPIModel

        model = DoubaoAPIModel(model_name=args.model_path,
                               api_key=args.api_key,
                               user_prompt=args.user_prompt)
    elif model_path.startswith("ollama"):
        from models.ollama_vision import OllamaVisionModel

        model_name = args.model_path.split(":", 1)[1] if ":" in args.model_path else "llava"
        model = OllamaVisionModel(model_name=model_name,
                                  endpoint=args.base_url if args.base_url else None,
                                  user_prompt=args.user_prompt)
    elif model_path.startswith("iaedu"):
        from models.iaedu_api import IAeduAPIModel

        model = IAeduAPIModel(model_name=args.model_path,
                              api_key=args.api_key,
                              endpoint=args.base_url if args.base_url else None,
                              channel_id=getattr(args, "channel_id", None),
                              thread_id=getattr(args, "thread_id", None),
                              user_prompt=args.user_prompt,
                              timeout=getattr(args, "api_timeout", 120))
    elif args.model_path in ["gpt-4o", "gpt-4o-mini"]:
        from models.openai_api import OpenAIAPIModel

        model = OpenAIAPIModel(model_name=args.model_path,
                               api_key=args.api_key,
                               base_url=args.base_url,
                               user_prompt=args.user_prompt)
    elif args.model_path in ["kimi-latest", "moonshot-v1-8k-vision-preview", "moonshot-v1-32k-vision-preview", "moonshot-v1-128k-vision-preview"]:
        from models.kimi_api import MoonshotAPIModel

        model = MoonshotAPIModel(model_name=args.model_path,
                                 api_key=args.api_key,
                                 user_prompt=args.user_prompt)
    elif "qwen" in model_path and "vl" in model_path:
        from models.qwenvl_model import QwenVisionModel

        model = QwenVisionModel(model_path=args.model_path,
                                user_prompt=args.user_prompt)
    elif "internvl" in model_path:
        from models.internvl_model import InternVLModel

        model = InternVLModel(model_path=args.model_path,
                              user_prompt=args.user_prompt)
    elif "llava" in model_path and "onevision" not in model_path:
        from models.llava_model import LlavaModel

        model = LlavaModel(model_path=args.model_path,
                           user_prompt=args.user_prompt)
    elif "llava" in model_path and "onevision" in model_path:
        from models.llavaonevision import LlavaOnevisionModel

        model = LlavaOnevisionModel(model_path=args.model_path,
                                    user_prompt=args.user_prompt)
    elif "minicpm-o" in model_path:
        from models.minicpm_o_model import MiniCPMOModel

        model = MiniCPMOModel(model_path=args.model_path,
                              user_prompt=args.user_prompt)
    elif "mplug" in model_path:
        from models.mplug_model import mPLUGModel

        model = mPLUGModel(model_path=args.model_path,
                           user_prompt=args.user_prompt)
    elif "ovis" in model_path:
        from models.ovis2_model import Ovis2Model

        model = Ovis2Model(model_path=args.model_path,
                           user_prompt=args.user_prompt)
    elif "glm" in model_path:
        from models.glm4v_model import GLM4VModel

        model = GLM4VModel(model_path=args.model_path,
                           user_prompt=args.user_prompt)
    elif "sharegpt" in model_path:
        from models.sharegpt4_model import ShareGPT4VModel

        model = ShareGPT4VModel(model_path=args.model_path,
                                user_prompt=args.user_prompt)
    else:
        raise ValueError(f"Unsupported model_path: {args.model_path}")

    return model
