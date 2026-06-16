from huggingface_hub import HfApi

api = HfApi()

try:
    info = api.model_info("nvidia/Cosmos-1.0-Guardrail")
    print("ACCESS OK")
except Exception as e:
    print(e)