
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

if __name__ == '__main__':
    configuration = CLIPTextConfig()
    model = CLIPTextModel(configuration)
    
