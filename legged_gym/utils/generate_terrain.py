import json
import torch
import base64
import requests
class TerrainGenerator:
    def __init__(self, generation_prompt, function_prompt, image_prompt, llm_model_name, vlm_model_name, instruct_image_path, instruct_lang, api_url="your-api-url", authen="your-api-key"):
        with open(generation_prompt, "r") as pf:
            """
            lang2terrain
            """
            self.prompt_content = pf.read()
        
        with open(function_prompt, "r") as ff:
            # function.json
            self.function_descriptions = ff.read()

        with open(image_prompt, "r") as file:
            # image2lang
            self.image_prompt = file.read()

        self.llm_model_name = llm_model_name
        self.vlm_model_name = vlm_model_name
        self.instruct_image_path = instruct_image_path
        self.instruct_lang = instruct_lang
        self.api_url = api_url
        self.authen = authen
        self.scene_description = instruct_lang

    def init_llm(self):
        from transformers import pipeline

        self.llm_pipeline = pipeline(
            "text-generation",
            model=self.llm_model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
    
    def init_vlm(self):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self.vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.vlm_model_name,
            torch_dtype=torch.float16, 
            device_map="auto"
        )

        self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_model_name)
    
    def release_models(self):
        if self.vlm_model is not None:
            del self.vlm_model
            del self.vlm_processor
        if self.llm_model is not None:
            del self.llm_pipeline
        torch.cuda.empty_cache()
        self.vlm_model = None
        self.vlm_processor = None
        self.llm_pipeline = None

    def vlm_generate_description(self):
        from qwen_vl_utils import process_vision_info

        self.init_vlm()

        messages = [{
            "role": "user",
            "content": [{
                "type": "image",
                "image": self.instruct_image_path,
                },
                {"type": "text", "text": self.image_prompt},
                ],
            }
        ]

        
        text = self.vlm_processor.apply_chat_template(
            messages,
            tokenize=False, 
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=1024)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        terrain_description = self.vlm_processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        self.release_models()
        return terrain_description
        
    def generate_json_terrain(self):
        if self.instruct_lang is None:
            self.instruct_lang = self.vlm_generate_description_api()
        return self.generate_json_terrain_api()
    
    def parse_output(self, outputs):
        try:
            generated_text = outputs[0]["generated_text"]
            function_called = json.loads(generated_text)
            return function_called
        except json.JSONDecodeError:
            print("Generated tool calling not in JSON format.")
            return None
    
    def vlm_generate_description_api(self):
        base64_image = ""
        with open(self.image_prompt, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        payload = {
            "model": self.vlm_model_name,
            "messages":[
            { 
                "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail":"low"
                    }
                },
                    {"type": "text", "text": self.image_prompt},
            ],
                "role": "user"
            }],
            "stream": False,
            "max_tokens": 1024,
            "stop": ["null"],
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {"type": "text"},
        }
        headers = {
            "Authorization": f"Bearer {self.authen}",
            "Content-Type": "application/json"
        }

        response = requests.request("POST", self.api_url, json=payload, headers=headers).json()
        if "choices" in response:
            self.scene_description = response['choices'][0]['message']['content']
            return response['choices'][0]['message']['content']
        else:
            return "Fail to generate scene prompt"

    def generate_json_terrain_api(self):
        payload = {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "messages":[
            { 
                "content": [
                    {"type": "text", "text": self.scene_description},
                    {"type": "text", "text": self.prompt_content}
            ],
                "role": "user"
            }],
            "stream": False,
            "max_tokens": 1024,
            "stop": ["null"],
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {"type": "text"},
            "tools": self.function_descriptions
        }
        headers = {
            "Authorization": f"Bearer {self.authen}",
            "Content-Type": "application/json"
        }

        response = requests.request("POST", self.api_url, json=payload, headers=headers).json()
        if "choices" in response:
            return response['choices'][0]['message']['tool_calls']
        else:
            return "Function generate fails."
