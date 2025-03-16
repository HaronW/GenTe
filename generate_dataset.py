import base64
import shutil
import numpy as np
# from utils.terrain_utils_single import * 

from utils.terrain_utils_parts import * 
import json
import os 
import numpy as np
from PIL import Image
import requests
import json

def convert_height_to_grayscale(height_field, output_path):
    height_array = np.array(height_field, dtype=np.float32)
    if np.all(height_array == height_array[0,0]):
        normalized = np.full_like(height_array, 128, dtype=np.uint8)  # 全灰
    else:
        min_val = np.min(height_array)
        max_val = np.max(height_array)
        normalized = (height_array - min_val) / (max_val - min_val) * 255
        normalized = normalized.astype(np.uint8)
    img = Image.fromarray(normalized, mode='L')
    img.save(output_path)

def image_to_base64(url):
    with open(url, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

def two_phase_generate():
    directory = '.\GenTe\resources\images'
    prompts_dir = '.\GenTe\resources\prompts'
    first_generate = 1
    fail_dir = []
    while len(fail_dir) != 0 or first_generate == 1:
        if first_generate == 1:
            file_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        else:
            file_list = [f"{i}.jpg" for i in fail_dir]
            fail_dir = []
        for file in file_list: 
            try:
                file_name_without_extension = os.path.splitext(os.path.basename(file))[0]
                dest = f"./dest/{file_name_without_extension}"
                if not os.path.exists(dest):
                    os.mkdir(dest)
                shutil.copy(f"{directory}/{file}", f"{dest}/{file}")
                base64_image = image_to_base64(f"{directory}/{file}")
                result_collection = {}

                with open(f'{prompts_dir}/image2lang.txt', 'r', encoding='utf-8') as file:
                    scene_prompt = file.read()    
                url = "url"

                payload = {
                    "model": "Qwen/Qwen2-VL-72B-Instruct",
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
                            {"type": "text", "text": scene_prompt},
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
                    "seed": 42
                }
                headers = {
                    "Authorization": "your-api-key",
                    "Content-Type": "application/json"
                }

                response = requests.request("POST", url, json=payload, headers=headers)
                scene_description = response.json()['choices'][0]['message']['content']
                result_collection['scene_description'] = scene_description

                function = ""
                with open(f'{prompts_dir}/function.json', 'r', encoding='utf-8') as file:
                    function = eval(file.read())

                with open(f'{prompts_dir}/lang2terrain.txt', 'r', encoding='utf-8') as file:
                    lang2terrain = file.read()

                payload = {
                    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    "messages":[
                    { 
                        "content": [
                            {"type": "text", "text": scene_description},
                            {"type": "text", "text": lang2terrain}
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
                    "tools": function,
                    "seed": 42
                }

                response = requests.request("POST", url, json=payload, headers=headers)

                toolcalls = response.json()['choices'][0]['message']['tool_calls']
                result_collection['tool_calls'] = toolcalls

                terrain = SubTerrain(
                    "terrain",
                    width=200,
                    length=200,
                    vertical_scale=0.005,
                    horizontal_scale=0.1
                )

                for func in toolcalls:
                    if func['function']['name'] is None:
                        raise ValueError(f"Function name {func['function']['name']} not found.")
                    function_exe = eval(func['function']['name'])

                    paras = eval(func['function']['arguments'])
                    if func['function']['name'] == 'generate_river_terrain':
                        paras['river_path'] = [[10.0, 15.0], [15.0, 20.0]]
                    paras['terrain'] = terrain

                    terrain = function_exe(**paras)
                terrain.smooth_transitions(int(terrain.width * 0.1), 2)
                gray_array = terrain.height_field_raw

                json_path = f'{dest}/data.json'

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result_collection, f, ensure_ascii=False, indent=4)

                array_path = f'{dest}/array.npy'
                np.save(array_path, terrain.height_field_raw)
                convert_height_to_grayscale(gray_array, f"{dest}/result.png")
                if not os.path.exists(f"{dest}/prompts"):
                    os.mkdir(f"{dest}/prompts")
                with open(f'{dest}/prompts/scene_prompt.txt', 'w', encoding='utf-8') as file:
                    file.write(scene_prompt)

                with open(f'{dest}/prompts/function.json', 'w', encoding='utf-8') as json_file:
                    json.dump({"description": function}, json_file, ensure_ascii=False, indent=4)

                with open(f'{dest}/prompts/lang2terrain.txt', 'w', encoding='utf-8') as file:
                    file.write(lang2terrain)
                first_generate = 0
            except Exception as e:
                print(e)
                fail_dir.append(file_name_without_extension)
    
    print("Done")
def one_phase_generate():
    directory = r'.\GenTe\resources\images'
    prompts_dir = r'.\GenTe\resources\prompts'
    first_generate = 1
    fail_dir = []
    while len(fail_dir) != 0 or first_generate == 1:
        if first_generate == 1:
            file_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        else:
            file_list = [f"{i}.txt" for i in fail_dir]
            fail_dir = []
        for file in file_list: 
            try:
                file_name_without_extension = os.path.splitext(os.path.basename(file))[0]
                dest = f"./dest/{file_name_without_extension}"
                if not os.path.exists(dest):
                    os.mkdir(dest)
                result_collection = {}
                url = "url"
                headers = {
                    "Authorization": "your-api-key",
                    "Content-Type": "application/json"
                }
                with open(f"{directory}/{file}", 'r', encoding='utf-8') as file:
                    scene_description = file.read()
                result_collection['scene_description'] = scene_description

                function = ""
                with open(f'{prompts_dir}/function.json', 'r', encoding='utf-8') as file:
                    function = eval(file.read())

                with open(f'{prompts_dir}/lang2terrain.txt', 'r', encoding='utf-8') as file:
                    lang2terrain = file.read()

                payload = {
                    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    "messages":[
                    { 
                        "content": [
                            {"type": "text", "text": scene_description},
                            {"type": "text", "text": lang2terrain}
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
                    "tools": function,
                    "seed": 42
                }

                response = requests.request("POST", url, json=payload, headers=headers)

                toolcalls = response.json()['choices'][0]['message']['tool_calls']
                result_collection['tool_calls'] = toolcalls

                terrain = SubTerrain(
                    "terrain",
                    width=200,
                    length=200,
                    vertical_scale=0.005,
                    horizontal_scale=0.1
                )

                for func in toolcalls:
                    if func['function']['name'] is None:
                        raise ValueError(f"Function name {func['function']['name']} not found.")
                    function_exe = eval(func['function']['name'])

                    paras = eval(func['function']['arguments'])
                    if func['function']['name'] == 'generate_river_terrain':
                        paras['river_path'] = [[10.0, 15.0], [15.0, 20.0]]
                    paras['terrain'] = terrain

                    terrain = function_exe(**paras)
                terrain.smooth_transitions(int(terrain.width * 0.1), 2)
                gray_array = terrain.height_field_raw
                

                json_path = f'{dest}/data.json'

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result_collection, f, ensure_ascii=False, indent=4)

                array_path = f'{dest}/array.npy'
                np.save(array_path, terrain.height_field_raw)
                convert_height_to_grayscale(gray_array, f"{dest}/result.png")
                if not os.path.exists(f"{dest}/prompts"):
                    os.mkdir(f"{dest}/prompts")

                with open(f'{dest}/prompts/function.json', 'w', encoding='utf-8') as json_file:
                    json.dump({"description": function}, json_file, ensure_ascii=False, indent=4)

                with open(f'{dest}/prompts/lang2terrain.txt', 'w', encoding='utf-8') as file:
                    file.write(lang2terrain)
                first_generate = 0
            except Exception as e:
                print(e)
                fail_dir.append(file_name_without_extension)
    
    print("Done")

one_phase_generate()
two_phase_generate()