from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import json

# 创建 FastAPI 实例
app = FastAPI()

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="sk-1bdfda498595411d961a4cbff37c892a",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 定义请求体的数据模型
class UserQuery(BaseModel):
    content: str

# 定义接口
@app.post("/v1/query")
async def ask_user(query: UserQuery):
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": """
你是一个厨房设计专家，请你从下面的自然语言描述中提取厨房功能模块信息，并按照以下格式严格输出，仅包含4个模块。输出内容为固定结构的 JSON 数组，格式如下：
[
    {
        "module_name": "加工出品模块",
        "area": 60,
        "adjacent_to": [...],
        "linked_door": "出餐口"
    },
    {
        "module_name": "凉菜及主食间模块",
        "area": 16,
        "adjacent_to": [],
        "linked_door": null
    },
    {
        "module_name": "仓储模块",
        "area": 32,
        "adjacent_to": ["加工出品模块"],
        "linked_door": "进货口"
    },
    {
        "module_name": "清洗模块",
        "area": 35,
        "adjacent_to": ["加工出品模块"],
        "linked_door": "回餐口"
    }
]

请注意：
1. 仅允许返回这4个模块，顺序、名称和面积必须一致；
2. "adjacent_to" 字段仅填写明确有邻接要求的模块；如果自然语言中没有描述邻接要求的模块，清洗模块紧邻仓储模块；
3. "linked_door" 字段仅填写自然语言中明确指出的模块对应的门口，则加工出品模块对应门口为出餐口，清洗模块对应门口时回餐口，仓储模块对应门口是进货口；
4. 不允许添加任何其他模块或额外解释说明；
5. 输出必须是 JSON 数组格式，不能带注释，不能有多余换行、说明、提示。
"""
},
            {"role": "user", "content": query.content},
        ],
    )
    responses = completion.model_dump_json()
    answer_text = json.loads(responses)
    kitcheb_data = answer_text["choices"][0]["message"]["content"]

    print(kitcheb_data)
    return {"response": kitcheb_data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


