import pandas as pd
import json
# 读取 parquet 文件
# parquet_file_path = "data/test.parquet"
parquet_file_path = "/inspire/hdd/project/qproject-fundationmodel/liupengfei-24025/ai_engineer/task/debug/ai-engineer-benchmark/evaluations/task_14/data/references/test_with_gt.parquet"

p_df = pd.read_parquet(parquet_file_path, engine="pyarrow")
print(p_df)
print("dataset len: ", len(p_df))
print(p_df.columns)
print(f'data_source: {p_df["data_source"][0]}')
print(f'prompt: {p_df["prompt"][0]}')
print(f'ability: {p_df["ability"][0]}')
print(f'reward_model: {p_df["reward_model"][0]}')
print(f'extra_info: {p_df["extra_info"][0]}')

# # 统计data_source列中的唯一标签
# unique_labels = p_df['data_source'].unique()
# print("data_source列的标签种类: ", unique_labels)

# # 整理reward_model中ground_truth的db_id字段
# print("\n=== 整理reward_model中ground_truth的db_id字段 ===")

# # 存储包含特定模式的内容
# contains_s_underscore = []
# contains_comma_underscore = []
# contains_pokemon_underscore = []

# for idx in range(len(p_df)):
#     try:
#         reward_model = p_df["reward_model"][idx]
        
#         # 如果reward_model是字符串，尝试解析为JSON
#         if isinstance(reward_model, str):
#             reward_model = json.loads(reward_model)
        
#         # 检查是否存在ground_truth和db_id字段
#         if isinstance(reward_model, dict) and 'ground_truth' in reward_model:
#             ground_truth = reward_model['ground_truth']
            
#             if isinstance(ground_truth, dict) and 'db_id' in ground_truth:
#                 db_id = ground_truth['db_id']
                
#                 # 检查是否包含"'s_"模式
#                 if "'s_" in str(db_id):
#                     contains_s_underscore.append({
#                         'index': idx,
#                         'db_id': db_id,
#                         'prompt': p_df["prompt"][idx] if idx < len(p_df) else None
#                     })
                
#                 # 检查是否包含",_"模式
#                 if ",_" in str(db_id):
#                     contains_comma_underscore.append({
#                         'index': idx,
#                         'db_id': db_id,
#                         'prompt': p_df["prompt"][idx] if idx < len(p_df) else None
#                     })
                    
#                 # pokémon
#                 if "é" in str(db_id):
#                     contains_pokemon_underscore.append({
#                         'index': idx,
#                         'db_id': db_id,
#                         'prompt': p_df["prompt"][idx] if idx < len(p_df) else None
#                     })
    
#     except Exception as e:
#         print(f"处理第{idx}行时出错: {e}")

# # 输出结果
# print(f"\n包含\"'s_\"模式的内容 (共{len(contains_s_underscore)}条):")
# for item in contains_s_underscore:
#     print(f"索引 {item['index']}: db_id = {item['db_id']}")
#     # if item['prompt']:
#     #     print(f"  相关prompt: {item['prompt'][:100]}...")
#     # print()

# print(f"\n包含\"é\"模式的内容 (共{len(contains_comma_underscore)}条):")
# for item in contains_comma_underscore:
#     print(f"索引 {item['index']}: db_id = {item['db_id']}")
#     # if item['prompt']:
#     #     print(f"  相关prompt: {item['prompt'][:100]}...")
#     # print()

# print(f"\n包含\"\"模式的内容 (共{len(contains_pokemon_underscore)}条):")
# for item in contains_pokemon_underscore:
#     print(f"索引 {item['index']}: db_id = {item['db_id']}")
#     # if item['prompt']:
#     #     print(f"  相关prompt: {item['prompt'][:100]}...")
#     # print()

# # 统计信息
# print(f"\n=== 统计信息 ===")
# print(f"总数据条数: {len(p_df)}")
# print(f"包含\"'s_\"模式的数据条数: {len(contains_s_underscore)}")
# print(f"包含\",_\"模式的数据条数: {len(contains_comma_underscore)}")
# print(f"包含\"é\"模式的数据条数: {len(contains_pokemon_underscore)}")

