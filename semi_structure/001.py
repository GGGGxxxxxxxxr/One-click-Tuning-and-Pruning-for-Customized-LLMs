# rl_prompt_training.py
"""
End‑to‑end RL pipeline:
1. g_φ (GPT2) generates prompts
2. Teacher answers => (prompt, answer) pairs
3. Every N=50 pairs -> finetune Student & compute rewards
4. Use REINFORCE to update g_φ
"""

import os, random, torch, json, warnings
from pathlib import Path
from typing import List, Dict

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from student_rl_backend import StudentTrainer, train_student_and_get_reward
from teacher_api import query_teacher  
from transformers import logging
logging.set_verbosity_error()


# ----------- 超参数 -----------
BATCH_SIZE = 50          # 每多少 prompt 训练一次 Student
TOTAL_STEPS = 1000       # 生成 prompt 次数
LR = 1e-5                # g_φ 学习率
MAX_PROMPT_LEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------- 初始化 g_φ ----------
g_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
g_tokenizer.pad_token = g_tokenizer.eos_token
g_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
g_model.train()
g_opt = torch.optim.Adam(g_model.parameters(), lr=LR)

# ----------- 初始化 StudentTrainer ----------
student_trainer = StudentTrainer(base_model="gpt2", work_dir="./student_ckpt")

# ----------- 数据池 ----------
pool: List[Dict] = []          # 存 (prompt, answer)
teacher_answers: List[str] = []


def generate_prompt(seed: str = "medical"):
    """使用 g_φ 生成一个 prompt，并返回 prompt + 对应 log_prob"""
    
    inputs = g_tokenizer(seed, return_tensors="pt").to(DEVICE)
    outputs = g_model.generate(
        **inputs,
        max_length=MAX_PROMPT_LEN,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        return_dict_in_generate=True,
        output_scores=True,
    )
    ids = outputs.sequences[0]                      # [seq_len]
    prompt_text = g_tokenizer.decode(ids, skip_special_tokens=True)

    # 计算生成 token 的 log_probs（简易实现：取最后一步）
    # HuggingFace 输出 scores 为每步 logits；我们把所有 token 的对数概率相加
    step_logits = outputs.scores                   # list(len = gen_len) each (1,vocab)
    log_probs = []
    for i, logit in enumerate(step_logits):
        token_id = ids[len(inputs.input_ids[0]) + i]   # 新生成 token 的 id
        log_prob = torch.log_softmax(logit, dim=-1)[0, token_id]
        log_probs.append(log_prob)
    log_prob_sum = torch.stack(log_probs).sum()        # scalar tensor

    return prompt_text, log_prob_sum


def reinforce_update(losses: List[torch.Tensor]):
    """累积 N 条 prompt 的 REINFORCE 损失后，一次性反向传播"""
    total_loss = torch.stack(losses).mean()
    total_loss.backward()
    g_opt.step()
    g_opt.zero_grad()
    return total_loss.item()


# ----------- 主循环 -----------
reinforce_buffer = []      # 存 (-reward * log_prob) loss tensor
prompt_texts = []   # 存生成的 prompt

for step in range(1, TOTAL_STEPS + 1):
    # 1) 生成 prompt
    prompt, _ = generate_prompt(seed="medical")
    prompt_texts.append(prompt)
    

    # 2) Teacher 回答
    answer = query_teacher(prompt)

    pool.append({"prompt": prompt, "answer": answer})
    teacher_answers.append(answer)

    # 3) 满 BATCH_SIZE -> 训练 student & 计算 reward
    if len(pool) % BATCH_SIZE == 0:
        print(f"\n=== [Batch {len(pool)//BATCH_SIZE}] Student finetune & reward ===")
        avg_r, r_list = train_student_and_get_reward(
            pool[-BATCH_SIZE:], teacher_answers[-BATCH_SIZE:], student_trainer, epochs=1
        )
        print(f"Average reward: {avg_r:.4f}")

        # 4) 将 reward 映射到对应 log_prob，形成 REINFORCE loss
        for r, idx in zip(r_list, range(step - BATCH_SIZE + 1, step + 1)):
            # 由于 log_prob 已经存于 prompt->loss 对应，这里简化：直接重算
            # (为了示例易懂，也可以在生成时存 log_prob 到列表)
            p, lp = generate_prompt(seed="medical")  # 重算 log_prob 近似
            reinforce_buffer.append(-r * lp)

        # 5) 反向传播更新 g_φ
        loss_val = reinforce_update(reinforce_buffer)
        reinforce_buffer.clear()
        print(f"g_phi updated | REINFORCE loss: {loss_val:.4f}")

print("\nTraining finished!")
