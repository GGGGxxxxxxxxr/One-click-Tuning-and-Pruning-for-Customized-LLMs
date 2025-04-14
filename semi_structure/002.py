# student_rl_backend.py
import json, os, tempfile, torch, shutil, warnings
from pathlib import Path
from typing import List, Dict, Tuple

from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import Dataset
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score


# ---------- 1. 数据集封装 ----------
class _QADataset(Dataset):
    """Prompt‑Answer 对 -> GPT2 自回归训练样本"""
    def __init__(self, qa_pairs: List[Dict], tokenizer, max_len=512):
        self.samples = []
        for item in qa_pairs:
            prompt = item["prompt"].strip()
            answer = item["answer"].strip()
            txt = f"<|startoftext|>{prompt}\nAnswer: {answer}<|endoftext|>"
            self.samples.append(
                tokenizer(txt, truncation=True, max_length=max_len)["input_ids"]
            )
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx])


# ---------- 2. StudentTrainer ----------
class StudentTrainer:
    """
    负责：微调 GPT‑2 Student + 评估 mimic 程度 + 计算 reward
    """
    def __init__(
        self,
        base_model: str = "gpt2",
        work_dir: str = "./student_ckpt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 tokenizer / model
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(base_model).to(device)

    # ---- 2.1 微调函数 ----
    def finetune(
        self,
        qa_pairs: List[Dict],
        epochs: int = 1,
        batch_size: int = 2,
        fp16: bool = True,
    ) -> str:
        """
        用给定 (prompt, answer) 列表做一次增量微调
        返回：保存 checkpoint 路径
        """
        dataset = _QADataset(qa_pairs, self.tokenizer)
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        # 每次微调存到临时子目录，方便后续加载
        ckpt_dir = tempfile.mkdtemp(dir=self.work_dir, prefix="step_")

        args = TrainingArguments(
            output_dir=ckpt_dir,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            logging_steps=10,
            fp16=fp16 and torch.cuda.is_available(),
            save_strategy="no",
            report_to=[],                # 关闭 wandb 等
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=collator,
        )
        # suppress HF warnings inside RL loop
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.train()

        # 保存权重
        trainer.save_model(ckpt_dir)
        return ckpt_dir

    # ---- 2.2 单条生成 ----
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )[0]
        return self.tokenizer.decode(out_ids, skip_special_tokens=True)

    # ---- 2.3 评估 + reward ----
    def evaluate_batch(
        self,
        qa_pairs: List[Dict],
        teacher_answers: List[str],
        bleu_weight: float = 0.5,
        bert_weight: float = 0.5,
    ) -> Tuple[List[float], float]:
        """
        给一批 prompt，比较 student vs teacher
        返回：每条 reward 列表 + 平均 reward
        """
        student_outs = [
            self.generate(item["prompt"]).replace(item["prompt"], "").strip()
            for item in qa_pairs
        ]

        # BLEU
        bleu_scores = [
            sentence_bleu([t.split()], s.split()) for s, t in zip(student_outs, teacher_answers)
        ]
        # BERTScore
        _, _, bert_f1 = bert_score(student_outs, teacher_answers, lang="en", verbose=False)

        # reward = 加权和
        rewards = [
            bleu_weight * b + bert_weight * f.item()
            for b, f in zip(bleu_scores, bert_f1)
        ]
        return rewards, sum(rewards) / len(rewards)


# ---------- 3. 强化学习循环可直接调用的高层函数 ----------
def train_student_and_get_reward(
    qa_pairs: List[Dict],
    teacher_answers: List[str],
    student_trainer: StudentTrainer,
    epochs: int = 1,
) -> Tuple[float, List[float]]:
    """
    RL 主循环每 50 个样本调用：
      1. 用最新样本增量微调 student
      2. 评估 mimic 效果 → reward
      3. 返回 avg_reward, individual_rewards
    """
    # 微调
    student_trainer.finetune(qa_pairs, epochs=epochs)

    # 评估
    rewards, avg = student_trainer.evaluate_batch(qa_pairs, teacher_answers)
    return avg, rewards

