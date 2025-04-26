import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Tuple
import numpy as np

import torch.nn as nn

class QwenRLModel:
    def __init__(self, model_name: str = "Qwen/Qwen-7B-Chat", device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True
        ).eval()

    def generate_response(self, prompt: str, max_length: int = 2048) -> str:
        """Generate response using the Qwen model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def compute_rewards(self, responses: List[str], reward_criteria: Dict[str, float]) -> torch.Tensor:
        """Compute rewards for generated responses based on given criteria"""
        rewards = []
        for response in responses:
            reward = 0
            # Add your reward computation logic here based on reward_criteria
            # Example: Check for keywords, length, quality metrics etc.
            rewards.append(reward)
        return torch.tensor(rewards, device=self.device)

    def reinforce_step(
        self, 
        prompt: str, 
        reward_criteria: Dict[str, float],
        learning_rate: float = 1e-5
    ) -> Tuple[str, float]:
        """Perform one step of REINFORCE algorithm"""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Generate response and compute log probabilities
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs)

        # Generate response for reward computation
        response = self.generate_response(prompt)
        reward = self.compute_rewards([response], reward_criteria)[0]

        # Compute REINFORCE loss
        loss = -log_probs * reward

        # Optimization step
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        self.model.eval()
        return response, loss.mean().item()

    def train_rl(
        self,
        prompts: List[str],
        reward_criteria: Dict[str, float],
        n_epochs: int = 5,
        learning_rate: float = 1e-5
    ) -> List[Dict[str, Any]]:
        """Train the model using REINFORCE algorithm"""
        training_history = []
        
        for epoch in range(n_epochs):
            epoch_losses = []
            epoch_rewards = []
            
            for prompt in prompts:
                response, loss = self.reinforce_step(prompt, reward_criteria, learning_rate)
                reward = self.compute_rewards([response], reward_criteria)[0]
                
                epoch_losses.append(loss)
                epoch_rewards.append(reward.item())
            
            history = {
                'epoch': epoch,
                'mean_loss': np.mean(epoch_losses),
                'mean_reward': np.mean(epoch_rewards)
            }
            training_history.append(history)
            
        return training_history

    def save_model(self, path: str):
        """Save the fine-tuned model"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path: str):
        """Load a fine-tuned model"""
        self.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)