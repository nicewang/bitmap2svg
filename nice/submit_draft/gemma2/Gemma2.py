from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Gemma2:
    def __init__(self, model_name="google/gemma-2b-it"):
        """
        Initialize Gemma2 with the specified model.
        
        Args:
            model_name (str): Name of the model to load (default: "google/gemma-2b-it")
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )

    def generate(self, prompt, max_length=2048, temperature=0.9, top_p=0.9):
        """
        Generate text based on the input prompt.
        
        Args:
            prompt (str): Input prompt for text generation
            max_length (int): Maximum length of generated text
            temperature (float): Controls randomness in generation
            top_p (float): Controls diversity of generated text
            
        Returns:
            str: Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def chat(self, messages, max_length=2048, temperature=0.9, top_p=0.9):
        """
        Generate chat response based on message history.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            max_length (int): Maximum length of generated text
            temperature (float): Controls randomness in generation
            top_p (float): Controls diversity of generated text
            
        Returns:
            str: Generated response
        """
        # Format messages into a prompt
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant: "
        
        return self.generate(prompt, max_length, temperature, top_p)