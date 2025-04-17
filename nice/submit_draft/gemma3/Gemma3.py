# Make sure to run this installation command if not already installed:
#
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Gemma3Model:
    
    def __init__(self, gemma_path: str):
        
        self.gemma_path = gemma_path
        self.tokenizer = AutoTokenizer.from_pretrained(gemma_path)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        self.model = Gemma3ForCausalLM.from_pretrained(
            gemma_path,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.model.gradient_checkpointing_enable()

        print(self.model.hf_device_map)

    def generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        
        input_ids = self.tokenizer(text=prompt, return_tensors="pt")
        # Move the input tensor to the correct device
        input_ids_0 = input_ids.to("cuda:0")
        # input_ids_1 = input_ids.to("cuda:1")
        # ...
        # input_ids_n = input_ids.to("cuda:n")
        # Thus, we could make sure that the input tensor has a copy on both GPUs
        outputs = self.model.generate(**input_ids_0, max_new_tokens=max_new_tokens)
        generated_text = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        return generated_text

    def optimize_code(self, code: str, custom_prompt: str = None, max_new_tokens: int = 256) -> str:
        
        if custom_prompt is None:
            # Default prompt instructs the model to provide multiple sections in its output.
            prompt = (
                "<start_of_turn>user:\n"
                "I have the following code:\n"
                f"{code}\n"
                "Could you provide an optimized version of this code along with detailed technical explanations? "
                "Include the following sections in your response:\n"
                "1. Optimized Code\n"
                "2. Technical Details\n"
                "3. Algorithmic Complexity (Big-O Analysis)\n"
                "4. Real World Application\n"
                "<end_of_turn>\n"
                "<start_of_turn>model:"
            )
        else:
            # Use the custom prompt provided by the user.
            prompt = (
                "<start_of_turn>user:\n"
                f"{custom_prompt}\n"
                f"{code}\n"
                "<end_of_turn>\n"
                "<start_of_turn>model:"
            )
        
        return self.generate_response(prompt, max_new_tokens=max_new_tokens)
