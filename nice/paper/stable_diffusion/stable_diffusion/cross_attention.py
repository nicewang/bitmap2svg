import torch
import torch.nn.functional as F

def cross_attention(visual_tokens, condition_tokens, W_q, W_k, W_v, W_o):
    """
    Cross-attention mechanism in Stable Diffusion
    
    Args:
        visual_tokens: [batch_size, m, d_model]  # m visual tokens
        condition_tokens: [batch_size, n, d_model]  # n condition tokens (text embeddings)
        W_q, W_k, W_v: Linear transformation weight matrices
        W_o: Output projection weight matrix
    
    Returns:
        output: [batch_size, m, d_model]  # m updated visual tokens
    """
    
    # 1. Linear transformation to get Q, K, V
    # Query comes from visual tokens (tokens to be updated)
    Q = torch.matmul(visual_tokens, W_q)  # [batch_size, m, d_k]
    
    # Key and Value come from condition tokens (tokens providing information)
    K = torch.matmul(condition_tokens, W_k)  # [batch_size, n, d_k]
    V = torch.matmul(condition_tokens, W_v)  # [batch_size, n, d_v]
    
    # 2. Calculate attention scores
    # Each visual token computes similarity with all condition tokens
    scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, m, n]
    
    # 3. Scaling (optional, for numerical stability)
    d_k = K.size(-1)
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # 4. Softmax normalization
    # Apply softmax separately for each visual token (along the last dimension)
    attention_weights = F.softmax(scores, dim=-1)  # [batch_size, m, n]
    
    # Now attention_weights[i, j, :] represents the weight distribution of the i-th visual token
    # over all condition tokens, and satisfies: sum(attention_weights[i, j, :]) = 1 for all i, j
    
    # 5. Weighted sum
    # Each visual token performs weighted average over condition token values based on attention weights
    attended_values = torch.matmul(attention_weights, V)  # [batch_size, m, d_v]
    
    # 6. Output projection (optional)
    output = torch.matmul(attended_values, W_o)  # [batch_size, m, d_model]
    
    return output

# Usage example
def example_usage():
    batch_size = 2
    m = 64  # 64 visual tokens (e.g., 8x8 spatial patches)
    n = 77  # 77 condition tokens (CLIP text tokens)
    d_model = 512
    
    # Simulate inputs
    visual_tokens = torch.randn(batch_size, m, d_model)
    condition_tokens = torch.randn(batch_size, n, d_model)
    
    # Weight matrices
    W_q = torch.randn(d_model, d_model)
    W_k = torch.randn(d_model, d_model)  
    W_v = torch.randn(d_model, d_model)
    W_o = torch.randn(d_model, d_model)
    
    # Execute cross-attention
    output = cross_attention(visual_tokens, condition_tokens, W_q, W_k, W_v, W_o)
    
    print(f"Input visual tokens shape: {visual_tokens.shape}")
    print(f"Input condition tokens shape: {condition_tokens.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verification: each visual token in the output has been updated with condition information
    assert output.shape == visual_tokens.shape

# Key points:
"""
1. Each visual token (spatial location) attends to all condition tokens
2. Softmax ensures that attention weights for each visual token sum to 1
3. Different visual tokens can focus on different condition information
4. Final result contains m updated visual tokens, preserving spatial structure
5. This enables different spatial locations to selectively use text condition information
"""