import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any


class UserAwareLLM(nn.Module):
    """
    User-aware LLM wrapper that incorporates user feature vectors during generation
    """
    
    def __init__(self, 
                 model_path: str,
                 user_emb_path: str,
                 user2id: Dict[str, int],
                 device: str,
                 fusion_method: str = "attention",
                 user_weight: float = 0.3):
        super().__init__()
        
        # Load base LLM
        self.llm = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # User-related components
        self.user_embedding = torch.load(user_emb_path).to(device)
        self.user2id = user2id
        self.device = device
        self.fusion_method = fusion_method
        self.user_weight = user_weight
        
        # User feature mapping layer
        emb_dim = self.llm.config.hidden_size
        self.user_map = nn.Linear(self.user_embedding.shape[1], emb_dim)
        
        # Feature fusion layer
        if fusion_method == "concat":
            self.fusion_layer = nn.Sequential(
                nn.Linear(emb_dim * 2, emb_dim),
                nn.Tanh(),
                nn.Linear(emb_dim, emb_dim)
            )
        elif fusion_method == "attention":
            self.user_attention = nn.MultiheadAttention(
                embed_dim=emb_dim,
                num_heads=8,
                batch_first=True
            )
            self.user_norm = nn.LayerNorm(emb_dim)
        
        # Ensure pad_token is set correctly
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.to(device)
        
    def get_user_embedding(self, user_id: str) -> torch.Tensor:
        """Get user embedding vector"""
        if user_id not in self.user2id:
            # If user does not exist, use average embedding
            return self.user_embedding.mean(dim=0, keepdim=True)
        
        user_idx = self.user2id[user_id]
        user_emb = self.user_embedding[[user_idx]]
        return self.user_map(user_emb)
    
    def fuse_user_features(self, 
                          hidden_states: torch.Tensor, 
                          user_emb: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fuse user features into hidden states"""
        
        if self.fusion_method == "concat":
            # Method 1: Concatenation fusion
            user_emb_expanded = user_emb.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
            fused_hidden = self.fusion_layer(torch.cat([hidden_states, user_emb_expanded], dim=-1))
            return fused_hidden
            
        elif self.fusion_method == "attention":
            # Method 2: Attention fusion
            user_emb_expanded = user_emb.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
            
            # Use user features as query, hidden states as key and value
            attn_output, _ = self.user_attention(
                query=user_emb_expanded,
                key=hidden_states,
                value=hidden_states,
                key_padding_mask=attention_mask if attention_mask is not None else None
            )
            
            # Residual connection and layer normalization
            fused_hidden = self.user_norm(hidden_states + self.user_weight * attn_output)
            return fused_hidden
            
        elif self.fusion_method == "add":
            # Method 3: Simple addition
            user_emb_expanded = user_emb.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
            return hidden_states + self.user_weight * user_emb_expanded
            
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                user_id: Optional[str] = None,
                **kwargs) -> Dict[str, Any]:
        """Forward pass, incorporating user features"""
        
        # Get user embedding
        if user_id is not None:
            user_emb = self.get_user_embedding(user_id)
        else:
            # If no user ID, use zero vector
            user_emb = torch.zeros(1, self.llm.config.hidden_size).to(self.device)
        
        # LLM forward pass
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # Get last layer hidden states
        last_hidden = outputs.hidden_states[-1]
        
        # Fuse user features
        fused_hidden = self.fuse_user_features(last_hidden, user_emb, attention_mask)
        
        # Update output
        outputs.hidden_states = list(outputs.hidden_states)
        outputs.hidden_states[-1] = fused_hidden
        
        return outputs
    
    def generate(self, 
                 input_ids: torch.Tensor,
                 attention_mask: Optional[torch.Tensor] = None,
                 user_id: Optional[str] = None,
                 **kwargs) -> torch.Tensor:
        """Generation method with user feature support"""
        
        # Override generation method to support user features
        def custom_forward(input_ids, attention_mask=None, **forward_kwargs):
            return self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                user_id=user_id,
                **forward_kwargs
            )
        
        # Temporarily replace model's forward method
        original_forward = self.llm.forward
        self.llm.forward = custom_forward
        
        try:
            # Use original LLM's generation method
            generated_ids = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        finally:
            # Restore original forward method
            self.llm.forward = original_forward
        
        return generated_ids