from typing import List, Dict, Tuple

def check_and_truncate_prompt(prompt: str, agent_tokenizer, max_model_len: int = 8192, reserve_tokens: int = 500, enable_check: bool = True) -> str:
    """
    Check prompt token length and automatically truncate if exceeding limit
    
    Args:
        prompt: Original prompt text
        agent_tokenizer: Tokenizer instance
        max_model_len: Maximum model length limit
        reserve_tokens: Number of tokens to reserve for generation
        enable_check: Whether to enable token length checking
    
    Returns:
        Truncated prompt text
    """
    if not enable_check:
        return prompt
    
    # Calculate current prompt token count
    tokens = agent_tokenizer(prompt, return_tensors="pt")
    current_tokens = tokens["input_ids"].shape[1]
    
    # Calculate maximum allowed input token count
    max_input_tokens = max_model_len - reserve_tokens
    
    if current_tokens <= max_input_tokens:
        # print(f"[Token Check] Prompt token count: {current_tokens}/{max_model_len}, no truncation needed")
        return prompt
    
    # print(f"[Token Check] Prompt token count: {current_tokens}/{max_model_len}, exceeding limit, starting truncation...")
    
    # If exceeding limit, need to truncate
    # Strategy: Keep instruction part, truncate user record part
    instruction_parts = [
        "You are an expert at summarizing user preferences. Please update the user's preference summary step by step based on their historical records.\n",
        "\n**Task:**\nUpdate the user preference summary in concise English, using markdown format. Only output the updated summary."
    ]
    
    # Calculate instruction part token count
    instruction_text = "".join(instruction_parts)
    instruction_tokens = agent_tokenizer(instruction_text, return_tensors="pt")["input_ids"].shape[1]
    
    # Calculate available tokens for user records
    available_tokens = max_input_tokens - instruction_tokens
    
    if available_tokens <= 0:
        # print(f"[Token Check] Warning: Insufficient available tokens, keeping only instruction part")
        return instruction_text
    
    # Find user record section
    if "**Current User Preference Summary:**" in prompt:
        # Case with current summary
        summary_start = prompt.find("**Current User Preference Summary:**")
        summary_end = prompt.find("**New User Records")
        if summary_end == -1:
            summary_end = prompt.find("**Task:**")
        
        summary_text = prompt[summary_start:summary_end]
        summary_tokens = agent_tokenizer(summary_text, return_tensors="pt")["input_ids"].shape[1]
        
        # Recalculate available tokens for records
        available_tokens = max_input_tokens - instruction_tokens - summary_tokens
        
        if available_tokens <= 0:
            # print(f"[Token Check] Warning: Summary uses too many tokens, keeping only instruction and summary")
            return instruction_parts[0] + summary_text + instruction_parts[1]
        
        # Truncate user record section
        records_start = prompt.find("**New User Records")
        records_end = prompt.find("**Task:**")
        records_text = prompt[records_start:records_end]
        
        # Gradually reduce record count until token limit is satisfied
        lines = records_text.split('\n')
        truncated_lines = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = agent_tokenizer(line + '\n', return_tensors="pt")["input_ids"].shape[1]
            if current_tokens + line_tokens <= available_tokens:
                truncated_lines.append(line)
                current_tokens += line_tokens
            else:
                break
        
        truncated_records = '\n'.join(truncated_lines)
        # print(f"[Token Check] Truncated record count: {len(truncated_lines)}, token count: {current_tokens}")
        
        return instruction_parts[0] + summary_text + truncated_records + instruction_parts[1]
    else:
        # Case without current summary, directly truncate user records
        records_start = prompt.find("**New User Records")
        records_end = prompt.find("**Task:**")
        records_text = prompt[records_start:records_end]
        
        # Gradually reduce record count until token limit is satisfied
        lines = records_text.split('\n')
        truncated_lines = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = agent_tokenizer(line + '\n', return_tensors="pt")["input_ids"].shape[1]
            if current_tokens + line_tokens <= available_tokens:
                truncated_lines.append(line)
                current_tokens += line_tokens
            else:
                break
        
        truncated_records = '\n'.join(truncated_lines)
        # print(f"[Token Check] Truncated record count: {len(truncated_lines)}, token count: {current_tokens}")
        
        return instruction_parts[0] + truncated_records + instruction_parts[1]

def llm_summarize(summary: str, records: List[str], agent_tokenizer, agent_llm, agent_sampling_params) -> str:
    """
    summary: current summary (str), empty string for the first round
    records: the k profile records to summarize (list of str)
    agent_tokenizer: tokenizer instance
    agent_llm: LLM instance
    agent_sampling_params: sampling params for LLM
    """
    prompt = "You are an expert at summarizing user preferences. Please update the user's preference summary step by step based on their historical records.\n"
    if summary:
        prompt += f"\n**Current User Preference Summary:**\n{summary}\n"
    prompt += f"\n**New User Records ({len(records)}):**\n"
    for i, rec in enumerate(records):
        prompt += f"- {rec}\n"
    prompt += "\n**Task:**\nUpdate the user preference summary in concise English, using markdown format. Only output the updated summary."
    
    # Check and truncate prompt
    max_model_len = 8192
    prompt = check_and_truncate_prompt(prompt, agent_tokenizer, max_model_len, enable_check=True)
    
    message = [{"role": "user", "content": prompt}]
    chat_prompt = agent_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    outputs = agent_llm.generate([chat_prompt], agent_sampling_params)
    return outputs[0].outputs[0].text.strip()

def llm_batch_summarize(user_profiles: Dict[str, List], agent_tokenizer, agent_llm, agent_sampling_params, summary_k: int, enable_token_check: bool = True, reserve_tokens: int = 500) -> Dict[str, str]:
    """
    Batch process multiple users' profile summaries
    
    Args:
        user_profiles: Dict[user_id, profile_list] - Mapping from user ID to profile list
        agent_tokenizer: Tokenizer instance
        agent_llm: LLM instance
        agent_sampling_params: LLM sampling parameters
        summary_k: Number of records processed per summary
        enable_token_check: Whether to enable token length checking
        reserve_tokens: Number of tokens to reserve for generation
    
    Returns:
        Dict[user_id, summary] - Mapping from user ID to summary
    """
    user_summaries = {user_id: "" for user_id in user_profiles.keys()}
    
    # Collect all batches that need processing
    all_batches = []
    batch_to_user = {}  # Record which user each batch belongs to
    
    for user_id, profile in user_profiles.items():
        for i in range(0, len(profile), summary_k):
            records = profile[i:i+summary_k]
            record_texts = [str(r) for r in records]
            batch_id = f"{user_id}_batch_{i//summary_k}"
            all_batches.append((batch_id, record_texts))
            batch_to_user[batch_id] = user_id
    
    print(f"[Batch Summary] Total batches to process: {len(all_batches)}, involving {len(user_profiles)} users")
    
    # Batch process all batches
    for batch_id, record_texts in all_batches:
        user_id = batch_to_user[batch_id]
        current_summary = user_summaries[user_id]
        
        # Build prompt
        prompt = "You are an expert at summarizing user preferences. Please update the user's preference summary step by step based on their historical records.\n"
        if current_summary:
            prompt += f"\n**Current User Preference Summary:**\n{current_summary}\n"
        prompt += f"\n**New User Records ({len(record_texts)}):**\n"
        for i, rec in enumerate(record_texts):
            prompt += f"- {rec}\n"
        prompt += "\n**Task:**\nUpdate the user preference summary in concise English, using markdown format. Only output the updated summary."
        
        # Check and truncate prompt
        max_model_len = 8192
        prompt = check_and_truncate_prompt(prompt, agent_tokenizer, max_model_len, reserve_tokens, enable_token_check)
        
        message = [{"role": "user", "content": prompt}]
        chat_prompt = agent_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        
        # Generate summary
        outputs = agent_llm.generate([chat_prompt], agent_sampling_params)
        new_summary = outputs[0].outputs[0].text.strip()
        
        # Update user summary
        user_summaries[user_id] = new_summary
        
        # print(f"[Batch Summary] Completed batch {batch_id.split('_')[-1]} for user {user_id}")
    
    return user_summaries

def llm_batch_summarize_parallel(user_profiles: Dict[str, List], agent_tokenizer, agent_llm, agent_sampling_params, summary_k: int, batch_size: int = 4, enable_token_check: bool = True, reserve_tokens: int = 500) -> Dict[str, str]:
    """
    Parallel batch processing of multiple users' profile summaries using vLLM's batch inference capability
    
    Args:
        user_profiles: Dict[user_id, profile_list] - Mapping from user ID to profile list
        agent_tokenizer: Tokenizer instance
        agent_llm: LLM instance
        agent_sampling_params: LLM sampling parameters
        summary_k: Number of records processed per summary
        batch_size: Batch size for parallel processing
        enable_token_check: Whether to enable token length checking
        reserve_tokens: Number of tokens to reserve for generation
    
    Returns:
        Dict[user_id, summary] - Mapping from user ID to summary
    """
    user_summaries = {user_id: "" for user_id in user_profiles.keys()}
    
    # Collect all batches that need processing
    all_batches = []
    batch_to_user = {}  # Record which user each batch belongs to
    batch_to_summary = {}  # Record current summary state for each batch
    
    for user_id, profile in user_profiles.items():
        for i in range(0, len(profile), summary_k):
            records = profile[i:i+summary_k]
            record_texts = [str(r) for r in records]
            batch_id = f"{user_id}_batch_{i//summary_k}"
            all_batches.append(batch_id)
            batch_to_user[batch_id] = user_id
            batch_to_summary[batch_id] = user_summaries[user_id]
    
    # print(f"[Parallel Batch Summary] Total batches to process: {len(all_batches)}, involving {len(user_profiles)} users")
    
    # Process in groups by batch size
    for i in range(0, len(all_batches), batch_size):
        batch_group = all_batches[i:i+batch_size]
        
        # Prepare batch prompts
        prompts = []
        batch_ids = []
        
        for batch_id in batch_group:
            user_id = batch_to_user[batch_id]
            current_summary = batch_to_summary[batch_id]
            
            # Get records for this batch
            batch_num = int(batch_id.split('_')[-1])
            profile = user_profiles[user_id]
            start_idx = batch_num * summary_k
            end_idx = min(start_idx + summary_k, len(profile))
            records = profile[start_idx:end_idx]
            record_texts = [str(r) for r in records]
            
            # Build prompt
            prompt = "You are an expert at summarizing user preferences. Please update the user's preference summary step by step based on their historical records.\n"
            if current_summary:
                prompt += f"\n**Current User Preference Summary:**\n{current_summary}\n"
            prompt += f"\n**New User Records ({len(record_texts)}):**\n"
            for j, rec in enumerate(record_texts):
                prompt += f"- {rec}\n"
            prompt += "\n**Task:**\nUpdate the user preference summary in concise English, using markdown format. Only output the updated summary."
            
            # Check and truncate prompt
            max_model_len = 8192
            prompt = check_and_truncate_prompt(prompt, agent_tokenizer, max_model_len, reserve_tokens, enable_token_check)
            
            message = [{"role": "user", "content": prompt}]
            chat_prompt = agent_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            
            prompts.append(chat_prompt)
            batch_ids.append(batch_id)
        
        # Batch generation
        outputs = agent_llm.generate(prompts, agent_sampling_params)
        
        # Update results
        for batch_id, output in zip(batch_ids, outputs):
            user_id = batch_to_user[batch_id]
            new_summary = output.outputs[0].text.strip()
            user_summaries[user_id] = new_summary
            
            # Update summary state for subsequent batches
            batch_num = int(batch_id.split('_')[-1])
            for future_batch_id in batch_to_summary:
                if (batch_to_user[future_batch_id] == user_id and 
                    int(future_batch_id.split('_')[-1]) > batch_num):
                    batch_to_summary[future_batch_id] = new_summary
        
        print(f"[Parallel Batch Summary] Completed batch {i//batch_size + 1}/{(len(all_batches) + batch_size - 1)//batch_size}")
    
    return user_summaries