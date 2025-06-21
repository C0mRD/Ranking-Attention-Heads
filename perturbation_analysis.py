import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

class AttentionHeadNoiseHook:
    """Hook class to add noise to specific attention heads"""
    def __init__(self, target_head, noise_scale=2.0, num_heads=32):
        self.target_head = target_head
        self.noise_scale = noise_scale
        self.num_heads = num_heads
        self.hook_applied = False
    
    def __call__(self, module, input, output):
        """Hook function to add noise to the target attention head"""
        # output is the attention output after projection
        # Shape: (batch_size, seq_len, hidden_size)
        
        # Get dimensions
        batch_size, seq_len, hidden_size = output.shape
        head_dim = hidden_size // self.num_heads
        
        # Reshape to separate heads: (batch, seq, num_heads, head_dim)
        reshaped = output.view(batch_size, seq_len, self.num_heads, head_dim)
        
        # Add noise to the target attention head only
        noise = torch.randn_like(reshaped[:, :, self.target_head, :]) * self.noise_scale
        reshaped[:, :, self.target_head, :] += noise
        
        # Reshape back and modify the output in place
        output.data = reshaped.view(batch_size, seq_len, hidden_size)
        
        self.hook_applied = True

def evaluate_model_with_hook(model, tokenizer, eval_data, few_shot_prompt, target_head=None, noise_scale=2.0):
    """Evaluate model performance with optional attention head intervention"""
    predictions = []
    true_labels = []
    
    # Set up hook if needed
    hook_handle = None
    if target_head is not None:
        hook_fn = AttentionHeadNoiseHook(target_head=target_head, noise_scale=noise_scale, num_heads=32)
        # Register hook on the last layer's attention output projection
        last_layer = model.model.layers[15]  # Last layer for Llama-3.2-1B
        hook_handle = last_layer.self_attn.o_proj.register_forward_hook(hook_fn)
    
    try:
        for idx, row in eval_data.iterrows():
            # Create prompt
            test_review = row['review'][:100] + "..." if len(row['review']) > 100 else row['review']
            prompt = few_shot_prompt + f"Review: {test_review}\nSentiment:"
            
            # Get prediction
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=3, do_sample=False, 
                                        pad_token_id=tokenizer.eos_token_id)
            
            generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], 
                                       skip_special_tokens=True).strip().lower()
            
            # Extract prediction
            if 'positive' in generated:
                pred = 'positive'
            elif 'negative' in generated:
                pred = 'negative'
            else:
                pred = 'unknown'
            
            # Debug printing
            if target_head is not None:
                print(f"  Generated: '{generated}' -> Extracted: '{pred}' (True: {row['sentiment']})")
            
            predictions.append(pred)
            true_labels.append(row['sentiment'])
    
    finally:
        # Always remove the hook
        if hook_handle is not None:
            hook_handle.remove()
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    
    # Check what unique values we have
    unique_preds = set(predictions)
    unique_true = set(true_labels)
    
    print(f"  Unique predictions: {unique_preds}")
    print(f"  Unique true labels: {unique_true}")
    
    # Handle f1_score calculation based on what we actually have
    if len(unique_preds) == 2 and len(unique_true) == 2 and 'unknown' not in unique_preds:
        f1 = f1_score(true_labels, predictions, pos_label='positive')
    else:
        # Use macro average for multiclass or when we have unknown predictions
        f1 = f1_score(true_labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'predictions': predictions,
        'true_labels': true_labels
    }

def main():
    # Configuration
    model_name = 'meta-llama/Llama-3.2-1B'
    n_samples = 30
    few_shot_k = 5
    noise_scale = 2.0  # Large noise for significant impact
    
    print("="*80)
    print("ATTENTION HEAD PERTURBATION ANALYSIS")
    print("="*80)
    
    print("Loading IMDB dataset...")
    df = pd.read_csv('IMDB_Dataset_100.csv')
    df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    print(f"Loaded {len(df)} samples")
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, cache_dir="./models")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded!")
    
    # Create few-shot examples
    pos_examples = df[df['sentiment'] == 'positive'].head(2)
    neg_examples = df[df['sentiment'] == 'negative'].head(2)
    
    few_shot_prompt = ""
    for _, row in pos_examples.iterrows():
        review = row['review'][:100] + "..." if len(row['review']) > 100 else row['review']
        few_shot_prompt += f"Review: {review}\nSentiment: positive\n\n"
    
    for _, row in neg_examples.iterrows():
        review = row['review'][:100] + "..." if len(row['review']) > 100 else row['review']
        few_shot_prompt += f"Review: {review}\nSentiment: negative\n\n"
    
    print("Few-shot examples created")
    
    # Evaluate on remaining data
    eval_data = df.iloc[few_shot_k:].reset_index(drop=True)
    
    # Baseline Evaluation
    print("\nRunning baseline evaluation...")
    baseline_results = evaluate_model_with_hook(model, tokenizer, eval_data, few_shot_prompt)
    result_original = baseline_results['accuracy']
    
    print(f"Baseline Results:")
    print(f"Accuracy:  {baseline_results['accuracy']:.4f}")
    print(f"F1 Score:  {baseline_results['f1']:.4f}")
    
    # Show baseline predictions for reference
    print(f"\nBaseline Predictions vs True Labels:")
    for i, (pred, true) in enumerate(zip(baseline_results['predictions'], baseline_results['true_labels'])):
        print(f"  Sample {i}: Predicted='{pred}', True='{true}'")
    
    # Perturbation Analysis of Last Layer Attention Heads
    print(f"\nPerturbation analysis of last layer attention heads...")
    print(f"Using noise scale: {noise_scale}")
    print(f"Testing {32} attention heads in last layer (layer 15)")
    
    num_heads = 32  # Llama has 32 attention heads
    head_contributions = []
    
    for head_idx in tqdm(range(num_heads), desc="Analyzing attention heads"):
        print(f"\nAnalyzing attention head {head_idx}")
        
        # Evaluate with intervention on this specific head
        perturbed_results = evaluate_model_with_hook(
            model, tokenizer, eval_data, few_shot_prompt, 
            target_head=head_idx, noise_scale=noise_scale
        )
        
        result_attn_i = perturbed_results['accuracy']
        
        # Calculate Operational Contribution
        performance_drop = result_original - result_attn_i
        head_contributions.append({
            'head_idx': head_idx,
            'original_accuracy': result_original,
            'perturbed_accuracy': result_attn_i,
            'performance_drop': performance_drop,
            'contribution_score': performance_drop  # Higher drop = more important
        })
        
        print(f"Head {head_idx}: Original={result_original:.4f}, "
              f"Perturbed={result_attn_i:.4f}, Drop={performance_drop:.4f}")
    
    # Rank Attention Heads
    print(f"\nRanking attention heads by operational contribution...")
    
    # Sort by performance drop (descending order)
    ranked_heads = sorted(head_contributions, key=lambda x: x['contribution_score'], reverse=True)
    
    print(f"\n{'='*80}")
    print(f"FINAL RANKING - ATTENTION HEADS BY OPERATIONAL IMPORTANCE")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Head':<6} {'Original':<10} {'Perturbed':<10} {'Drop':<10} {'Impact':<10}")
    print(f"{'-'*80}")
    
    for rank, head_info in enumerate(ranked_heads, 1):
        impact_level = "HIGH" if head_info['contribution_score'] > 0.1 else \
                      "MEDIUM" if head_info['contribution_score'] > 0.05 else "LOW"
        
        print(f"{rank:<6} {head_info['head_idx']:<6} "
              f"{head_info['original_accuracy']:<10.4f} "
              f"{head_info['perturbed_accuracy']:<10.4f} "
              f"{head_info['contribution_score']:<10.4f} "
              f"{impact_level:<10}")
    
    # Save results
    results_df = pd.DataFrame(ranked_heads)
    results_df['rank'] = range(1, len(ranked_heads) + 1)
    results_df.to_csv('attention_head_ranking_hooks.csv', index=False)
    print(f"\nResults saved to 'attention_head_ranking_hooks.csv'")
    
    # Summary statistics
    print(f"\n{'='*50}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*50}")
    print(f"Total heads analyzed: {num_heads}")
    print(f"Average performance drop: {np.mean([h['contribution_score'] for h in ranked_heads]):.4f}")
    print(f"Max performance drop: {max([h['contribution_score'] for h in ranked_heads]):.4f}")
    print(f"Min performance drop: {min([h['contribution_score'] for h in ranked_heads]):.4f}")
    
    high_impact_heads = [h for h in ranked_heads if h['contribution_score'] > 0.1]
    print(f"High impact heads (>10% drop): {len(high_impact_heads)}")
    if high_impact_heads:
        print(f"Most critical head: Head {high_impact_heads[0]['head_idx']} "
              f"(drop: {high_impact_heads[0]['contribution_score']:.4f})")
    
    # Show top 5 most important heads
    print(f"\n{'='*50}")
    print(f"TOP 5 MOST IMPORTANT ATTENTION HEADS")
    print(f"{'='*50}")
    for i, head_info in enumerate(ranked_heads[:5], 1):
        print(f"{i}. Head {head_info['head_idx']}: {head_info['contribution_score']:.4f} drop")

if __name__ == "__main__":
    main()