from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def query_model(model, tokenizer, question):
    # Format the query
    prompt = f"Based on UK Biobank publications, answer this question:\n\n{question}\n\n"
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)
    
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.95
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "ANALYSIS/final_model",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("ANALYSIS/final_model", trust_remote_code=True)
    
    # Your questions
    questions = [
        "Which imaging data fields are most commonly used with genetic data?",
        "What UK Biobank fields are trending in recent publications?", 
        "Which data categories are under-utilized?",
        "How are researchers combining different types of UK Biobank data?"
    ]
    
    # Query each question
    print("\nQuerying the model...\n")
    for question in questions:
        print(f"Question: {question}")
        print("Answer:", query_model(model, tokenizer, question))
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
