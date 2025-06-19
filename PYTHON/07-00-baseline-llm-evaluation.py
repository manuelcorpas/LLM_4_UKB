#!/usr/bin/env python3
"""
Working baseline evaluation using the successful parsing fix
"""

import pandas as pd
import numpy as np
import random
from collections import Counter
from typing import List, Dict

def load_uk_biobank_data():
    """Load the UK Biobank data using the working parsing approach"""
    print("📁 Loading UK Biobank Schema files...")
    
    # Schema 19 - Publications
    schema19 = pd.read_csv("DATA/schema_19.txt", sep='\t', low_memory=False)
    print(f"✅ Schema 19 (Publications): {schema19.shape}")
    
    # Schema 27 - Applications (using skip bad lines)
    schema27 = pd.read_csv("DATA/schema_27.txt", sep='\t', on_bad_lines='skip', low_memory=False)
    print(f"✅ Schema 27 (Applications): {schema27.shape}")
    
    return schema19, schema27

def extract_baseline_terms(schema19, schema27):
    """Extract terms for baseline generation"""
    print("\n🔍 Extracting terms for baseline generation...")
    
    # Extract keywords
    keywords = set()
    for keyword_string in schema19['keywords'].dropna():
        try:
            terms = [term.strip() for term in str(keyword_string).split('|')]
            keywords.update(terms)
        except:
            continue
    keywords = [k for k in keywords if len(k) > 2 and k not in ['', 'nan']]
    
    # Extract authors
    authors = set()
    for author_string in schema19['authors'].dropna():
        try:
            names = [name.strip() for name in str(author_string).split('|')]
            authors.update(names)
        except:
            continue
    authors = [a for a in authors if len(a) > 3 and a not in ['', 'nan']]
    
    # Extract paper titles
    papers = [str(title).strip() for title in schema19['title'].dropna() 
              if len(str(title).strip()) > 10 and str(title) != 'nan']
    
    # Extract institutions
    institutions = [str(inst).strip() for inst in schema27['institution'].dropna() 
                   if len(str(inst).strip()) > 3 and str(inst) != 'nan']
    # Remove duplicates
    institutions = list(dict.fromkeys(institutions))
    
    print(f"📊 Extracted terms:")
    print(f"  Keywords: {len(keywords)}")
    print(f"  Authors: {len(authors)}")
    print(f"  Papers: {len(papers)}")
    print(f"  Institutions: {len(institutions)}")
    
    return {
        'keywords': keywords,
        'authors': authors,
        'papers': papers,
        'institutions': institutions
    }

def get_ground_truth_terms(schema19, schema27, task, top_n=20):
    """Get ground truth terms for evaluation"""
    
    if task == 'keywords':
        # Count keyword frequencies
        keyword_counts = Counter()
        for keyword_string in schema19['keywords'].dropna():
            try:
                terms = [term.strip() for term in str(keyword_string).split('|')]
                keyword_counts.update(terms)
            except:
                continue
        return [kw for kw, count in keyword_counts.most_common(top_n) if kw not in ['', 'nan']]
    
    elif task == 'papers':
        # Get most cited papers
        schema19_clean = schema19.copy()
        schema19_clean['cite_total'] = pd.to_numeric(schema19_clean['cite_total'], errors='coerce')
        top_papers = schema19_clean.nlargest(top_n, 'cite_total')
        return top_papers['title'].dropna().tolist()
    
    elif task == 'authors':
        # Count author frequencies
        author_counts = Counter()
        for author_string in schema19['authors'].dropna():
            try:
                names = [name.strip() for name in str(author_string).split('|')]
                author_counts.update(names)
            except:
                continue
        return [author for author, count in author_counts.most_common(top_n) if author not in ['', 'nan']]
    
    elif task == 'institutions':
        # Count institution frequencies
        institution_counts = schema27['institution'].value_counts()
        return institution_counts.head(top_n).index.tolist()
    
    return []

def generate_baseline_response(task, terms, num_items=20):
    """Generate a baseline response by randomly sampling terms"""
    
    if len(terms) < num_items:
        sampled = terms
    else:
        sampled = random.sample(terms, num_items)
    
    # Create natural language response
    if task == 'keywords':
        intro = "The most commonly occurring keywords in UK Biobank papers include: "
        response = intro + ", ".join(sampled[:15])
        if len(sampled) > 15:
            response += f", and {len(sampled) - 15} other related terms."
        response += " These reflect the diverse research areas covered by UK Biobank studies."
        
    elif task == 'papers':
        intro = "The most cited papers relating to the UK Biobank include:\n"
        response = intro
        for i, paper in enumerate(sampled[:10], 1):
            response += f"{i}. {paper}\n"
            
    elif task == 'authors':
        intro = "The top prolific authors publishing on the UK Biobank include: "
        response = intro + ", ".join(sampled[:15])
        if len(sampled) > 15:
            response += f", among {len(sampled)} other researchers."
        response += " These researchers have made significant contributions to biobank-related studies."
        
    elif task == 'institutions':
        intro = "The leading institutions in terms of UK Biobank applications include:\n"
        response = intro
        for i, inst in enumerate(sampled[:10], 1):
            response += f"{i}. {inst}\n"
    
    return response

def calculate_coverage_score(response, reference_terms):
    """Calculate coverage score using simple keyword matching"""
    response_lower = response.lower()
    
    # Basic coverage
    matches = 0
    for term in reference_terms:
        if term.lower() in response_lower:
            matches += 1
    
    coverage = matches / len(reference_terms) if reference_terms else 0
    
    # Weighted coverage (weight by frequency order)
    weighted_matches = 0
    total_weight = 0
    for i, term in enumerate(reference_terms):
        weight = len(reference_terms) - i  # Higher weight for earlier (more frequent) terms
        total_weight += weight
        if term.lower() in response_lower:
            weighted_matches += weight
    
    weighted_coverage = weighted_matches / total_weight if total_weight > 0 else 0
    
    return coverage, weighted_coverage

def run_baseline_evaluation():
    """Run the complete baseline evaluation"""
    
    print("🚀 UK BIOBANK BASELINE EVALUATION")
    print("=" * 50)
    
    # Load data
    schema19, schema27 = load_uk_biobank_data()
    
    # Extract terms for baseline generation
    baseline_terms = extract_baseline_terms(schema19, schema27)
    
    # Define tasks
    tasks = ['keywords', 'papers', 'authors', 'institutions']
    num_baseline_runs = 100
    
    results = []
    
    for task in tasks:
        print(f"\n📋 Evaluating {task}...")
        
        # Get ground truth terms
        ground_truth = get_ground_truth_terms(schema19, schema27, task, 20)
        if not ground_truth:
            print(f"  ⚠️  No ground truth available for {task}, skipping...")
            continue
            
        print(f"  Ground truth: {len(ground_truth)} terms")
        print(f"  Sample: {ground_truth[:3]}...")
        
        # Generate baseline responses
        coverage_scores = []
        weighted_coverage_scores = []
        
        available_terms = baseline_terms.get(task, [])
        if not available_terms:
            print(f"  ⚠️  No baseline terms available for {task}, skipping...")
            continue
        
        for run in range(num_baseline_runs):
            # Generate baseline response
            baseline_response = generate_baseline_response(task, available_terms)
            
            # Calculate coverage
            coverage, weighted_coverage = calculate_coverage_score(baseline_response, ground_truth)
            coverage_scores.append(coverage)
            weighted_coverage_scores.append(weighted_coverage)
        
        # Calculate statistics
        result = {
            'task': task,
            'coverage_mean': np.mean(coverage_scores),
            'coverage_std': np.std(coverage_scores),
            'coverage_min': np.min(coverage_scores),
            'coverage_max': np.max(coverage_scores),
            'weighted_coverage_mean': np.mean(weighted_coverage_scores),
            'weighted_coverage_std': np.std(weighted_coverage_scores),
            'weighted_coverage_min': np.min(weighted_coverage_scores),
            'weighted_coverage_max': np.max(weighted_coverage_scores),
            'num_samples': num_baseline_runs,
            'num_ground_truth_terms': len(ground_truth),
            'num_baseline_terms': len(available_terms)
        }
        
        results.append(result)
        
        print(f"  Coverage: μ={result['coverage_mean']:.3f} ± {result['coverage_std']:.3f} "
              f"(max: {result['coverage_max']:.3f})")
        print(f"  Weighted: μ={result['weighted_coverage_mean']:.3f} ± {result['weighted_coverage_std']:.3f} "
              f"(max: {result['weighted_coverage_max']:.3f})")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("DATA/baseline_evaluation_results.csv", index=False)
    
    # Summary
    print(f"\n📊 BASELINE EVALUATION SUMMARY")
    print("=" * 50)
    
    for result in results:
        task = result['task']
        mean_cov = result['coverage_mean']
        max_cov = result['coverage_max']
        mean_weighted = result['weighted_coverage_mean']
        
        print(f"{task:12}: Coverage μ={mean_cov:.3f} (max={max_cov:.3f}), "
              f"Weighted μ={mean_weighted:.3f}")
    
    overall_mean = np.mean([r['coverage_mean'] for r in results])
    overall_max = np.max([r['coverage_max'] for r in results])
    
    print(f"\nOverall baseline performance:")
    print(f"  Mean coverage: {overall_mean:.3f}")
    print(f"  Maximum coverage achieved: {overall_max:.3f}")
    
    # Implications for paper
    print(f"\n📝 FOR YOUR PAPER:")
    print("=" * 30)
    print(f"Random baseline results:")
    for result in results:
        print(f"  {result['task']:12}: {result['coverage_mean']:.1%} ± {result['coverage_std']:.1%}")
    
    print(f"\nIf your best LLMs achieve:")
    for threshold in [0.4, 0.5, 0.6, 0.7]:
        improvement = threshold / overall_mean
        print(f"  {threshold:.1%} performance = {improvement:.1f}× improvement over baseline")
    
    print(f"\nSuggested paper text:")
    print(f'"Random baseline performance ranged from {min(r["coverage_mean"] for r in results):.1%} to {max(r["coverage_mean"] for r in results):.1%} across tasks. High-performing LLMs significantly outperformed baselines with improvement factors of X.X× to Y.Y×, confirming genuine biobank-specific knowledge acquisition."')
    
    return results_df

if __name__ == "__main__":
    results = run_baseline_evaluation()
    print(f"\n✅ Baseline evaluation complete!")
    print(f"📁 Results saved to: DATA/baseline_evaluation_results.csv")