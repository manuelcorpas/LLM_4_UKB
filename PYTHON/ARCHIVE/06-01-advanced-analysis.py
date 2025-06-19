#!/usr/bin/env python3
"""
Enhanced Statistical Analysis Framework for LLM Evaluation Results
Adapted to work with the output from 06-00-enhanced-llm-evaluation.py (actual_llm_data.csv)

Addresses reviewer concerns about statistical rigor through:
1. Multiple comparisons correction (Benjamini-Hochberg FDR)
2. Comprehensive effect size calculations (Cohen's d, Cliff's delta)
3. Confidence intervals for all key metrics
4. Pairwise model comparisons with proper statistical validation
5. Robust baseline comparisons with multiple testing correction
6. Works specifically with actual_llm_data.csv output
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, t, norm
from statsmodels.stats.multitest import multipletests
import itertools
import os
import json
import warnings
warnings.filterwarnings('ignore')

class EnhancedStatisticalAnalyzer:
    """Enhanced statistical analyzer adapted for LLM evaluation results."""
    
    def __init__(self, csv_file: str = None):
        """Initialize with LLM evaluation results from actual_llm_data.csv."""
        
        # Default file path based on the enhanced evaluation script output
        if csv_file is None:
            possible_files = [
                'actual_llm_data.csv',
                'RESULTS/REAL_LLM_EVALUATION/actual_llm_data.csv',
                'RESULTS/00-ENHANCED_LLM_EVALUATION/actual_llm_data.csv'
            ]
            
            csv_file = None
            for file in possible_files:
                if os.path.exists(file):
                    csv_file = file
                    print(f"✅ Found evaluation results: {file}")
                    break
                else:
                    print(f"🔍 Checking: {file} - Not found")
            
            if csv_file is None:
                raise FileNotFoundError("❌ No actual_llm_data.csv found. Please run the enhanced evaluation script first.")
        
        try:
            self.df = pd.read_csv(csv_file)
            print(f"✅ Loaded {len(self.df)} evaluation records")
            print(f"📋 Columns: {list(self.df.columns)}")
        except Exception as e:
            raise Exception(f"❌ Error loading data: {e}")
        
        # Define key metrics based on the actual CSV output
        self.key_metrics = [
            'semantic_accuracy', 'reasoning_quality', 'domain_knowledge_score',
            'factual_correctness', 'depth_score', 'biobank_specificity'
        ]
        
        # Filter metrics that actually exist in the data
        available_metrics = [m for m in self.key_metrics if m in self.df.columns]
        if not available_metrics:
            print("❌ No expected metrics found in data!")
            print(f"Available numeric columns: {self.df.select_dtypes(include=[np.number]).columns.tolist()}")
            return
        
        self.key_metrics = available_metrics
        print(f"📊 Using {len(self.key_metrics)} metrics: {self.key_metrics}")
        
        # Separate LLM and baseline data (note: our CSV only has LLM models with is_baseline=False)
        if 'is_baseline' in self.df.columns:
            self.llm_data = self.df[self.df['is_baseline'] == False].copy()
            self.baseline_data = self.df[self.df['is_baseline'] == True].copy()
            print(f"🤖 LLM evaluations: {len(self.llm_data)}")
            print(f"📊 Baseline evaluations: {len(self.baseline_data)}")
        else:
            # Assume all data is LLM data if no baseline column
            self.llm_data = self.df.copy()
            self.baseline_data = pd.DataFrame()
            print("⚠️ No baseline data found - assuming all data is LLM")
        
        # Get unique models for analysis
        if 'model' in self.df.columns:
            self.llm_models = self.llm_data['model'].unique()
            self.baseline_models = self.baseline_data['model'].unique() if len(self.baseline_data) > 0 else []
            print(f"🤖 LLM models ({len(self.llm_models)}): {list(self.llm_models)}")
            if len(self.baseline_models) > 0:
                print(f"📊 Baseline models ({len(self.baseline_models)}): {list(self.baseline_models)}")
        else:
            raise Exception("❌ No 'model' column found in data!")
        
        # Validate we have the expected 8 models
        expected_models = [
            "ChatGPT_4o", "GPT_o1", "GPT_o1_Pro", "Claude_3_5_Sonnet", 
            "Gemini_2_0_Flash", "Mistral_Large_2", "Llama_3_1_405B", "DeepSeek_DeepThink_R1"
        ]
        
        print(f"\n🔍 Expected 8 models: {expected_models}")
        print(f"🔍 Found models: {list(self.llm_models)}")
        
        if len(self.llm_models) != 8:
            print(f"⚠️ Warning: Expected 8 models, found {len(self.llm_models)}")
    
    def calculate_cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        if n1 == 0 or n2 == 0:
            return 0
        
        pooled_std = np.sqrt(((n1 - 1) * group1.var() + (n2 - 1) * group2.var()) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0
        
        cohens_d = (group1.mean() - group2.mean()) / pooled_std
        return cohens_d
    
    def calculate_cliffs_delta(self, group1, group2):
        """Calculate Cliff's delta non-parametric effect size."""
        n1, n2 = len(group1), len(group2)
        
        if n1 == 0 or n2 == 0:
            return 0
        
        # Count comparisons
        greater = 0
        less = 0
        
        for x in group1:
            for y in group2:
                if x > y:
                    greater += 1
                elif x < y:
                    less += 1
        
        cliffs_delta = (greater - less) / (n1 * n2)
        return cliffs_delta
    
    def interpret_effect_size(self, cohens_d=None, cliffs_delta=None):
        """Interpret effect size magnitude."""
        if cohens_d is not None:
            abs_d = abs(cohens_d)
            if abs_d < 0.2:
                return "negligible"
            elif abs_d < 0.5:
                return "small"
            elif abs_d < 0.8:
                return "medium"
            else:
                return "large"
        
        if cliffs_delta is not None:
            abs_delta = abs(cliffs_delta)
            if abs_delta < 0.147:
                return "negligible"
            elif abs_delta < 0.33:
                return "small"
            elif abs_delta < 0.474:
                return "medium"
            else:
                return "large"
        
        return "unknown"
    
    def calculate_confidence_interval(self, data, confidence=0.95):
        """Calculate confidence interval for mean."""
        if len(data) == 0:
            return (np.nan, np.nan, np.nan)
        
        mean = np.mean(data)
        sem = stats.sem(data)
        margin_error = sem * t.ppf((1 + confidence) / 2, len(data) - 1)
        
        return mean, mean - margin_error, mean + margin_error
    
    def pairwise_model_comparisons(self, metric='semantic_accuracy'):
        """Perform all pairwise comparisons between LLM models with multiple testing correction."""
        print(f"\n🔬 PAIRWISE LLM MODEL COMPARISONS: {metric.upper()}")
        print("="*70)
        
        if metric not in self.df.columns:
            print(f"❌ Metric '{metric}' not found in data")
            return {}
        
        # Get data for each LLM model
        model_data = {}
        for model in self.llm_models:
            data = self.llm_data[self.llm_data['model'] == model][metric].dropna()
            if len(data) > 0:
                model_data[model] = data
        
        # Generate all pairwise combinations
        model_pairs = list(itertools.combinations(model_data.keys(), 2))
        
        if len(model_pairs) == 0:
            print("❌ No valid model pairs found for comparison")
            return {}
        
        # Store results
        comparison_results = []
        p_values = []
        
        print(f"📊 Performing {len(model_pairs)} pairwise comparisons...")
        
        for model1, model2 in model_pairs:
            data1 = model_data[model1]
            data2 = model_data[model2]
            
            # Mann-Whitney U test (non-parametric)
            try:
                statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
            except ValueError:
                statistic, p_value = np.nan, 1.0
            
            # Effect sizes
            cohens_d = self.calculate_cohens_d(data1, data2)
            cliffs_delta = self.calculate_cliffs_delta(data1, data2)
            
            # Confidence intervals
            mean1, ci1_lower, ci1_upper = self.calculate_confidence_interval(data1)
            mean2, ci2_lower, ci2_upper = self.calculate_confidence_interval(data2)
            
            comparison_results.append({
                'metric': metric,
                'model1': model1,
                'model2': model2,
                'model1_mean': mean1,
                'model1_ci_lower': ci1_lower,
                'model1_ci_upper': ci1_upper,
                'model1_n': len(data1),
                'model2_mean': mean2,
                'model2_ci_lower': ci2_lower,
                'model2_ci_upper': ci2_upper,
                'model2_n': len(data2),
                'mean_difference': mean1 - mean2,
                'mann_whitney_u': statistic,
                'p_value_raw': p_value,
                'cohens_d': cohens_d,
                'cliffs_delta': cliffs_delta,
                'cohens_d_interpretation': self.interpret_effect_size(cohens_d=cohens_d),
                'cliffs_delta_interpretation': self.interpret_effect_size(cliffs_delta=cliffs_delta)
            })
            
            p_values.append(p_value)
        
        # Multiple testing correction (Benjamini-Hochberg FDR)
        if len(p_values) > 0:
            rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                p_values, 
                alpha=0.05, 
                method='fdr_bh'
            )
            
            # Add corrected p-values and significance to results
            for i, result in enumerate(comparison_results):
                result['p_value_corrected'] = p_corrected[i]
                result['significant_raw'] = result['p_value_raw'] < 0.05
                result['significant_corrected'] = rejected[i]
                result['fdr_rejected'] = rejected[i]
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(comparison_results)
        
        # Display summary
        if len(results_df) > 0:
            print(f"📊 Total comparisons: {len(results_df)}")
            print(f"📈 Significant (raw p < 0.05): {sum(results_df['significant_raw'])}")
            print(f"📈 Significant (FDR-corrected): {sum(results_df['significant_corrected'])}")
            print(f"📐 Large effect sizes (Cohen's d): {sum(results_df['cohens_d_interpretation'] == 'large')}")
            print(f"📐 Large effect sizes (Cliff's delta): {sum(results_df['cliffs_delta_interpretation'] == 'large')}")
            
            # Show top significant differences
            significant_results = results_df[results_df['significant_corrected']].copy()
            if len(significant_results) > 0:
                significant_results['abs_effect'] = significant_results['cohens_d'].abs()
                top_significant = significant_results.nlargest(5, 'abs_effect')
                
                print(f"\n🏆 TOP SIGNIFICANT DIFFERENCES (FDR-corrected):")
                for _, row in top_significant.iterrows():
                    print(f"   {row['model1']} vs {row['model2']}: "
                          f"d={row['cohens_d']:.3f} ({row['cohens_d_interpretation']}), "
                          f"p={row['p_value_corrected']:.4f}")
        
        return {
            'results_df': results_df,
            'summary': {
                'total_comparisons': len(results_df),
                'significant_raw': sum(results_df['significant_raw']) if len(results_df) > 0 else 0,
                'significant_corrected': sum(results_df['significant_corrected']) if len(results_df) > 0 else 0,
                'large_cohens_d': sum(results_df['cohens_d_interpretation'] == 'large') if len(results_df) > 0 else 0,
                'large_cliffs_delta': sum(results_df['cliffs_delta_interpretation'] == 'large') if len(results_df) > 0 else 0
            }
        }
    
    def openai_family_analysis(self, metric='semantic_accuracy'):
        """Specific analysis comparing all OpenAI models (ChatGPT-4o, GPT-o1, GPT-o1 Pro)."""
        print(f"\n🤖 OPENAI FAMILY ANALYSIS: {metric.upper()}")
        print("="*70)
        
        # Check if metric exists in the data
        if metric not in self.llm_data.columns:
            print(f"❌ Metric '{metric}' not found in data")
            available_metrics = [m for m in self.key_metrics if m in self.llm_data.columns]
            if available_metrics:
                metric = available_metrics[0]
                print(f"🔄 Using available metric: {metric}")
            else:
                print("❌ No valid metrics available for OpenAI analysis")
                return {}
        
        # Filter for OpenAI models specifically
        openai_models = []
        for model in self.llm_models:
            model_upper = model.upper()
            if any(openai_pattern in model_upper for openai_pattern in ['CHATGPT', 'GPT_O1']):
                openai_models.append(model)
        
        if len(openai_models) < 2:
            print(f"❌ Found only {len(openai_models)} OpenAI model(s): {openai_models}")
            print("Need at least 2 OpenAI models for family analysis")
            return {}
        
        print(f"🔍 Analyzing {len(openai_models)} OpenAI versions: {openai_models}")
        
        # Get OpenAI model data
        openai_data = {}
        for model in openai_models:
            data = self.llm_data[self.llm_data['model'] == model][metric].dropna()
            if len(data) > 0:
                openai_data[model] = data
                print(f"   📊 {model}: {len(data)} evaluations, mean={data.mean():.3f}")
        
        # Perform ANOVA to test for overall differences
        openai_scores = [openai_data[model] for model in openai_data.keys()]
        if len(openai_scores) > 1:
            try:
                f_stat, p_value_anova = stats.f_oneway(*openai_scores)
                print(f"📊 ANOVA F-statistic: {f_stat:.3f}, p-value: {p_value_anova:.4f}")
            except:
                f_stat, p_value_anova = np.nan, 1.0
        
        # Pairwise comparisons within OpenAI family
        openai_pairs = list(itertools.combinations(openai_data.keys(), 2))
        openai_comparison_results = []
        
        for model1, model2 in openai_pairs:
            data1 = openai_data[model1]
            data2 = openai_data[model2]
            
            # Statistical test
            try:
                statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
            except ValueError:
                statistic, p_value = np.nan, 1.0
            
            # Effect sizes
            cohens_d = self.calculate_cohens_d(data1, data2)
            
            # Means and CIs
            mean1, ci1_lower, ci1_upper = self.calculate_confidence_interval(data1)
            mean2, ci2_lower, ci2_upper = self.calculate_confidence_interval(data2)
            
            openai_comparison_results.append({
                'model1': model1,
                'model2': model2,
                'model1_mean': mean1,
                'model1_ci_lower': ci1_lower,
                'model1_ci_upper': ci1_upper,
                'model2_mean': mean2,
                'model2_ci_lower': ci2_lower,
                'model2_ci_upper': ci2_upper,
                'mean_difference': mean1 - mean2,
                'cohens_d': cohens_d,
                'p_value': p_value,
                'effect_size_interpretation': self.interpret_effect_size(cohens_d=cohens_d)
            })
        
        # Multiple testing correction for OpenAI comparisons
        if openai_comparison_results:
            p_values = [result['p_value'] for result in openai_comparison_results]
            rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
            
            for i, result in enumerate(openai_comparison_results):
                result['p_value_corrected'] = p_corrected[i]
                result['significant_corrected'] = rejected[i]
        
        openai_results_df = pd.DataFrame(openai_comparison_results)
        
        # Display OpenAI family ranking
        openai_means = {model: openai_data[model].mean() for model in openai_data.keys()}
        openai_ranking = sorted(openai_means.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 OPENAI FAMILY RANKING ({metric}):")
        for i, (model, mean_score) in enumerate(openai_ranking, 1):
            ci_mean, ci_lower, ci_upper = self.calculate_confidence_interval(openai_data[model])
            print(f"   {i}. {model}: {mean_score:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
        
        return {
            'openai_results_df': openai_results_df,
            'openai_ranking': openai_ranking,
            'anova_f_stat': f_stat if 'f_stat' in locals() else np.nan,
            'anova_p_value': p_value_anova if 'p_value_anova' in locals() else np.nan,
            'openai_models_analyzed': openai_models
        }
    
    def comprehensive_model_ranking(self):
        """Generate comprehensive model ranking with statistical validation."""
        print(f"\n🏆 COMPREHENSIVE LLM MODEL RANKING WITH STATISTICAL VALIDATION")
        print("="*70)
        
        # Calculate mean performance and confidence intervals for each model
        model_performance_data = []
        
        for model in self.llm_models:
            model_data = self.llm_data[self.llm_data['model'] == model]
            
            for metric in self.key_metrics:
                data = model_data[metric].dropna()
                if len(data) > 0:
                    mean, ci_lower, ci_upper = self.calculate_confidence_interval(data)
                    
                    model_performance_data.append({
                        'model': model,
                        'metric': metric,
                        'mean': mean,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'std': data.std(),
                        'n': len(data),
                        'median': data.median(),
                        'min': data.min(),
                        'max': data.max()
                    })
        
        model_performance_df = pd.DataFrame(model_performance_data)
        
        # Create composite ranking based on weighted average of all metrics
        model_rankings = []
        
        for model in self.llm_models:
            model_metrics = model_performance_df[model_performance_df['model'] == model]
            if len(model_metrics) > 0:
                # Calculate weighted average (equal weights for simplicity)
                weighted_score = model_metrics['mean'].mean()
                weighted_ci_lower = model_metrics['ci_lower'].mean()
                weighted_ci_upper = model_metrics['ci_upper'].mean()
                
                # Calculate score on key semantic metrics
                key_semantic_metrics = ['semantic_accuracy', 'reasoning_quality', 'factual_correctness']
                semantic_metrics = model_metrics[model_metrics['metric'].isin(key_semantic_metrics)]
                semantic_score = semantic_metrics['mean'].mean() if len(semantic_metrics) > 0 else weighted_score
                
                model_rankings.append({
                    'model': model,
                    'weighted_score': weighted_score,
                    'weighted_ci_lower': weighted_ci_lower,
                    'weighted_ci_upper': weighted_ci_upper,
                    'semantic_score': semantic_score,
                    'metrics_count': len(model_metrics)
                })
        
        ranking_df = pd.DataFrame(model_rankings)
        ranking_df = ranking_df.sort_values('weighted_score', ascending=False).reset_index(drop=True)
        ranking_df['rank'] = range(1, len(ranking_df) + 1)
        
        print(f"📊 OVERALL LLM MODEL RANKING (weighted average with 95% CI):")
        for _, row in ranking_df.iterrows():
            print(f"   {row['rank']}. {row['model']}: {row['weighted_score']:.3f} "
                  f"(95% CI: {row['weighted_ci_lower']:.3f}-{row['weighted_ci_upper']:.3f})")
        
        return {
            'model_performance_df': model_performance_df,
            'ranking_df': ranking_df
        }
    
    def query_type_analysis(self):
        """Analyze performance differences across query types."""
        print(f"\n📋 QUERY TYPE ANALYSIS")
        print("="*70)
        
        if 'query_id' not in self.llm_data.columns:
            print("❌ No query_id column found for query type analysis")
            return {}
        
        query_types = self.llm_data['query_id'].unique()
        print(f"🔍 Found {len(query_types)} query types: {list(query_types)}")
        
        query_analysis_results = []
        
        for metric in self.key_metrics:
            print(f"\n📊 Analyzing {metric} across query types")
            
            query_data = {}
            for query_type in query_types:
                data = self.llm_data[self.llm_data['query_id'] == query_type][metric].dropna()
                if len(data) > 0:
                    query_data[query_type] = data
            
            # Perform ANOVA across query types
            if len(query_data) > 1:
                query_scores = [query_data[qt] for qt in query_data.keys()]
                try:
                    f_stat, p_value_anova = stats.f_oneway(*query_scores)
                    print(f"   📊 ANOVA F-statistic: {f_stat:.3f}, p-value: {p_value_anova:.4f}")
                except:
                    f_stat, p_value_anova = np.nan, 1.0
                
                # Calculate means for each query type
                for query_type in query_data.keys():
                    data = query_data[query_type]
                    mean, ci_lower, ci_upper = self.calculate_confidence_interval(data)
                    
                    query_analysis_results.append({
                        'metric': metric,
                        'query_type': query_type,
                        'mean': mean,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'std': data.std(),
                        'n': len(data),
                        'anova_f_stat': f_stat,
                        'anova_p_value': p_value_anova
                    })
        
        return pd.DataFrame(query_analysis_results)
    
    def run_complete_analysis(self):
        """Run complete statistical analysis and return all results."""
        print(f"\n📋 RUNNING COMPLETE ENHANCED STATISTICAL ANALYSIS")
        print("="*80)
        print(f"📊 Dataset: {len(self.df)} total evaluations")
        print(f"🤖 LLM Models: {len(self.llm_models)} ({list(self.llm_models)})")
        print(f"📊 Baseline Models: {len(self.baseline_models)} ({list(self.baseline_models)})")
        print(f"📈 Metrics: {len(self.key_metrics)} ({self.key_metrics})")
        
        all_results = {}
        
        # Note: Since our data only has LLM models (no baselines), we skip baseline comparison
        if len(self.baseline_data) == 0:
            print("\n⚠️ No baseline data found - skipping baseline comparisons")
            all_results['baseline_comparisons'] = pd.DataFrame()
        
        # 1. OpenAI family-specific analysis  
        print("\n1️⃣ OPENAI FAMILY ANALYSIS")
        # Use semantic_accuracy if available, otherwise first metric
        primary_metric = 'semantic_accuracy' if 'semantic_accuracy' in self.key_metrics else self.key_metrics[0]
        openai_analysis = self.openai_family_analysis(primary_metric)
        all_results['openai_family_analysis'] = openai_analysis
        
        # 2. Query type analysis
        print("\n2️⃣ QUERY TYPE ANALYSIS")
        query_analysis = self.query_type_analysis()
        all_results['query_analysis'] = query_analysis
        
        # 3. Pairwise model comparisons for all metrics
        print("\n3️⃣ PAIRWISE MODEL COMPARISONS")
        pairwise_results = []
        
        for metric in self.key_metrics:
            if metric in self.llm_data.columns:
                print(f"\n🔍 Analyzing {metric}")
                comparison_result = self.pairwise_model_comparisons(metric)
                if 'results_df' in comparison_result and len(comparison_result['results_df']) > 0:
                    pairwise_results.append(comparison_result['results_df'])
        
        if pairwise_results:
            all_pairwise_df = pd.concat(pairwise_results, ignore_index=True)
            all_results['pairwise_comparisons'] = all_pairwise_df
        else:
            all_results['pairwise_comparisons'] = pd.DataFrame()
        
        # 4. Comprehensive model ranking
        print("\n4️⃣ MODEL RANKINGS")
        ranking_results = self.comprehensive_model_ranking()
        all_results['model_performance'] = ranking_results['model_performance_df']
        all_results['model_ranking'] = ranking_results['ranking_df']
        
        return all_results
    
    def save_results_to_csv(self, results, output_dir='RESULTS/01-ADVANCED_ANALYSIS'):
        """Save all statistical analysis results to CSV files in RESULTS/01-ADVANCED_ANALYSIS."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 SAVING RESULTS TO {output_dir}")
        print("-" * 50)
        
        saved_files = []
        
        # Save OpenAI family analysis
        if 'openai_family_analysis' in results and 'openai_results_df' in results['openai_family_analysis']:
            if len(results['openai_family_analysis']['openai_results_df']) > 0:
                openai_file = f'{output_dir}/openai_family_analysis.csv'
                results['openai_family_analysis']['openai_results_df'].to_csv(openai_file, index=False)
                print(f"🤖 OpenAI family analysis: {openai_file}")
                saved_files.append(openai_file)
        
        # Save query type analysis
        if 'query_analysis' in results and len(results['query_analysis']) > 0:
            query_file = f'{output_dir}/query_type_analysis.csv'
            results['query_analysis'].to_csv(query_file, index=False)
            print(f"📋 Query type analysis: {query_file}")
            saved_files.append(query_file)
        
        # Save pairwise comparisons
        if 'pairwise_comparisons' in results and len(results['pairwise_comparisons']) > 0:
            pairwise_file = f'{output_dir}/pairwise_comparisons_with_statistics.csv'
            results['pairwise_comparisons'].to_csv(pairwise_file, index=False)
            print(f"🔍 Pairwise comparisons: {pairwise_file}")
            saved_files.append(pairwise_file)
        
        # Save model performance
        if 'model_performance' in results and len(results['model_performance']) > 0:
            performance_file = f'{output_dir}/model_performance_with_ci.csv'
            results['model_performance'].to_csv(performance_file, index=False)
            print(f"📈 Model performance: {performance_file}")
            saved_files.append(performance_file)
        
        # Save model ranking
        if 'model_ranking' in results and len(results['model_ranking']) > 0:
            ranking_file = f'{output_dir}/model_ranking_with_statistics.csv'
            results['model_ranking'].to_csv(ranking_file, index=False)
            print(f"🏆 Model ranking: {ranking_file}")
            saved_files.append(ranking_file)
        
        # Create summary statistics file
        summary_stats = self._create_summary_statistics(results)
        summary_file = f'{output_dir}/statistical_summary.csv'
        summary_stats.to_csv(summary_file, index=False)
        print(f"📋 Summary statistics: {summary_file}")
        saved_files.append(summary_file)
        
        # Save analysis metadata
        metadata = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'total_evaluations': len(self.df),
            'llm_models': list(self.llm_models),
            'baseline_models': list(self.baseline_models),
            'metrics_analyzed': self.key_metrics,
            'files_generated': saved_files
        }
        
        metadata_file = f'{output_dir}/analysis_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"📄 Analysis metadata: {metadata_file}")
        
        print(f"\n✅ All statistical analysis results saved to {output_dir}")
        return output_dir
    
    def _create_summary_statistics(self, results):
        """Create summary statistics DataFrame."""
        summary_data = []
        
        # Overall statistics
        summary_data.append({
            'statistic': 'total_llm_models_evaluated',
            'value': len(self.llm_models),
            'description': 'Number of LLM models evaluated'
        })
        
        summary_data.append({
            'statistic': 'total_baseline_models',
            'value': len(self.baseline_models),
            'description': 'Number of baseline models'
        })
        
        openai_models = [model for model in self.llm_models if 'GPT' in model.upper() or 'CHATGPT' in model.upper()]
        summary_data.append({
            'statistic': 'openai_models_evaluated',
            'value': len(openai_models),
            'description': 'Number of OpenAI models evaluated'
        })
        
        summary_data.append({
            'statistic': 'total_llm_evaluations',
            'value': len(self.llm_data),
            'description': 'Total number of LLM evaluations'
        })
        
        summary_data.append({
            'statistic': 'total_baseline_evaluations',
            'value': len(self.baseline_data),
            'description': 'Total number of baseline evaluations'
        })
        
        summary_data.append({
            'statistic': 'metrics_evaluated',
            'value': len(self.key_metrics),
            'description': 'Number of performance metrics evaluated'
        })
        
        # Pairwise comparison statistics
        if 'pairwise_comparisons' in results and len(results['pairwise_comparisons']) > 0:
            pairwise_df = results['pairwise_comparisons']
            
            summary_data.append({
                'statistic': 'total_pairwise_comparisons',
                'value': len(pairwise_df),
                'description': 'Total number of pairwise model comparisons'
            })
            
            summary_data.append({
                'statistic': 'significant_pairwise_comparisons',
                'value': sum(pairwise_df['significant_corrected']),
                'description': 'Pairwise comparisons with significant differences (FDR-corrected)'
            })
        
        return pd.DataFrame(summary_data)

def main():
    """Execute enhanced statistical analysis adapted for LLM evaluation results."""
    
    print("🔬 Enhanced Statistical Analysis for LLM Evaluation Results")
    print("="*70)
    print("📥 Looking for input: actual_llm_data.csv")
    print("📁 All outputs will be saved to: RESULTS/01-ADVANCED_ANALYSIS")
    print("✅ Includes multiple testing correction and effect sizes")
    print("✅ OpenAI family analysis and comprehensive model comparisons")
    print("✅ Statistical validation with confidence intervals")
    print("🤖 Expects 8 LLM models: ChatGPT-4o, GPT-o1, GPT-o1 Pro, Claude 3.5 Sonnet,")
    print("                         Gemini 2.0 Flash, Mistral Large 2, Llama 3.1 405B, DeepSeek DeepThink R1")
    print()
    
    try:
        # Initialize analyzer (will auto-detect the CSV file)
        analyzer = EnhancedStatisticalAnalyzer()
        
        # Run complete analysis
        results = analyzer.run_complete_analysis()
        
        # Save results to CSV
        output_dir = analyzer.save_results_to_csv(results, 'RESULTS/01-ADVANCED_ANALYSIS')
        
        print("\n✅ ENHANCED STATISTICAL ANALYSIS COMPLETED!")
        print("\n🎯 KEY FINDINGS FOR PUBLICATION:")
        print("-" * 50)
        
        # Display key findings for paper
        if 'pairwise_comparisons' in results and len(results['pairwise_comparisons']) > 0:
            pairwise_df = results['pairwise_comparisons']
            significant_pairwise = sum(pairwise_df['significant_corrected'])
            
            print(f"🔍 Significant pairwise differences (FDR-corrected): {significant_pairwise}/{len(pairwise_df)}")
        
        if 'model_ranking' in results and len(results['model_ranking']) > 0:
            best_model = results['model_ranking'].iloc[0]
            print(f"🏆 Best performing LLM: {best_model['model']} (score: {best_model['weighted_score']:.3f})")
        
        # OpenAI-specific findings
        if 'openai_family_analysis' in results and 'openai_ranking' in results['openai_family_analysis']:
            openai_ranking = results['openai_family_analysis']['openai_ranking']
            if openai_ranking:
                best_openai = openai_ranking[0]
                print(f"🤖 Best OpenAI model: {best_openai[0]} (score: {best_openai[1]:.3f})")
        
        print(f"\n📁 All publication-ready statistics saved to: {output_dir}")
        print("   These results include proper multiple testing correction,")
        print("   effect sizes, confidence intervals, and comprehensive analysis.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()