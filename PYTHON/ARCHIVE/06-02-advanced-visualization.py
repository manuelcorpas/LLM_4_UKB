#!/usr/bin/env python3
"""
Professional LLM Visualization Suite - Clean Publication-Ready Figures
Generates publication-ready visualizations matching the provided examples
Input: RESULTS/01-ADVANCED_ANALYSIS/
Output: RESULTS/02-ADVANCED-VIZ/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

class ProfessionalLLMVisualizer:
    """Generate clean, professional visualizations matching the example style."""
    
    def __init__(self, input_dir='RESULTS/01-ADVANCED_ANALYSIS', output_dir='RESULTS/02-ADVANCED-VIZ'):
        """Initialize with clean styling."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set professional styling
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
        
        self.load_data()
        self.setup_colors()
        
    def load_data(self):
        """Load statistical analysis data."""
        # Load model performance data
        perf_file = f'{self.input_dir}/model_performance_with_ci.csv'
        self.model_performance_df = pd.read_csv(perf_file)
        
        # Load model rankings
        ranking_file = f'{self.input_dir}/model_ranking_with_statistics.csv'
        self.model_ranking_df = pd.read_csv(ranking_file)
        
        # Load pairwise comparisons
        pairwise_file = f'{self.input_dir}/pairwise_comparisons_with_statistics.csv'
        self.pairwise_df = pd.read_csv(pairwise_file)
        
        # Extract models and metrics
        self.models = sorted(self.model_performance_df['model'].unique())
        self.key_metrics = sorted(self.model_performance_df['metric'].unique())
        
        # Create clean model names
        self.model_names = {
            "ChatGPT_4o": "GPT-4o",
            "GPT_o1_Pro": "GPT-o1-Pro", 
            "GPT_o1": "GPT-o1",
            "Claude_3_5_Sonnet": "Claude-3.5",
            "DeepSeek_DeepThink_R1": "DeepThink-R1",
            "Llama_3_1_405B": "Llama-3.1",
            "Mistral_Large_2": "Mistral-L2",
            "Gemini_2_0_Flash": "Gemini-2.0"
        }
        
        # Create clean metric names
        self.metric_names = {
            'semantic_accuracy': 'Semantic Accuracy',
            'reasoning_quality': 'Reasoning Quality', 
            'domain_knowledge_score': 'Domain Knowledge',
            'factual_correctness': 'Factual Correctness',
            'depth_score': 'Response Depth',
            'biobank_specificity': 'Biobank Specificity'
        }
        
        print(f"✅ Loaded data: {len(self.models)} models, {len(self.key_metrics)} metrics")
        
    def setup_colors(self):
        """Setup professional color schemes."""
        # Model performance colors (green=good, red=bad)
        self.performance_colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', 
                                 '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
        
        # Metric colors
        self.metric_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
        
        # Ranking colors (gradient)
        self.rank_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 8))
        
    def get_clean_name(self, name, name_type='model'):
        """Get clean display name."""
        if name_type == 'model':
            return self.model_names.get(name, name)
        elif name_type == 'metric':
            return self.metric_names.get(name, name.replace('_', ' ').title())
        return name
    
    def create_performance_summary_table(self, save_path=None):
        """Create Figure 1: Performance summary table."""
        if save_path is None:
            save_path = f'{self.output_dir}/01_performance_summary_table.png'
        
        # Calculate summary statistics
        summary_data = []
        for model in self.models:
            model_data = self.model_performance_df[self.model_performance_df['model'] == model]['mean']
            summary_data.append({
                'Model': self.get_clean_name(model),
                'Mean': model_data.mean(),
                'Std_Dev': model_data.std(),
                'Min': model_data.min(),
                'Max': model_data.max(),
                'N_Tests': len(model_data)
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Mean', ascending=False).reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table_data = []
        colors = []
        
        for i, row in df.iterrows():
            table_data.append([
                row['Rank'],
                f"{row['Mean']:.3f}",
                f"{row['Std_Dev']:.3f}",
                f"{row['Min']:.3f}",
                f"{row['Max']:.3f}",
                row['N_Tests']
            ])
            
            # Color coding: green for top performers, red for bottom
            if row['Rank'] <= 3:
                colors.append(['#d9f2d9'] * 6)  # Light green
            elif row['Rank'] >= 7:
                colors.append(['#f2d9d9'] * 6)  # Light red  
            else:
                colors.append(['#f9f9f9'] * 6)  # Light gray
        
        table = ax.table(
            cellText=table_data,
            colLabels=['Rank', 'Mean', 'Std Dev', 'Min', 'Max', 'N Tests'],
            rowLabels=[row['Model'] for _, row in df.iterrows()],
            cellLoc='center',
            loc='center',
            cellColours=colors
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        
        # Style table
        for i in range(len(df)):
            table[(i+1, -1)].set_facecolor('#e6e6e6')  # Row labels
        for j in range(6):
            table[(0, j)].set_facecolor('#cccccc')  # Column headers
            
        plt.title('Model Performance Summary\n(Sorted by Overall Rank)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Performance summary table saved to: {save_path}")
    
    def create_four_panel_analysis(self, save_path=None):
        """Create Figure 2: Four-panel analysis."""
        if save_path is None:
            save_path = f'{self.output_dir}/02_four_panel_analysis.png'
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Model Analysis', fontsize=16, fontweight='bold')
        
        # Panel 1: Query Champions (Best model per task/metric)
        metric_champions = {}
        for metric in self.key_metrics:
            metric_data = self.model_performance_df[self.model_performance_df['metric'] == metric]
            best_model = metric_data.loc[metric_data['mean'].idxmax()]
            metric_champions[metric] = {
                'model': self.get_clean_name(best_model['model']),
                'score': best_model['mean']
            }
        
        metrics = list(metric_champions.keys())
        scores = [metric_champions[m]['score'] for m in metrics]
        champions = [metric_champions[m]['model'] for m in metrics]
        
        bars1 = ax1.bar(range(len(metrics)), scores, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0'])
        ax1.set_xticks(range(len(metrics)))
        ax1.set_xticklabels([self.get_clean_name(m, 'metric') for m in metrics], rotation=45, ha='right')
        ax1.set_ylabel('Best Performance Score')
        ax1.set_title('Query Champions\n(Best Model per Task)')
        
        # Add champion labels on bars
        for i, (bar, champion, score) in enumerate(zip(bars1, champions, scores)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.05,
                    champion, ha='center', va='top', fontweight='bold', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Panel 2: Task Difficulty Ranking
        avg_scores = []
        for metric in self.key_metrics:
            metric_data = self.model_performance_df[self.model_performance_df['metric'] == metric]
            avg_scores.append(metric_data['mean'].mean())
        
        difficulty_df = pd.DataFrame({
            'metric': metrics,
            'avg_score': avg_scores
        }).sort_values('avg_score')
        
        bars2 = ax2.bar(range(len(difficulty_df)), difficulty_df['avg_score'], 
                       color=plt.cm.Reds(np.linspace(0.4, 0.8, len(difficulty_df))))
        ax2.set_xticks(range(len(difficulty_df)))
        ax2.set_xticklabels([self.get_clean_name(m, 'metric') for m in difficulty_df['metric']], 
                           rotation=45, ha='right')
        ax2.set_ylabel('Average Performance (All Models)')
        ax2.set_title('Task Difficulty Ranking\n(Lower Average = Harder Task)')
        
        # Add score labels
        for bar, score in zip(bars2, difficulty_df['avg_score']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Panel 3: Complete Performance Matrix
        pivot_data = self.model_performance_df.pivot_table(
            values='mean', index='metric', columns='model', aggfunc='first'
        )
        
        # Reorder columns and rows
        model_order = [m for m in self.models if m in pivot_data.columns]
        metric_order = self.key_metrics
        pivot_data = pivot_data.reindex(index=metric_order, columns=model_order)
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax3,
                   xticklabels=[self.get_clean_name(m) for m in model_order],
                   yticklabels=[self.get_clean_name(m, 'metric') for m in metric_order],
                   cbar_kws={'label': 'Performance Score'})
        ax3.set_title('Complete Performance Matrix\n(All Models × All Tasks)')
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Tasks')
        
        # Panel 4: Model Performance Range
        model_stats = []
        for model in self.models:
            model_data = self.model_performance_df[self.model_performance_df['model'] == model]
            model_stats.append({
                'model': self.get_clean_name(model),
                'mean': model_data['mean'].mean(),
                'min': model_data['mean'].min(),
                'max': model_data['mean'].max()
            })
        
        stats_df = pd.DataFrame(model_stats).sort_values('mean', ascending=True)
        
        y_pos = range(len(stats_df))
        ax4.errorbar(stats_df['mean'], y_pos, 
                    xerr=[stats_df['mean'] - stats_df['min'], stats_df['max'] - stats_df['mean']],
                    fmt='o', capsize=5, capthick=2, markersize=8, color='steelblue')
        
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(stats_df['model'])
        ax4.set_xlabel('Performance Score')
        ax4.set_title('Model Performance Range\n(Mean ± Min/Max Range)')
        
        # Add mean values
        for i, (y, mean) in enumerate(zip(y_pos, stats_df['mean'])):
            ax4.text(mean + 0.02, y, f'{mean:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Four-panel analysis saved to: {save_path}")
    
    def create_overall_performance(self, save_path=None):
        """Create Figure 3: Overall performance and consistency."""
        if save_path is None:
            save_path = f'{self.output_dir}/03_overall_performance.png'
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('LLM Performance Analysis', fontsize=16, fontweight='bold')
        
        # Panel 1: Overall Performance
        ranking_data = self.model_ranking_df.sort_values('weighted_score', ascending=False)
        
        bars1 = ax1.bar(range(len(ranking_data)), ranking_data['weighted_score'],
                       color=self.rank_colors, edgecolor='black', linewidth=0.5)
        
        ax1.set_xticks(range(len(ranking_data)))
        ax1.set_xticklabels([self.get_clean_name(m) for m in ranking_data['model']], 
                           rotation=45, ha='right')
        ax1.set_ylabel('Semantic Accuracy')
        ax1.set_title('All LLM Overall Performance\n(Sorted Best to Worst)')
        
        # Add score labels
        for bar, score in zip(bars1, ranking_data['weighted_score']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Panel 2: Consistency Ranking  
        consistency_data = []
        for model in self.models:
            model_data = self.model_performance_df[self.model_performance_df['model'] == model]
            cv = (model_data['std'].mean() / model_data['mean'].mean()) * 100
            consistency_data.append({
                'model': self.get_clean_name(model),
                'consistency': 100 - cv  # Higher = more consistent
            })
        
        consist_df = pd.DataFrame(consistency_data).sort_values('consistency', ascending=False)
        
        bars2 = ax2.bar(range(len(consist_df)), consist_df['consistency'],
                       color=plt.cm.Blues(np.linspace(0.3, 0.9, len(consist_df))))
        
        ax2.set_xticks(range(len(consist_df)))
        ax2.set_xticklabels(consist_df['model'], rotation=45, ha='right')
        ax2.set_ylabel('Consistency Score')
        ax2.set_title('All LLM Consistency Ranking\n(Higher = More Consistent)')
        
        # Add consistency scores
        for bar, score in zip(bars2, consist_df['consistency']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{score:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Overall performance saved to: {save_path}")
    
    def create_ranking_heatmap(self, save_path=None):
        """Create Figure 4: Model rankings across metrics heatmap."""
        if save_path is None:
            save_path = f'{self.output_dir}/04_ranking_heatmap.png'
            
        # Create ranking matrix
        ranking_matrix = []
        for metric in self.key_metrics:
            metric_data = self.model_performance_df[self.model_performance_df['metric'] == metric]
            metric_data = metric_data.sort_values('mean', ascending=False)
            
            ranking = {}
            for rank, (_, row) in enumerate(metric_data.iterrows(), 1):
                ranking[row['model']] = rank
            
            ranking_matrix.append([ranking.get(model, 8) for model in self.models])
        
        ranking_df = pd.DataFrame(ranking_matrix, 
                                 index=[self.get_clean_name(m, 'metric') for m in self.key_metrics],
                                 columns=[self.get_clean_name(m) for m in self.models])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(ranking_df, annot=True, fmt='d', cmap='RdYlGn_r', ax=ax,
                   cbar_kws={'label': 'Rank (1=Best)'}, vmin=1, vmax=8)
        
        ax.set_title('Model Rankings Across Metrics\n(1=Best Performance)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Models')
        ax.set_ylabel('Evaluation Metrics')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Ranking heatmap saved to: {save_path}")
    
    def create_individual_comparison(self, save_path=None):
        """Create Figure 5: Individual model performance comparison."""
        if save_path is None:
            save_path = f'{self.output_dir}/05_individual_comparison.png'
            
        # Select key metrics for comparison
        key_metrics = ['semantic_accuracy', 'reasoning_quality', 'domain_knowledge_score']
        
        # Prepare data
        comparison_data = []
        for model in self.models:
            model_data = self.model_performance_df[self.model_performance_df['model'] == model]
            model_metrics = {}
            for metric in key_metrics:
                metric_row = model_data[model_data['metric'] == metric]
                if not metric_row.empty:
                    model_metrics[metric] = metric_row.iloc[0]['mean']
                else:
                    model_metrics[metric] = 0
            comparison_data.append({
                'model': self.get_clean_name(model),
                **model_metrics
            })
        
        comp_df = pd.DataFrame(comparison_data)
        comp_df = comp_df.sort_values('semantic_accuracy', ascending=True)
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(14, 10))
        
        y_pos = np.arange(len(comp_df))
        width = 0.25
        
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        metric_labels = ['Semantic Accuracy', 'Reasoning Quality', 'Domain Knowledge']
        
        for i, (metric, color, label) in enumerate(zip(key_metrics, colors, metric_labels)):
            bars = ax.barh(y_pos + i*width, comp_df[metric], width, 
                          color=color, alpha=0.8, label=label)
            
            # Add value labels
            for bar, value in zip(bars, comp_df[metric]):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{value:.2f}', va='center', fontweight='bold', fontsize=9)
        
        ax.set_yticks(y_pos + width)
        ax.set_yticklabels(comp_df['model'])
        ax.set_xlabel('Performance Score')
        ax.set_title('Individual Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Individual comparison saved to: {save_path}")
    
    def create_capability_radar(self, save_path=None):
        """Create Figure 6: LLM capability profiles radar chart."""
        if save_path is None:
            save_path = f'{self.output_dir}/06_capability_radar.png'
            
        # Prepare radar chart data
        metrics = self.key_metrics
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        # Color scheme
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.models)))
        
        for i, model in enumerate(self.models):
            model_data = self.model_performance_df[self.model_performance_df['model'] == model]
            
            values = []
            for metric in metrics:
                metric_row = model_data[model_data['metric'] == metric]
                if not metric_row.empty:
                    values.append(metric_row.iloc[0]['mean'])
                else:
                    values.append(0)
            
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=self.get_clean_name(model), color=colors[i], markersize=4)
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # Customize radar chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([self.get_clean_name(m, 'metric') for m in metrics], fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('LLM Capability Profiles\n(All Models)', fontsize=16, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Capability radar saved to: {save_path}")
    
    def generate_all_figures(self):
        """Generate all professional figures."""
        print(f"\n🎨 GENERATING PROFESSIONAL VISUALIZATION SUITE")
        print("="*70)
        print(f"📥 Input: {self.input_dir}")
        print(f"📁 Output: {self.output_dir}")
        print(f"🤖 Models: {len(self.models)}")
        print(f"📈 Metrics: {len(self.key_metrics)}")
        print("-" * 70)
        
        try:
            self.create_performance_summary_table()
            print("✅ Figure 1: Performance summary table")
        except Exception as e:
            print(f"❌ Error in Figure 1: {e}")
        
        try:
            self.create_four_panel_analysis()
            print("✅ Figure 2: Four-panel analysis")
        except Exception as e:
            print(f"❌ Error in Figure 2: {e}")
            
        try:
            self.create_overall_performance()
            print("✅ Figure 3: Overall performance")
        except Exception as e:
            print(f"❌ Error in Figure 3: {e}")
            
        try:
            self.create_ranking_heatmap()
            print("✅ Figure 4: Ranking heatmap")
        except Exception as e:
            print(f"❌ Error in Figure 4: {e}")
            
        try:
            self.create_individual_comparison()
            print("✅ Figure 5: Individual comparison")
        except Exception as e:
            print(f"❌ Error in Figure 5: {e}")
            
        try:
            self.create_capability_radar()
            print("✅ Figure 6: Capability radar")
        except Exception as e:
            print(f"❌ Error in Figure 6: {e}")
        
        print(f"\n✅ PROFESSIONAL VISUALIZATION SUITE COMPLETED!")
        print(f"📁 All files saved to: {self.output_dir}")
        print("   • 01_performance_summary_table.png")
        print("   • 02_four_panel_analysis.png") 
        print("   • 03_overall_performance.png")
        print("   • 04_ranking_heatmap.png")
        print("   • 05_individual_comparison.png")
        print("   • 06_capability_radar.png")

def main():
    """Generate professional visualizations matching the example style."""
    
    print("🎨 Professional LLM Visualization Suite")
    print("="*70)
    print("📥 Input: RESULTS/01-ADVANCED_ANALYSIS/")
    print("📁 Output: RESULTS/02-ADVANCED-VIZ/")
    print("✅ Generates clean, publication-ready visualizations")
    print("✅ Matches the style of provided examples")
    print("✅ Professional styling and layouts")
    print()
    
    # Check required files
    input_dir = 'RESULTS/01-ADVANCED_ANALYSIS'
    required_files = [
        'model_performance_with_ci.csv',
        'model_ranking_with_statistics.csv', 
        'pairwise_comparisons_with_statistics.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(f'{input_dir}/{file}'):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files:")
        for file in missing_files:
            print(f"   • {file}")
        print(f"\n💡 Please run the statistical analysis script first!")
        return
    
    try:
        # Generate professional visualizations
        visualizer = ProfessionalLLMVisualizer(input_dir, 'RESULTS/02-ADVANCED-VIZ')
        visualizer.generate_all_figures()
        
        print("\n🎯 PROFESSIONAL RESULTS GENERATED:")
        print("-" * 50)
        print("✅ Clean, publication-ready figures")
        print("✅ Professional styling and color schemes")
        print("✅ Matches provided example layouts")
        print("✅ Ready for publication or presentation")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()