import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv

class UKBPublicationAnalyzer:
    def __init__(self, schema_dir='/Users/manuelcorpas1/Library/Mobile Documents/com~apple~CloudDocs/UKBiobank/RAG-UKBAnalyzer/ukb_schemas'):
        self.schema_dir = schema_dir
        self.publications_df = None

    def parse_publications_schema(self):
        """Parse schema 19 (publications) file with improved handling of multiline fields and numeric conversion."""
        try:
            file_path = os.path.join(self.schema_dir, 'schema_19.txt')
            if not os.path.exists(file_path):
                print("⚠️ Publications schema file not found.")
                return None

            # Read the file while handling multiline fields properly
            self.publications_df = pd.read_csv(
                file_path,
                sep='\t',
                quoting=csv.QUOTE_NONE,  # Prevents issues with quotes inside text fields
                dtype=str,  # Read all columns as strings first
                on_bad_lines='skip',  # Skip problematic lines
                engine="python"
            )

            # Ensure cite_total is numeric
            if 'cite_total' in self.publications_df.columns:
                self.publications_df['cite_total'] = (
                    self.publications_df['cite_total']
                    .str.strip()  # Remove any leading/trailing spaces
                    .replace('', '0')  # Replace empty strings with zero
                    .astype(float)  # Convert to numeric (float)
                )

            return self.publications_df
        except Exception as e:
            print(f"⚠️ Error parsing publications schema: {e}")
            return None

    def analyze_keywords(self):
        """Analyze and visualize keyword statistics."""
        if self.publications_df is None or 'keywords' not in self.publications_df.columns:
            print("⚠️ No 'keywords' column in the dataset.")
            return

        keywords_series = self.publications_df['keywords'].dropna().str.split('|').explode()
        top_keywords = keywords_series.value_counts().head(20)

        print("\n📌 Top Keywords:")
        print(top_keywords)

        plt.figure(figsize=(10, 6))
        top_keywords.sort_values().plot(kind='barh', color='skyblue')
        plt.title("Top 20 Keywords in Publications")
        plt.xlabel("Count")
        plt.ylabel("Keywords")
        plt.tight_layout()
        plt.savefig("top_keywords.png")
        plt.show()

    def analyze_authors(self):
        """Analyze and visualize author statistics."""
        if self.publications_df is None or 'authors' not in self.publications_df.columns:
            print("⚠️ No 'authors' column in the dataset.")
            return

        authors_series = self.publications_df['authors'].dropna().str.split('|').explode()
        top_authors = authors_series.value_counts().head(20)

        print("\n📚 Top Authors:")
        print(top_authors)

        plt.figure(figsize=(10, 6))
        top_authors.sort_values().plot(kind='barh', color='lightgreen')
        plt.title("Top 20 Authors by Number of Publications")
        plt.xlabel("Count")
        plt.ylabel("Authors")
        plt.tight_layout()
        plt.savefig("top_authors.png")
        plt.show()

    def visualize_year_pub(self):
        """Visualize the number of publications per year."""
        if self.publications_df is None or 'year_pub' not in self.publications_df.columns:
            print("⚠️ No 'year_pub' column in the dataset.")
            return

        year_counts = self.publications_df['year_pub'].value_counts().sort_index()

        print("\n📅 Publications by Year:")
        print(year_counts)

        plt.figure(figsize=(10, 6))
        year_counts.plot(kind='bar', color='coral')
        plt.title("Number of Publications by Year")
        plt.xlabel("Year")
        plt.ylabel("Number of Publications")
        plt.tight_layout()
        plt.savefig("publications_by_year.png")
        plt.show()

    def visualize_most_cited_articles(self):
        """Visualize the most cited articles with titles and journals."""
        required_columns = {'cite_total', 'title', 'journal'}
        if self.publications_df is None or not required_columns.issubset(self.publications_df.columns):
            print("⚠️ Required columns ('cite_total', 'title', 'journal') are missing in the dataset.")
            return

        # Ensure cite_total is numeric and drop NaNs
        most_cited = (
            self.publications_df
            .dropna(subset=['cite_total'])  # Remove missing values in cite_total
            .nlargest(10, 'cite_total')  # Get top 10 most cited
            [['pub_id', 'title', 'journal', 'cite_total']]
        )

        # Truncate long titles
        most_cited['short_title'] = most_cited['title'].apply(lambda x: x[:60] + '...' if len(x) > 60 else x)
        most_cited['title_with_journal'] = most_cited['short_title'] + "\n(" + most_cited['journal'] + ")"

        print("\n🏆 Most Cited Articles:")
        print(most_cited[['pub_id', 'title_with_journal', 'cite_total']])

        plt.figure(figsize=(12, 8))
        sns.barplot(data=most_cited, x='cite_total', y='title_with_journal', palette='viridis')
        plt.title("Top 10 Most Cited Articles")
        plt.xlabel("Total Citations")
        plt.ylabel("Article Title\n(with Journal)")
        plt.tight_layout()
        plt.savefig("most_cited_articles.png")
        plt.show()

    def generate_report(self):
        """Generate a comprehensive report."""
        self.parse_publications_schema()

        if self.publications_df is None:
            print("⚠️ No valid data found in schema_19.txt. Exiting.")
            return

        print("\n🔎 Analyzing Keywords...")
        self.analyze_keywords()

        print("\n🔎 Analyzing Authors...")
        self.analyze_authors()

        print("\n📊 Visualizing Publications by Year...")
        self.visualize_year_pub()

        print("\n📊 Visualizing Most Cited Articles...")
        self.visualize_most_cited_articles()

        print("\n✅ Report Generation Complete!")

def main():
    analyzer = UKBPublicationAnalyzer()
    analyzer.generate_report()

if __name__ == "__main__":
    main()

