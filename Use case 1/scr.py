# Databricks notebook source
# COMMAND ----------

import os
import pandas as pd
from collections import Counter
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Scikit-Learn Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer

# Plotly for Visualizations
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# LangChain for LLMs
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Stop Words for Text Processing
from sklearn.feature_extraction import text


# COMMAND ----------

class ConversationAnalyser:
    def __init__(self, folder_path, test_data, llm_params):
        # Initialize paths, test data, and the language model
        self.folder_path = folder_path
        self.test_df = pd.DataFrame(test_data)
        self.transcripts_df = self.load_and_prepare_transcripts()
        
        # Set up the Azure Language Model
        os.environ.update(llm_params)
        self.llm = AzureChatOpenAI(
            deployment_name=llm_params['deployment_name'],
            model_name=llm_params['model_name'],
            temperature=0,
            max_tokens=4000
        )

        # Set up the language model chains
        self.sentiment_chain = self._initialize_chain(
            """
            Analyse the sentiment of the following conversation, considering only the "Member" (customer) statements.
            Determine if the overall sentiment is "positive," "neutral," or "negative" based on the customer's tone.
            Respond with only one word: "positive," "neutral," or "negative"—nothing else.

            Member Statements:
            {member_text}

            Sentiment:
            """
        )
        self.outcome_chain = self._initialize_chain(
        """
        Determine the outcome of the following conversation based on only the "Member" (customer) statements.
        Identify if the call resulted in "issue resolved" or "follow-up action needed."
        Only respond with one of these two outcomes—nothing else.

        Member Statements:
        {member_text}

        Outcome:
        """
        )
    
    def load_and_prepare_transcripts(self):
        # Load and clean transcripts, convert 'file_number' to numeric, and join with test data
        file_data = [
            {"file_name": f, "text": open(os.path.join(self.folder_path, f), 'r', encoding='utf-8').read()}
            for f in os.listdir(self.folder_path) if f.endswith(".txt")
        ]
        transcripts = pd.DataFrame(file_data)
        transcripts['file_number'] = transcripts['file_name'].str.extract('(\d+)').astype(int)
        
        # Merge with test data on 'file_number'
        transcripts_df = pd.merge(transcripts, self.test_df, on='file_number', how='left')
        
        # Add agent-specific flags
        transcripts_df["member_text"] = transcripts_df["text"].apply(lambda x: self.extract_text_by_prefix(x, "Member:"))
        for agent_type in ["customer_support", "technical_support", "pa_agent", "agent"]:
            if agent_type == "pa_agent":
                prefix = f"{agent_type.split('_')[0].upper()} {agent_type.split('_')[1].title()}:"
            else:
                prefix = f"{agent_type.replace('_', ' ').title()}:"
            transcripts_df[f"{agent_type}_text"] = transcripts_df["text"].apply(lambda x: self.extract_text_by_prefix(x, prefix))
            transcripts_df[f"{agent_type}_flag"] = transcripts_df[f"{agent_type}_text"].notnull().astype(int)

        return transcripts_df

    def get_transcripts_df(self):
        """Returns the transcripts DataFrame after processing."""
        return self.transcripts_df

    @staticmethod
    def extract_text_by_prefix(conversation_text, prefix):
        """Extract lines starting with a specific prefix from conversation text."""
        lines = [line for line in conversation_text.splitlines() if line.startswith(prefix)]
        return "\n".join(lines) if lines else None

    def _initialize_chain(self, prompt_text):
        """Set up an LLMChain with the given prompt."""
        prompt = PromptTemplate(input_variables=["member_text"], template=prompt_text)
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def analyse_conversation(self, member_text):
        """Apply sentiment and outcome analysis to a conversation."""
        sentiment_result = self.sentiment_chain.run(member_text=member_text).strip().lower()
        outcome_result = self.outcome_chain.run(member_text=member_text).strip().lower()
        return sentiment_result, outcome_result

    def process_transcripts(self):
        """Run analysis on all transcripts in the dataframe."""
        self.transcripts_df[['call_sentiment', 'call_outcome']] = self.transcripts_df['member_text'].apply(
            lambda text: pd.Series(self.analyse_conversation(text))
        )
    
    def calculate_metrics(self):
        """Calculate and return accuracy, precision, recall, and F1 for sentiment and outcome prediction."""
        df = self.transcripts_df.dropna(subset=['test_sentiment', 'test_outcome'])
        metrics = {}
        for metric_type in ['sentiment', 'outcome']:
            true_values = df[f'test_{metric_type}']
            pred_values = df[f'call_{metric_type}']
            metrics[f'{metric_type}_accuracy'] = accuracy_score(true_values, pred_values)
            metrics[f'{metric_type}_precision'] = precision_score(true_values, pred_values, average='weighted')
            metrics[f'{metric_type}_recall'] = recall_score(true_values, pred_values, average='weighted')
            metrics[f'{metric_type}_f1'] = f1_score(true_values, pred_values, average='weighted')
        return metrics

    def display_metrics(self):
        """Prints out calculated performance metrics."""
        metrics = self.calculate_metrics()
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")

class ConversationDataPreprocessing:
    def __init__(self, df):
        self.df = df
    
    def preprocess_features(self):
        # Add message and word count features
        self.df['message_count'] = self.df['text'].apply(lambda x: len(x.split('\n')))
        self.df['word_count'] = self.df['text'].apply(lambda x: len(x.split()))

    def add_sentiment_score(self):
        # Map sentiment to numerical values
        sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
        self.df['sentiment_score'] = self.df['call_sentiment'].map(sentiment_mapping)

class SentimentAnalysis:
    def __init__(self, df):
        self.df = df

    def plot_sentiment_distribution(self):
        sentiment_counts = self.df['call_sentiment'].value_counts(normalize=True) * 100
        fig = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values,
                     title='Distribution of Sentiments in Customer Conversations (Percentage)',
                     labels={'x': 'Sentiment', 'y': 'Percentage'})
        fig.update_layout(xaxis_title="Sentiment", yaxis_title="Percentage")
        fig.show()

    def plot_outcome_distribution(self):
        outcome_counts = self.df['call_outcome'].value_counts(normalize=True) * 100
        fig = px.bar(outcome_counts, x=outcome_counts.index, y=outcome_counts.values,
                     title='Distribution of Call Outcomes (Percentage)',
                     labels={'x': 'Call Outcome', 'y': 'Percentage'})
        fig.update_layout(xaxis_title="Call Outcome", yaxis_title="Percentage")
        fig.show()

    def plot_sentiment_vs_outcome(self):
        sentiment_outcome = (self.df.groupby(['call_sentiment', 'call_outcome'])
                             .size().groupby(level=0).apply(lambda x: 100 * x / x.sum()).reset_index(name='percentage'))
        fig = px.bar(sentiment_outcome, x='call_sentiment', y='percentage', color='call_outcome',
                     title='Sentiment vs Call Outcome (Percentage)', barmode='stack')
        fig.update_layout(xaxis_title="Sentiment", yaxis_title="Percentage of Outcomes")
        fig.show()

    def plot_word_count_by_sentiment(self):
        fig = px.box(self.df, x='call_sentiment', y='word_count', title='Word Count by Sentiment')
        fig.update_layout(xaxis_title="Sentiment", yaxis_title="Word Count")
        fig.show()

    def plot_message_count_by_sentiment(self):
        fig = px.box(self.df, x='call_sentiment', y='message_count', title='Message Count by Sentiment')
        fig.update_layout(xaxis_title="Sentiment", yaxis_title="Message Count")
        fig.show()

class CommonWordsAnalysis:
    def __init__(self, df):
        self.df = df

    def plot_common_words(self, sentiment, max_words=10):
        text_data = self.df[self.df['call_sentiment'] == sentiment]['text'].dropna()
        
        # Create a custom stop words list by combining default 'english' stop words with custom stop words
        custom_stop_words = list(text.ENGLISH_STOP_WORDS.union(['customer', 'support', 'technical', 'pa', 'agent','member']))
        
        # Initialize CountVectorizer with the custom stop words list
        vectorizer = CountVectorizer(stop_words=custom_stop_words, max_features=max_words)
        
        X = vectorizer.fit_transform(text_data)
        word_counts = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0]))

        fig = go.Figure(data=[go.Bar(
            x=list(word_counts.keys()),
            y=list(word_counts.values()),
            text=list(word_counts.values()),
            textposition='auto'
        )])
        fig.update_layout(title=f'Most Common Words in {sentiment.capitalize()} Sentiment Conversations',
                          xaxis_title='Words', yaxis_title='Frequency')
        fig.show()

class AgentAnalysis:
    def __init__(self, df, agent_cols):
        self.df = df
        self.agent_cols = agent_cols

    def sentiment_summary_by_agent(self):
        sentiment_summary = self.df.groupby(self.agent_cols)['sentiment_score'].mean()
        print("Average Sentiment Score by Agent Type:\n", sentiment_summary)

    def outcome_summary_by_agent(self):
        outcome_summary = self.df.groupby(self.agent_cols)['call_outcome'].value_counts(normalize=True).unstack()
        print("\nCall Outcome Proportion by Agent Type:\n", outcome_summary)

    def plot_sentiment_score_by_agent(self):
        agent_sentiment_df = self.df.melt(
            id_vars=['sentiment_score', 'call_outcome'],
            value_vars=self.agent_cols,
            var_name="agent_type",
            value_name="flag"
        )
        agent_sentiment_df = agent_sentiment_df[agent_sentiment_df['flag'] == 1]
        fig = px.box(agent_sentiment_df, x="agent_type", y="sentiment_score", 
                     title="Sentiment Score by Agent Type",
                     labels={"agent_type": "Agent Type", "sentiment_score": "Sentiment Score"})
        fig.show()

    def plot_call_outcome_by_agent(self):
        agent_sentiment_df = self.df.melt(
            id_vars=['sentiment_score', 'call_outcome'],
            value_vars=self.agent_cols,
            var_name="agent_type",
            value_name="flag"
        )
        agent_sentiment_df = agent_sentiment_df[agent_sentiment_df['flag'] == 1]
        fig = px.histogram(agent_sentiment_df, x="agent_type", color="call_outcome", 
                           title="Call Outcome Proportion by Agent Type", barmode="group",
                           labels={"agent_type": "Agent Type", "call_outcome": "Call Outcome"})
        fig.show()

class StatisticalTests:
    def __init__(self, df):
        self.df = df

    def chi_square_test(self, agent_col, outcome_col='call_outcome'):
        contingency_table = pd.crosstab(self.df[agent_col], self.df[outcome_col])
        chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
        print(f"Chi-square Test for {agent_col} and {outcome_col}:")
        print(f"Chi2: {chi2}, p-value: {p}\n")

    def anova_test(self, df, agent_cols):
        melted_df = df.melt(id_vars=['sentiment_score'], value_vars=agent_cols, 
                            var_name='agent_type', value_name='flag')
        melted_df = melted_df[melted_df['flag'] == 1]
        model = ols('sentiment_score ~ C(agent_type)', data=melted_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print("ANOVA Results:\n", anova_table)

class CorrelationAnalysis:
    def __init__(self, df, agent_cols):
        self.df = df
        self.agent_cols = agent_cols

    def plot_correlation_heatmap(self):
        outcome_mapping = {'resolved': 1, 'follow-up needed': 0}
        self.df['call_outcome_numeric'] = self.df['call_outcome'].map(outcome_mapping)
        correlation_df = self.df[self.agent_cols + ['sentiment_score', 'call_outcome_numeric']]
        corr_matrix = correlation_df.corr()

        fig = ff.create_annotated_heatmap(
            z=corr_matrix.values, 
            x=list(corr_matrix.columns), 
            y=list(corr_matrix.index),
            colorscale='Viridis'
        )
        fig.update_layout(title="Correlation Heatmap of Agent Types, Sentiment, and Call Outcome")
        fig.show()


# COMMAND ----------


folder_path = "/dbfs/FileStore/transcripts_v3/"

test_data = {
    'file_number': [105,108,111,167,173,3,192,22,64,49,199,12,10,101,107,128,151,184],
    'test_sentiment': ['negative','negative','negative','negative','negative','negative','neutral','neutral','neutral','neutral','neutral','neutral','positive','positive','positive','positive','positive','positive',],
    'test_outcome': ['follow-up action needed','follow-up action needed','issue resolved','follow-up action needed','issue resolved','follow-up action needed','follow-up action needed','follow-up action needed','follow-up action needed','follow-up action needed','follow-up action needed','issue resolved','issue resolved','issue resolved','issue resolved','follow-up action needed','follow-up action needed','follow-up action needed']
}

# COMMAND ----------


llm_params = {
    "OPENAI_API_TYPE": "azure",
    "OPENAI_API_VERSION": "",
    "OPENAI_API_BASE": "",
    "OPENAI_API_KEY": "",
    "deployment_name": "gpt-4o-mini",
    "model_name": "gpt-4o-mini"
}

# Instantiate and use the analyser
analyser = ConversationAnalyser(folder_path, test_data, llm_params)
analyser.process_transcripts()
analyser.display_metrics()
transcripts_df = analyser.transcripts_df


# COMMAND ----------


# Sample Usage
# Initialize data and preprocessing
preprocessor = ConversationDataPreprocessing(transcripts_df)
preprocessor.preprocess_features()
preprocessor.add_sentiment_score()

# Sentiment and outcome analysis
sentiment_analysis = SentimentAnalysis(transcripts_df)
sentiment_analysis.plot_sentiment_distribution()
sentiment_analysis.plot_outcome_distribution()
sentiment_analysis.plot_sentiment_vs_outcome()
sentiment_analysis.plot_word_count_by_sentiment()
sentiment_analysis.plot_message_count_by_sentiment()

# Common words analysis
common_words_analysis = CommonWordsAnalysis(transcripts_df)
common_words_analysis.plot_common_words('negative')
common_words_analysis.plot_common_words('positive')

# Agent analysis
agent_cols = ['customer_support_flag', 'technical_support_flag', 'pa_agent_flag', 'agent_flag']
agent_analysis = AgentAnalysis(transcripts_df, agent_cols)
agent_analysis.sentiment_summary_by_agent()
agent_analysis.outcome_summary_by_agent()
agent_analysis.plot_sentiment_score_by_agent()
agent_analysis.plot_call_outcome_by_agent()

# Statistical tests
stat_tests = StatisticalTests(transcripts_df)
for agent in agent_cols:
    stat_tests.chi_square_test(agent)

stat_tests.anova_test(transcripts_df, agent_cols)

# Correlation analysis
correlation_analysis = CorrelationAnalysis(transcripts_df, agent_cols)
correlation_analysis.plot_correlation_heatmap()

