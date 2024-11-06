import pandas as pd

# Define the test data
test_data = {
    'file_number': [105, 108, 111, 167, 173, 3, 192, 22, 64, 49, 199, 12, 10, 101, 107, 128, 151, 184],
    'test_sentiment': [
        'negative', 'negative', 'negative', 'negative', 'negative', 'negative', 'neutral', 'neutral',
        'neutral', 'neutral', 'neutral', 'neutral', 'positive', 'positive', 'positive', 'positive',
        'positive', 'positive'
    ],
    'test_outcome': [
        'follow-up action needed', 'follow-up action needed', 'issue resolved', 'follow-up action needed',
        'issue resolved', 'follow-up action needed', 'follow-up action needed', 'follow-up action needed',
        'follow-up action needed', 'follow-up action needed', 'follow-up action needed', 'issue resolved',
        'issue resolved', 'issue resolved', 'issue resolved', 'follow-up action needed', 'follow-up action needed',
        'follow-up action needed'
    ]
}

# Convert the dictionary into a DataFrame
test_df = pd.DataFrame(test_data)
