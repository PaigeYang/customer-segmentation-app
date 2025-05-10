# segmentation.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Simple RFM segmentation using KMeans

def segment_customers(df):
    # Ensure required columns are present
    required_cols = ['CustomerID', 'TotalSpend', 'PurchaseCount', 'LastPurchaseDate']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.copy()

    # Convert LastPurchaseDate to datetime and calculate recency
    df['LastPurchaseDate'] = pd.to_datetime(df['LastPurchaseDate'])
    df['Recency'] = (pd.Timestamp.today() - df['LastPurchaseDate']).dt.days

    # RFM features
    rfm = df[['TotalSpend', 'PurchaseCount', 'Recency']]
    rfm.columns = ['Monetary', 'Frequency', 'Recency']

    # Normalize features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(rfm_scaled)
    df['Segment'] = clusters

    # Aggregate results
    segments = []
    for seg_id in sorted(df['Segment'].unique()):
        seg_df = df[df['Segment'] == seg_id]
        stats = {
            'avg_spend': round(seg_df['TotalSpend'].mean(), 2),
            'avg_frequency': round(seg_df['PurchaseCount'].mean(), 2),
            'avg_recency': round(seg_df['Recency'].mean(), 1),
            'count': len(seg_df)
        }
        segments.append({
            'name': f"Segment {seg_id}",
            'stats': stats
        })

    return segments