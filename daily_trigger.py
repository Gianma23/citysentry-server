import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import torch.nn as nn
from datetime import datetime, timedelta, timezone
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

cred = credentials.Certificate("city-sentry-firebase.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

#############################################
#               Cluster Data
#############################################
end_date = datetime.now(timezone.utc)
start_date = end_date - timedelta(weeks=2)

reports_ref = db.collection("reports")
query = reports_ref.where("timestamp", ">=", start_date).where("timestamp", "<=", end_date)
docs = query.stream()
reports = [doc.to_dict() for doc in docs]
df = pd.DataFrame(reports)
print(df.head())


tag_groups = {
    'environmental': ['Litter', 'Illegal Dumping', 'Air Pollution', 'Water Pollution'],
    'infrastructure': ['Pothole', 'Cracked Pavement', 'Broken Streetlight', 'Damaged Bench', 'Blocked Drainage', 'Abandoned Vehicle'],
    'safety': ['Vandalism', 'Unsafe Building', 'Unsafe Bridge', 'Broken Traffic Signals', 'Open Manholes'],
    'aesthetic': ['Overgrown Vegetation', 'Graffiti', 'Neglected Monuments', 'Faded Paint'],
}

eps = 0.008
min_samples = 3

def cluster_reports_by_group(data, group_name, tags):
    results = []
    data = data[data['tags'].apply(lambda x: any(tag in tags for tag in x))]
    if not data.empty:
        # Extract coordinates from the 'location' field
        coords = np.array([[loc['latitude'], loc['longitude']] for loc in data['location']])

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        data['cluster'] = db.labels_
        
        # For each identified cluster (ignoring noise with label -1)
        for cluster_id in set(db.labels_):
            if cluster_id != -1:
                cluster_points = data[data['cluster'] == cluster_id]
                # Calculate the centroid as the mean latitude and longitude
                centroid_lat = cluster_points['location'].apply(lambda loc: loc['latitude']).mean()
                centroid_lon = cluster_points['location'].apply(lambda loc: loc['longitude']).mean()
                
                results.append({
                    'group': group_name,
                    'start_date': start_date,
                    'lat': centroid_lat,
                    'lon': centroid_lon,
                    'volume': len(cluster_points),
                })
    
    return results

all_results = []
for group_name, tags in tag_groups.items():
    group_results = cluster_reports_by_group(df, group_name, tags)
    all_results.extend(group_results)

clusters_df = pd.DataFrame(all_results)

clusters_data = clusters_df.to_dict(orient='records')
collection = db.collection("clusters")
for record in clusters_data:
    collection.add(record)
    
#############################################
#            Model Definition
#############################################

class DeepSetEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=64):
        """
        Encodes a set of clusters (each with `input_dim` features) into a fixed-length vector.
        """
        super(DeepSetEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        x: Tensor of shape (batch, num_clusters, 3)
        """
        mask = (x.abs().sum(dim=-1) != 0).float()  # Mask out zero-padded clusters
        embed = self.mlp(x)

        aggregated_avg = (embed * mask.unsqueeze(-1)).sum(dim=2) / mask.sum(dim=2).clamp(min=1).unsqueeze(-1)
        count = mask.sum(dim=2).unsqueeze(-1)  # shape: (batch, seq_len, 1)
        aggregated = torch.cat([aggregated_avg, count], dim=-1)

        return aggregated  # Average the embeddings of all clusters

class ForecastModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_hidden_dim, future_steps=1, max_clusters=10):
        super(ForecastModel, self).__init__()
        self.future_steps = future_steps
        self.max_clusters = max_clusters
        self.input_dim = input_dim

        self.deepset = DeepSetEncoder(input_dim, hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim+1, lstm_hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        
        self.cluster_count_predictor = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim//2),
            nn.LeakyReLU(),
            nn.Linear(lstm_hidden_dim//2, 1),
        )
        
        self.cluster_decoder = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(lstm_hidden_dim, max_clusters * input_dim)
        )  # Predict cluster properties

    def forward(self, x):
        aggregated = self.deepset(x)
        lstm_out, _ = self.lstm(aggregated)
        # Predict the number of clusters
        num_clusters = self.cluster_count_predictor(lstm_out[:, -1, :])

        # Predict cluster properties for each cluster
        cluster_predictions = self.cluster_decoder(lstm_out[:, -1, :])
        cluster_predictions = cluster_predictions.view(-1, self.max_clusters, self.input_dim)
        # Mask out predictions for non-existent clusters
        mask = torch.arange(self.max_clusters, device=cluster_predictions.device).unsqueeze(0) < num_clusters.round()
        cluster_predictions = cluster_predictions * mask.unsqueeze(-1).float()
        cluster_predictions = torch.where(cluster_predictions == 0, torch.tensor(0.0, device=cluster_predictions.device), cluster_predictions)
        
        return torch.squeeze(num_clusters), torch.unsqueeze(cluster_predictions, 1)


#############################################

MAX_CLUSTERS = 10
SEQ_LEN = 30
PADDING_VALUE = 0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for group in tag_groups.keys():
    clusters_ref = db.collection("clusters")
    query = clusters_ref.where("group", "==", group).order_by("start_date", direction=firestore.Query.DESCENDING)
    docs = query.stream()
    clusters = [doc.to_dict() for doc in docs]
    clusters_df = pd.DataFrame(clusters)
    if clusters_df.empty:
        print(f"No clusters found for group '{group}'.")
        continue
    print(clusters_df.head())
    clusters_df['start_date'] = pd.to_datetime(clusters_df['start_date'])
    unique_dates = clusters_df['start_date'].sort_values(ascending=False).unique()[:SEQ_LEN]

    # Create a (SEQ_LEN, MAX_CLUSTERS, 3) tensor
    tensor_data = np.full((SEQ_LEN, MAX_CLUSTERS, 3), PADDING_VALUE, dtype=np.float32)
    
    for i, date in enumerate(unique_dates):
        date_clusters = clusters_df[clusters_df['start_date'] == date].head(MAX_CLUSTERS)  # Take up to MAX_CLUSTERS for that date
        tensor_data[i, :len(date_clusters), :] = date_clusters[['lat', 'lon', 'volume']].to_numpy()
    
    input_tensor = torch.tensor(tensor_data).unsqueeze(0)
    
    # Load the model and perform inference
    model = ForecastModel(input_dim=3, 
                        hidden_dim=128,
                        lstm_hidden_dim=512,
                        future_steps=1,
                        max_clusters=MAX_CLUSTERS)
    model.to(device)
    model.load_state_dict(torch.load(f"models/seq_30_eps_0.008_window_14_{group}_model.pth"))  # Update with your model path

    model.eval()
    with torch.no_grad():        
        n_clusters, predictions = model(input_tensor.to(device))
        print(f"Predicted {n_clusters.round()} clusters for group '{group}'.")
        predicted_clusters = predictions.squeeze(0).squeeze(0).detach().cpu().numpy()  # shape: (max_clusters, features)

    # Get a timestamp for record keeping
    timestamp = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()

# Iterate over each cluster and save it as a separate document
for idx, cluster in enumerate(predicted_clusters):
    print(cluster)
    if cluster[2] <= 0:  
        continue

    cluster_data = {
        "group": group,
        "pred_date": timestamp,
        "latitude": float(cluster[0]),
        "longitude": float(cluster[1]),
        "volume": float(cluster[2]),
    }
    
    collection = db.collection("predictions")
    collection.add(cluster_data)

print("Predictions saved to Firestore.")
