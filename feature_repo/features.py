"""
Feast Feature Definitions for Fraud Detection
"""

from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

# Define the transaction entity
transaction_entity = Entity(
    name="transaction_id",
    join_keys=["transaction_id"],
    description="Unique transaction identifier"
)

# Define the data source (pointing to our Parquet file)
transaction_source = FileSource(
    path="../data/transactions.parquet",
    timestamp_field="event_timestamp"
)

# Define features (V1-V28, Amount, location)
feature_fields = [Field(name=f"V{i}", dtype=Float64) for i in range(1, 29)]
feature_fields.append(Field(name="Amount", dtype=Float64))
feature_fields.append(Field(name="location", dtype=String))

# Create the Feature View
transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction_entity],
    ttl=timedelta(days=365),  # Features valid for 1 year
    schema=feature_fields,
    source=transaction_source,
    online=True,
    tags={"team": "fraud_detection"}
)
