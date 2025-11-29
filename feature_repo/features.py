"""
Feast Feature Definitions for Fraud Detection
"""

from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, String

# Define the transaction entity with old-style ValueType
transaction_entity = Entity(
    name="transaction_id",
    join_keys=["transaction_id"],
    value_type=ValueType.INT64,
    description="Unique transaction identifier"
)

# Define the data source
transaction_source = FileSource(
    path="../data/transactions.parquet",
    timestamp_field="event_timestamp"
)

# Define features (using new Field syntax for feature view)
feature_fields = [Field(name=f"V{i}", dtype=Float64) for i in range(1, 29)]
feature_fields.append(Field(name="Amount", dtype=Float64))
feature_fields.append(Field(name="location", dtype=String))

# Create the Feature View
transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction_entity],
    ttl=timedelta(days=365),
    schema=feature_fields,
    source=transaction_source,
    online=True,
    tags={"team": "fraud_detection"}
)
