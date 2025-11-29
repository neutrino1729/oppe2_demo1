#!/bin/bash

EXTERNAL_IP=35.225.23.174
API_URL="http://$EXTERNAL_IP"

echo "=========================================="
echo "🧪 Fraud Detection API Testing"
echo "=========================================="
echo ""

# Function to test prediction
test_prediction() {
    local test_name=$1
    local amount=$2
    local v1=$3
    
    echo "Test: $test_name"
    echo "Amount: \$$amount"
    
    response=$(curl -s -X POST http://$EXTERNAL_IP/predict \
      -H "Content-Type: application/json" \
      -d "{
        \"Time\": 100.0,
        \"V1\": $v1, \"V2\": 0.3, \"V3\": -0.2, \"V4\": 0.1,
        \"V5\": 0.4, \"V6\": -0.1, \"V7\": 0.2, \"V8\": 0.0,
        \"V9\": -0.3, \"V10\": 0.1, \"V11\": 0.2, \"V12\": -0.1,
        \"V13\": 0.0, \"V14\": 0.3, \"V15\": -0.2, \"V16\": 0.1,
        \"V17\": 0.0, \"V18\": -0.1, \"V19\": 0.2, \"V20\": 0.1,
        \"V21\": 0.0, \"V22\": -0.1, \"V23\": 0.1, \"V24\": 0.0,
        \"V25\": 0.2, \"V26\": -0.1, \"V27\": 0.0, \"V28\": 0.1,
        \"Amount\": $amount
      }")
    
    echo "Response: $response"
    echo ""
}

# Run tests
test_prediction "Small Purchase" 25.50 0.5
test_prediction "Medium Purchase" 150.00 1.2
test_prediction "Large Purchase" 999.00 -2.5
test_prediction "Suspicious Pattern" 500.00 -3.8

echo "=========================================="
echo "✅ API Testing Complete!"
echo "=========================================="
