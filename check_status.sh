#!/bin/bash
echo "=== GKE Deployment Status ==="
echo ""
echo "1. Cluster:"
kubectl cluster-info | head -2
echo ""
echo "2. Nodes:"
kubectl get nodes
echo ""
echo "3. Deployments:"
kubectl get deployments
echo ""
echo "4. Pods:"
kubectl get pods -l app=fraud-detection
echo ""
echo "5. Services:"
kubectl get services
echo ""
echo "6. HPA:"
kubectl get hpa
echo ""
EXTERNAL_IP=$(kubectl get service fraud-detection-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ -n "$EXTERNAL_IP" ]; then
  echo "7. API URL: http://$EXTERNAL_IP"
  echo "   Docs: http://$EXTERNAL_IP/docs"
else
  echo "7. Waiting for external IP..."
fi
echo ""
echo "================================"
