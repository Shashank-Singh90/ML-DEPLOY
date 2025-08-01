#!/bin/bash
# 🚀 Production Deployment Script for IoT Cybersecurity ML System

set -e  # Exit on any error

echo "🚀 Starting Production Deployment..."
echo "=" * 60

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed"
    exit 1
fi

print_status "Prerequisites check passed"

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose.production.yml down --remove-orphans || true
print_status "Existing containers stopped"

# Build images
echo "🏗️  Building production images..."
docker-compose -f docker-compose.production.yml build --no-cache
print_status "Images built successfully"

# Start services
echo "🚀 Starting production services..."
docker-compose -f docker-compose.production.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Health checks
echo "🏥 Performing health checks..."

# Check ML API
if curl -f http://localhost:5000/health > /dev/null 2>&1; then
    print_status "ML API is healthy"
else
    print_error "ML API health check failed"
    exit 1
fi

# Check Prometheus
if curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
    print_status "Prometheus is healthy"
else
    print_warning "Prometheus health check failed"
fi

# Check Grafana
if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
    print_status "Grafana is healthy"
else
    print_warning "Grafana health check failed"
fi

# Display service status
echo ""
echo "📊 Service Status:"
docker-compose -f docker-compose.production.yml ps

# Display access URLs
echo ""
echo "🌐 Access URLs:"
echo "  📊 ML API:     http://localhost:5000"
echo "  📈 Grafana:    http://localhost:3000 (admin/iot_admin_2025)"
echo "  🔍 Prometheus: http://localhost:9090"
echo "  🧪 MLflow:     http://localhost:5001"

# Display logs command
echo ""
echo "📝 To view logs:"
echo "  docker-compose -f docker-compose.production.yml logs -f"

# Display stop command
echo ""
echo "🛑 To stop all services:"
echo "  docker-compose -f docker-compose.production.yml down"

echo ""
print_status "Production deployment completed successfully! 🎉"