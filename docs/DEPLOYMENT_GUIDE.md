# IoT Predictive Maintenance System - Deployment Guide

## 🚀 Production Deployment Guide

This comprehensive guide covers the deployment of the IoT Predictive Maintenance System with the enhanced dashboard and all Batch 3 features.

## 📋 Prerequisites

### System Requirements

#### Minimum Hardware Requirements
- **CPU**: 4 cores, 2.5 GHz
- **RAM**: 8 GB
- **Storage**: 50 GB available space
- **Network**: Stable internet connection

#### Recommended Hardware Requirements
- **CPU**: 8 cores, 3.0 GHz
- **RAM**: 16 GB
- **Storage**: 100 GB SSD
- **Network**: High-speed connection (100 Mbps+)

### Software Dependencies

#### Core Dependencies
```bash
# Python 3.9+
python --version  # Should be >= 3.9

# Required Python packages
pip install -r requirements.txt
```

#### System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev build-essential libssl-dev libffi-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel openssl-devel libffi-devel

# macOS
brew install python@3.9
xcode-select --install
```

## 🛠️ Installation Process

### 1. Environment Setup

#### Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

#### Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### 2. Database Setup

#### Initialize Database
```bash
# Run database migrations
python scripts/setup_database.py

# Verify database connection
python scripts/verify_setup.py
```

#### Sample Data (Optional)
```bash
# Load sample data for testing
python scripts/load_sample_data.py
```

### 3. Configuration

#### Environment Configuration
Create environment-specific configuration files:

```bash
# Copy example configuration
cp config/config.example.yaml config/production.yaml

# Edit production configuration
nano config/production.yaml
```

#### Key Configuration Parameters
```yaml
# config/production.yaml
environment: production

database:
  host: localhost
  port: 5432
  name: iot_maintenance
  user: iot_user
  password: ${DB_PASSWORD}

dashboard:
  host: 0.0.0.0
  port: 8050
  debug: false
  secret_key: ${SECRET_KEY}

monitoring:
  refresh_interval: 15
  alert_threshold: 0.8
  max_alerts: 100

logging:
  level: INFO
  file: logs/application.log
  max_size: 100MB
  backup_count: 5
```

#### Environment Variables
```bash
# Create .env file
cat > .env << EOF
# Database Configuration
DB_PASSWORD=your_secure_password
DB_HOST=localhost
DB_PORT=5432

# Application Security
SECRET_KEY=your_secret_key_here
FLASK_ENV=production

# Monitoring
MONITORING_ENABLED=true
ALERT_EMAIL=admin@yourcompany.com

# NASA Data Access
NASA_API_KEY=your_nasa_api_key
EOF
```

### 4. Testing the Installation

#### Run Test Suite
```bash
# Quick tests
python run_tests.py quick

# Full test suite
python run_tests.py all

# Check test coverage
python run_tests.py all --verbose
```

#### Manual Verification
```bash
# Test dashboard startup
python src/presentation/dashboard/enhanced_app.py

# Verify services
python scripts/verify_services.py

# Check system health
python scripts/health_check.py
```

## 🌐 Deployment Options

### Option 1: Standalone Deployment

#### Direct Python Execution
```bash
# Start the enhanced dashboard
python src/presentation/dashboard/enhanced_app.py --host=0.0.0.0 --port=8050

# Or use the application launcher
python app.py --mode=production
```

#### Using Gunicorn (Recommended)
```bash
# Install Gunicorn
pip install gunicorn

# Start with Gunicorn
gunicorn --bind 0.0.0.0:8050 --workers 4 app:server
```

### Option 2: Docker Deployment

#### Build Docker Image
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python scripts/health_check.py || exit 1

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "4", "app:server"]
```

#### Docker Commands
```bash
# Build image
docker build -t iot-maintenance-system .

# Run container
docker run -d \
    --name iot-dashboard \
    -p 8050:8050 \
    --env-file .env \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/data:/app/data \
    iot-maintenance-system

# Check logs
docker logs iot-dashboard

# Stop container
docker stop iot-dashboard
```

### Option 3: Docker Compose Deployment

#### docker-compose.yml
```yaml
version: '3.8'

services:
  dashboard:
    build: .
    ports:
      - "8050:8050"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://postgres:password@db:5432/iot_maintenance
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:13
    environment:
      POSTGRES_DB: iot_maintenance
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - dashboard
    restart: unless-stopped

volumes:
  postgres_data:
```

#### Start with Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🔧 Production Configuration

### Nginx Configuration

#### nginx.conf
```nginx
events {
    worker_connections 1024;
}

http {
    upstream dashboard {
        server dashboard:8050;
    }

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        location / {
            proxy_pass http://dashboard;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

### SSL/TLS Configuration

#### Generate SSL Certificate
```bash
# Using Let's Encrypt (recommended)
sudo apt-get install certbot
sudo certbot certonly --standalone -d your-domain.com

# Or generate self-signed certificate for testing
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
```

### Environment-Specific Settings

#### Production Settings
```python
# config/production.py
import os

class ProductionConfig:
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY')

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Caching
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')

    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'logs/production.log'

    # Security
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    PERMANENT_SESSION_LIFETIME = 1800  # 30 minutes
```

## 📊 Monitoring and Maintenance

### Health Checks

#### Application Health Check
```python
# scripts/health_check.py
import requests
import sys

def check_dashboard_health():
    try:
        response = requests.get('http://localhost:8050/health', timeout=10)
        return response.status_code == 200
    except:
        return False

def check_database_health():
    # Database connection check
    pass

def main():
    checks = [
        ('Dashboard', check_dashboard_health()),
        ('Database', check_database_health()),
    ]

    all_healthy = all(check[1] for check in checks)

    for name, status in checks:
        print(f"{name}: {'✓' if status else '✗'}")

    sys.exit(0 if all_healthy else 1)

if __name__ == '__main__':
    main()
```

### Logging Configuration

#### Structured Logging
```python
# config/logging.py
import logging
import logging.handlers
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        return json.dumps(log_entry)

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        'logs/application.log',
        maxBytes=100*1024*1024,  # 100MB
        backupCount=5
    )
    file_handler.setFormatter(JSONFormatter())

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
```

### Backup Strategy

#### Automated Backup Script
```bash
#!/bin/bash
# scripts/backup.sh

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Database backup
pg_dump iot_maintenance > "$BACKUP_DIR/db_backup_$DATE.sql"

# Application data backup
tar -czf "$BACKUP_DIR/app_backup_$DATE.tar.gz" data/ logs/ config/

# Clean old backups (keep last 7 days)
find "$BACKUP_DIR" -name "*.sql" -mtime +7 -delete
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $DATE"
```

#### Schedule Backups
```bash
# Add to crontab
crontab -e

# Backup every day at 2 AM
0 2 * * * /path/to/scripts/backup.sh
```

## 🔐 Security Considerations

### Security Checklist

#### Application Security
- [ ] Secret keys properly configured
- [ ] Database credentials secured
- [ ] HTTPS enabled
- [ ] Input validation implemented
- [ ] Error messages sanitized
- [ ] Session management secure

#### Infrastructure Security
- [ ] Firewall configured
- [ ] Regular security updates
- [ ] Access controls implemented
- [ ] Monitoring and alerting active
- [ ] Backup and recovery tested
- [ ] SSL certificates valid

### Security Configuration

#### Firewall Rules
```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw enable

# Block direct access to application port
sudo ufw deny 8050/tcp
```

#### Environment Variable Security
```bash
# Use a secrets management system
# Example with systemd environment file
sudo systemctl edit iot-dashboard

# Add:
[Service]
EnvironmentFile=/etc/iot-dashboard/secrets.env
```

## 🚨 Troubleshooting

### Common Issues

#### Dashboard Won't Start
```bash
# Check logs
tail -f logs/application.log

# Verify dependencies
pip check

# Test configuration
python scripts/verify_setup.py
```

#### Performance Issues
```bash
# Check system resources
htop
df -h

# Monitor application metrics
python scripts/performance_monitor.py

# Check database performance
psql -c "SELECT * FROM pg_stat_activity;"
```

#### Database Connection Issues
```bash
# Test database connection
python -c "
from src.infrastructure.database import DatabaseManager
db = DatabaseManager()
print('Database connection:', 'OK' if db.test_connection() else 'FAILED')
"
```

### Log Analysis

#### Key Log Patterns
```bash
# Error patterns
grep -i "error\|exception\|failed" logs/application.log

# Performance issues
grep -i "slow\|timeout\|memory" logs/application.log

# Security events
grep -i "unauthorized\|forbidden\|attack" logs/application.log
```

## 📞 Support and Maintenance

### Maintenance Schedule

#### Daily Tasks
- Monitor system health
- Check error logs
- Verify backup completion

#### Weekly Tasks
- Review performance metrics
- Update security patches
- Test backup restoration

#### Monthly Tasks
- Security audit
- Performance optimization
- Documentation updates

### Contact Information

For deployment support and maintenance:
- **Technical Issues**: Create issue in project repository
- **Security Concerns**: Contact security team immediately
- **Performance Issues**: Monitor dashboard metrics first

---

**Last Updated**: December 2024
**Version**: Batch 3 Final
**Deployment Team**: IoT Maintenance System Team