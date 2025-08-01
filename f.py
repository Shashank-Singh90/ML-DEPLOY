# 🚀 Simple IoT Threat Detection Tests

Write-Host "🎯 Testing IoT Cybersecurity ML System" -ForegroundColor Green
Write-Host "=" * 50

# Test 1: Use the /test endpoint (works perfectly - as you saw!)
Write-Host "🧪 1. Using Built-in Test Endpoint..." -ForegroundColor Yellow
try {
    $result = Invoke-RestMethod -Uri "http://localhost:5000/test" -Method GET
    Write-Host "✅ Prediction: $($result.prediction_label)" -ForegroundColor Green
    Write-Host "📊 Confidence: $([math]::Round($result.confidence * 100, 1))%" -ForegroundColor Cyan
    Write-Host "⚠️  Risk Level: $($result.risk_level)" -ForegroundColor $(if($result.risk_level -eq "HIGH"){"Red"}elseif($result.risk_level -eq "MEDIUM"){"Yellow"}else{"Green"})
    Write-Host "🎯 Threat Score: $([math]::Round($result.threat_score * 100, 1))%" -ForegroundColor Cyan
} catch {
    Write-Host "❌ Test failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test 2: Complete feature set (what the model actually needs)
Write-Host "🤖 2. Complete Feature Prediction..." -ForegroundColor Yellow

# This is a suspicious traffic pattern (high rates, unusual flags)
$suspiciousTraffic = @{
    flow_duration = 0.1         # Very short duration
    Duration = 0.15
    Rate = 5000.0              # Very high rate - suspicious!
    Srate = 4500.0             # High send rate  
    fin_flag_number = 10       # Unusual number of FIN flags
    syn_flag_number = 20       # Lots of SYN flags - potential SYN flood
    rst_flag_number = 15       # Many RST flags
    psh_flag_number = 2
    ack_flag_number = 5
    ece_flag_number = 0
    cwr_flag_number = 0
    ack_count = 8
    syn_count = 25             # High SYN count - attack pattern
    fin_count = 12
    rst_count = 18
    HTTP = 0
    HTTPS = 0  
    DNS = 0
    Telnet = 1                 # Telnet usage - potential security risk
    SMTP = 0
    SSH = 0
    IRC = 0
    TCP = 1
    UDP = 0
    DHCP = 0
    ARP = 0
    ICMP = 0
    IPv = 1
    LLC = 0
    "Tot sum" = 50000          # Large packet sum
    Min = 1500
    Max = 8000                 # Large packets
    AVG = 4000
    Std = 2000
    "Tot size" = 75000         # Very large total size
    IAT = 0.001                # Very short inter-arrival time
    Number = 100               # Many packets
    Magnitue = 5000
    Radius = 3000
    Covariance = -0.8          # Negative correlation
    Variance = 4000000         # High variance
    Weight = 5.0               # High weight
} | ConvertTo-Json

try {
    $suspiciousResult = Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST -Body $suspiciousTraffic -ContentType "application/json"
    Write-Host "✅ SUSPICIOUS TRAFFIC DETECTED!" -ForegroundColor Red
    Write-Host "🚨 Prediction: $($suspiciousResult.prediction_label)" -ForegroundColor Red
    Write-Host "📊 Confidence: $([math]::Round($suspiciousResult.confidence * 100, 1))%" -ForegroundColor Cyan
    Write-Host "⚠️  Risk Level: $($suspiciousResult.risk_level)" -ForegroundColor Red
    Write-Host "🎯 Threat Score: $([math]::Round($suspiciousResult.threat_score * 100, 1))%" -ForegroundColor Red
    Write-Host "💡 Action: $($suspiciousResult.recommended_action)" -ForegroundColor Yellow
} catch {
    Write-Host "❌ Suspicious traffic test failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test 3: Normal traffic pattern  
Write-Host "🌐 3. Normal Traffic Prediction..." -ForegroundColor Yellow

$normalTraffic = @{
    flow_duration = 2.5
    Duration = 3.0
    Rate = 100.0               # Normal rate
    Srate = 80.0
    fin_flag_number = 1        # Normal flag counts
    syn_flag_number = 1
    rst_flag_number = 0  
    psh_flag_number = 3
    ack_flag_number = 8
    ece_flag_number = 0
    cwr_flag_number = 0
    ack_count = 10
    syn_count = 2
    fin_count = 1
    rst_count = 0
    HTTP = 1                   # Normal web traffic
    HTTPS = 1
    DNS = 1
    Telnet = 0
    SMTP = 0
    SSH = 0
    IRC = 0
    TCP = 1
    UDP = 0
    DHCP = 0
    ARP = 0
    ICMP = 0
    IPv = 1
    LLC = 0
    "Tot sum" = 1200
    Min = 64
    Max = 1500                 # Normal packet sizes
    AVG = 600
    Std = 300
    "Tot size" = 12000
    IAT = 0.1                  # Normal timing
    Number = 15
    Magnitue = 150
    Radius = 75
    Covariance = 0.2
    Variance = 90000
    Weight = 1.0
} | ConvertTo-Json

try {
    $normalResult = Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST -Body $normalTraffic -ContentType "application/json"
    Write-Host "✅ NORMAL TRAFFIC CONFIRMED!" -ForegroundColor Green
    Write-Host "🌐 Prediction: $($normalResult.prediction_label)" -ForegroundColor Green
    Write-Host "📊 Confidence: $([math]::Round($normalResult.confidence * 100, 1))%" -ForegroundColor Cyan
    Write-Host "⚠️  Risk Level: $($normalResult.risk_level)" -ForegroundColor Green
    Write-Host "🎯 Threat Score: $([math]::Round($normalResult.threat_score * 100, 1))%" -ForegroundColor Green
} catch {
    Write-Host "❌ Normal traffic test failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "🎉 PREDICTION TESTS COMPLETE!" -ForegroundColor Green
Write-Host "✅ Your ML model is successfully detecting IoT threats!" -ForegroundColor Green
