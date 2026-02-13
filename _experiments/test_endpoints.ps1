# Quick PowerShell Test Script for PrepGen AI Service
# Tests all endpoints using native PowerShell commands

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "PrepGen AI Service - Endpoint Testing" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$baseUrl = "http://localhost:8000"

# Test 1: Health Check
Write-Host "[TEST 1] Health Check" -ForegroundColor Yellow
Write-Host "GET $baseUrl/health" -ForegroundColor Gray
try {
    $health = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get
    Write-Host "✅ PASSED" -ForegroundColor Green
    Write-Host ($health | ConvertTo-Json) -ForegroundColor White
} catch {
    Write-Host "❌ FAILED: $_" -ForegroundColor Red
}

Write-Host "`n----------------------------------------`n"

# Test 2: Session Stats
Write-Host "[TEST 2] Session Statistics" -ForegroundColor Yellow
Write-Host "GET $baseUrl/sessions/stats" -ForegroundColor Gray
try {
    $stats = Invoke-RestMethod -Uri "$baseUrl/sessions/stats" -Method Get
    Write-Host "✅ PASSED" -ForegroundColor Green
    Write-Host ($stats | ConvertTo-Json) -ForegroundColor White
} catch {
    Write-Host "❌ FAILED: $_" -ForegroundColor Red
}

Write-Host "`n----------------------------------------`n"

# Test 3: Upload File
Write-Host "[TEST 3] File Upload" -ForegroundColor Yellow

# Create test file if it doesn't exist
if (-not (Test-Path "sample.txt")) {
    Write-Host "Creating sample.txt..." -ForegroundColor Gray
    @"
Machine Learning is a subset of artificial intelligence that focuses on building systems 
that can learn from and make decisions based on data. It involves training algorithms on 
large datasets to recognize patterns and make predictions.

Deep Learning is a specialized branch of machine learning that uses neural networks with 
multiple layers. These networks can automatically learn hierarchical representations of data.
"@ | Out-File -FilePath "sample.txt" -Encoding UTF8
}

Write-Host "POST $baseUrl/upload" -ForegroundColor Gray
try {
    # PowerShell file upload
    $filePath = "sample.txt"
    $fileBytes = [System.IO.File]::ReadAllBytes($filePath)
    $fileEnc = [System.Text.Encoding]::GetEncoding('iso-8859-1').GetString($fileBytes)
    $boundary = [System.Guid]::NewGuid().ToString()
    
    $LF = "`r`n"
    $bodyLines = (
        "--$boundary",
        "Content-Disposition: form-data; name=`"file`"; filename=`"sample.txt`"",
        "Content-Type: text/plain$LF",
        $fileEnc,
        "--$boundary--$LF"
    ) -join $LF

    $upload = Invoke-RestMethod -Uri "$baseUrl/upload" `
        -Method Post `
        -ContentType "multipart/form-data; boundary=$boundary" `
        -Body $bodyLines
    
    $sessionId = $upload.session_id
    Write-Host "✅ PASSED" -ForegroundColor Green
    Write-Host ($upload | ConvertTo-Json) -ForegroundColor White
    Write-Host "`nSession ID: $sessionId" -ForegroundColor Cyan
} catch {
    Write-Host "❌ FAILED: $_" -ForegroundColor Red
    $sessionId = $null
}

if ($sessionId) {
    Write-Host "`n----------------------------------------`n"
    
    # Test 4: Summarization (First time - no cache)
    Write-Host "[TEST 4] Summarization (Building Cache)" -ForegroundColor Yellow
    Write-Host "POST $baseUrl/summarize" -ForegroundColor Gray
    try {
        $body = @{
            session_id = $sessionId
            max_length = 150
            min_length = 50
        } | ConvertTo-Json
        
        $startTime = Get-Date
        $summary1 = Invoke-RestMethod -Uri "$baseUrl/summarize" `
            -Method Post `
            -ContentType "application/json" `
            -Body $body
        $duration1 = (Get-Date) - $startTime
        
        Write-Host "✅ PASSED" -ForegroundColor Green
        Write-Host "Duration: $($duration1.TotalSeconds) seconds" -ForegroundColor White
        Write-Host ($summary1 | ConvertTo-Json) -ForegroundColor White
        
        # Test 5: Summarization (Second time - with cache)
        Write-Host "`n[TEST 5] Summarization (Using Cache)" -ForegroundColor Yellow
        Write-Host "POST $baseUrl/summarize" -ForegroundColor Gray
        
        $startTime = Get-Date
        $summary2 = Invoke-RestMethod -Uri "$baseUrl/summarize" `
            -Method Post `
            -ContentType "application/json" `
            -Body $body
        $duration2 = (Get-Date) - $startTime
        
        Write-Host "✅ PASSED" -ForegroundColor Green
        Write-Host "Duration: $($duration2.TotalSeconds) seconds" -ForegroundColor White
        
        if ($duration2.TotalSeconds -gt 0) {
            $speedup = $duration1.TotalSeconds / $duration2.TotalSeconds
            Write-Host "🚀 Cache Speedup: $([math]::Round($speedup, 1))x faster!" -ForegroundColor Magenta
        }
    } catch {
        Write-Host "❌ FAILED: $_" -ForegroundColor Red
    }
    
    Write-Host "`n----------------------------------------`n"
    
    # Test 6: Quiz Generation
    Write-Host "[TEST 6] Quiz Generation" -ForegroundColor Yellow
    Write-Host "POST $baseUrl/quiz" -ForegroundColor Gray
    Write-Host "(This may take 30-60 seconds...)" -ForegroundColor Gray
    try {
        $body = @{
            session_id = $sessionId
            num_questions = 3
            difficulty = "medium"
        } | ConvertTo-Json
        
        $startTime = Get-Date
        $quiz = Invoke-RestMethod -Uri "$baseUrl/quiz" `
            -Method Post `
            -ContentType "application/json" `
            -Body $body `
            -TimeoutSec 120
        $duration = (Get-Date) - $startTime
        
        Write-Host "✅ PASSED" -ForegroundColor Green
        Write-Host "Duration: $($duration.TotalSeconds) seconds" -ForegroundColor White
        Write-Host ($quiz | ConvertTo-Json -Depth 5) -ForegroundColor White
    } catch {
        Write-Host "❌ FAILED: $_" -ForegroundColor Red
    }
    
    Write-Host "`n----------------------------------------`n"
    
    # Test 7: Ask Question (RAG)
    Write-Host "[TEST 7] Ask Question (RAG)" -ForegroundColor Yellow
    Write-Host "POST $baseUrl/ask" -ForegroundColor Gray
    Write-Host "(This may take 30-60 seconds...)" -ForegroundColor Gray
    try {
        $body = @{
            session_id = $sessionId
            question = "What is machine learning?"
        } | ConvertTo-Json
        
        $startTime = Get-Date
        $answer = Invoke-RestMethod -Uri "$baseUrl/ask" `
            -Method Post `
            -ContentType "application/json" `
            -Body $body `
            -TimeoutSec 120
        $duration = (Get-Date) - $startTime
        
        Write-Host "✅ PASSED" -ForegroundColor Green
        Write-Host "Duration: $($duration.TotalSeconds) seconds" -ForegroundColor White
        Write-Host ($answer | ConvertTo-Json) -ForegroundColor White
    } catch {
        Write-Host "❌ FAILED: $_" -ForegroundColor Red
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Testing Complete!" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Install ngrok: choco install ngrok (or download from ngrok.com)" -ForegroundColor White
Write-Host "2. Run in new terminal: ngrok http 8000" -ForegroundColor White
Write-Host "3. Copy the ngrok URL (https://xxxx.ngrok-free.app)" -ForegroundColor White
Write-Host "4. Share with your friend" -ForegroundColor White
Write-Host "`nSee NGROK_SETUP_GUIDE.md for detailed instructions`n" -ForegroundColor Cyan
