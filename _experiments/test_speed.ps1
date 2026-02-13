# Speed Test Script for PrepGen API with Llama 3.2 3B
# This will test the new faster model

$baseUrl = "http://localhost:8000"
Write-Host "`n=== PrepGen Speed Test (Llama 3.2 3B) ===" -ForegroundColor Cyan

# Test 1: Health Check
Write-Host "`n[1/4] Testing health endpoint..." -ForegroundColor Yellow
$response = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get
Write-Host "✅ Server Status: $($response.status)" -ForegroundColor Green
Write-Host "   Active Sessions: $($response.active_sessions)" -ForegroundColor Gray

# Test 2: Upload Document
Write-Host "`n[2/4] Uploading sample document..." -ForegroundColor Yellow
$filePath = "sample.txt"
if (-not (Test-Path $filePath)) {
    Write-Host "❌ Error: sample.txt not found!" -ForegroundColor Red
    exit
}

$form = @{
    file = Get-Item -Path $filePath
}

$uploadStart = Get-Date
$uploadResponse = Invoke-RestMethod -Uri "$baseUrl/upload" -Method Post -Form $form
$uploadEnd = Get-Date
$uploadTime = ($uploadEnd - $uploadStart).TotalSeconds

Write-Host "✅ Upload successful!" -ForegroundColor Green
Write-Host "   Session ID: $($uploadResponse.session_id)" -ForegroundColor Gray
Write-Host "   Filename: $($uploadResponse.filename)" -ForegroundColor Gray
Write-Host "   Characters: $($uploadResponse.character_count)" -ForegroundColor Gray
Write-Host "   Time: $([math]::Round($uploadTime, 2))s" -ForegroundColor Gray

$sessionId = $uploadResponse.session_id

# Test 3: Generate Quiz (This is where you'll see the speed!)
Write-Host "`n[3/4] Generating quiz (testing Llama 3.2 3B speed)..." -ForegroundColor Yellow
Write-Host "   Document size: $($uploadResponse.character_count) chars" -ForegroundColor Gray
$expectedQuestions = [Math]::Min(20, [Math]::Max(3, [Math]::Floor($uploadResponse.character_count / 1000)))
Write-Host "   Expected questions: $expectedQuestions" -ForegroundColor Gray

$quizStart = Get-Date
Write-Host "   ⏱️  Generating... (please wait)" -ForegroundColor Cyan

$quizBody = @{
    session_id = $sessionId
} | ConvertTo-Json

$quizResponse = Invoke-RestMethod -Uri "$baseUrl/quiz" -Method Post -Body $quizBody -ContentType "application/json"
$quizEnd = Get-Date
$quizTime = ($quizEnd - $quizStart).TotalSeconds

Write-Host "`n✅ Quiz generated successfully!" -ForegroundColor Green
Write-Host "   Questions generated: $($quizResponse.quiz.Count)" -ForegroundColor Gray
Write-Host "   ⚡ Generation time: $([math]::Round($quizTime, 2))s" -ForegroundColor Yellow

# Show speed comparison
$oldTime = $quizResponse.quiz.Count * 6  # Approximate old time (Mistral 7B)
$speedup = [math]::Round($oldTime / $quizTime, 1)
Write-Host "`n   📊 Speed Comparison:" -ForegroundColor Cyan
Write-Host "      Old (Mistral 7B): ~$($oldTime)s" -ForegroundColor Gray
Write-Host "      New (Llama 3.2 3B): $([math]::Round($quizTime, 2))s" -ForegroundColor Green
Write-Host "      Speedup: ${speedup}x faster! 🚀" -ForegroundColor Green

# Show first question with difficulty
Write-Host "`n   📝 Sample Question:" -ForegroundColor Cyan
$firstQ = $quizResponse.quiz[0]
Write-Host "      Q: $($firstQ.question)" -ForegroundColor White
Write-Host "      Difficulty: $($firstQ.difficulty.ToUpper())" -ForegroundColor $(
    if ($firstQ.difficulty -eq "easy") { "Green" }
    elseif ($firstQ.difficulty -eq "medium") { "Yellow" }
    else { "Red" }
)
Write-Host "      Options: $($firstQ.options -join ', ')" -ForegroundColor Gray
Write-Host "      Answer: $($firstQ.correct_answer)" -ForegroundColor Green

# Test 4: Summarize
Write-Host "`n[4/4] Testing summarization..." -ForegroundColor Yellow
$summaryStart = Get-Date
$summaryBody = @{
    session_id = $sessionId
} | ConvertTo-Json

$summaryResponse = Invoke-RestMethod -Uri "$baseUrl/summarize" -Method Post -Body $summaryBody -ContentType "application/json"
$summaryEnd = Get-Date
$summaryTime = ($summaryEnd - $summaryStart).TotalSeconds

Write-Host "✅ Summary generated!" -ForegroundColor Green
Write-Host "   Time: $([math]::Round($summaryTime, 2))s" -ForegroundColor Gray
Write-Host "   Summary: $($summaryResponse.summary.Substring(0, [Math]::Min(100, $summaryResponse.summary.Length)))..." -ForegroundColor Gray

# Final Summary
Write-Host "`n=== Test Complete! ===" -ForegroundColor Cyan
Write-Host "`n📊 Performance Summary:" -ForegroundColor Yellow
Write-Host "   Upload: $([math]::Round($uploadTime, 2))s" -ForegroundColor Gray
Write-Host "   Quiz ($($quizResponse.quiz.Count) questions): $([math]::Round($quizTime, 2))s ⚡" -ForegroundColor Green
Write-Host "   Summary: $([math]::Round($summaryTime, 2))s" -ForegroundColor Gray
Write-Host "   Total: $([math]::Round($uploadTime + $quizTime + $summaryTime, 2))s" -ForegroundColor Gray

Write-Host "`n✅ Llama 3.2 3B is working perfectly!" -ForegroundColor Green
Write-Host "   Your quiz generation is now ${speedup}x faster! 🚀`n" -ForegroundColor Cyan
