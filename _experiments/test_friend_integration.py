"""
Integration Test - Simulates Friend's Backend Calling PrepGen
This script simulates the complete flow from friend's frontend → backend → PrepGen
"""

import requests
import time
import json
from pathlib import Path

# Configuration
PREPGEN_URL = "http://localhost:8000"  # Change to ngrok URL when testing remotely
TEST_FILE = "sample.txt"

def print_step(step_num, description):
    """Print formatted step header"""
    print(f"\n{'='*70}")
    print(f"[Step {step_num}] {description}")
    print('='*70)

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_info(message):
    """Print info message"""
    print(f"ℹ️  {message}")

def print_result(label, value):
    """Print result"""
    print(f"📊 {label}: {value}")

def simulate_full_integration():
    """
    Simulates the complete integration flow:
    User → Friend's Frontend → Friend's Backend → Your PrepGen → Response back
    """
    
    print("\n" + "="*70)
    print(" "*15 + "🤝 FRIEND INTEGRATION SIMULATION")
    print("="*70)
    print_info(f"PrepGen URL: {PREPGEN_URL}")
    print_info(f"Test file: {TEST_FILE}")
    
    # Ensure test file exists
    if not Path(TEST_FILE).exists():
        print_info(f"Creating test file: {TEST_FILE}")
        with open(TEST_FILE, 'w', encoding='utf-8') as f:
            f.write("""
Machine Learning is a subset of artificial intelligence that focuses on building systems 
that can learn from and make decisions based on data. It involves training algorithms on 
large datasets to recognize patterns and make predictions. Common applications include 
image recognition, natural language processing, recommendation systems, and autonomous vehicles.

Deep Learning is a specialized branch of machine learning that uses neural networks with 
multiple layers. These networks can automatically learn hierarchical representations of data, 
making them particularly effective for complex tasks like computer vision and speech recognition.

The field has grown exponentially in recent years due to increased computing power, availability 
of large datasets, and advances in algorithms. Modern AI systems can now perform tasks that 
were previously thought to require human intelligence, such as playing complex games, 
translating languages, and even generating creative content.
            """.strip())
        print_success(f"Created {TEST_FILE}")
    
    session_id = None
    
    # ========================================
    # STEP 1: Health Check
    # ========================================
    print_step(1, "Friend's Backend: Health Check PrepGen")
    print_info("Friend's backend wants to verify PrepGen is online")
    
    try:
        response = requests.get(f"{PREPGEN_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print_success("PrepGen is online!")
            print_result("Status", health.get('status'))
            print_result("AI Models Loaded", health.get('ai_models_loaded'))
            print_result("Active Sessions", health.get('active_sessions', 0))
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to PrepGen: {e}")
        print(f"\nMake sure PrepGen is running:")
        print(f"  cd PrepGen")
        print(f"  python main.py")
        return
    
    # ========================================
    # STEP 2: User Uploads File via Frontend
    # ========================================
    print_step(2, "User: Upload File via Friend's Frontend")
    print_info("User clicks 'Upload Document' in frontend")
    print_info("Frontend sends file to friend's backend")
    print_info("Friend's backend forwards to PrepGen /upload")
    
    try:
        with open(TEST_FILE, 'rb') as f:
            files = {'file': (TEST_FILE, f, 'text/plain')}
            start_time = time.time()
            response = requests.post(f"{PREPGEN_URL}/upload", files=files, timeout=30)
            upload_duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            session_id = data.get('session_id')
            print_success(f"File uploaded in {upload_duration:.2f}s")
            print_result("Session ID", session_id)
            print_result("Filename", data.get('filename'))
            print_result("Pages/Chunks", data.get('num_pages', 'N/A'))
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(response.text)
            return
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return
    
    if not session_id:
        print("❌ No session ID received")
        return
    
    # ========================================
    # STEP 3: User Requests Summary (First Time - No Cache)
    # ========================================
    print_step(3, "User: Request Summary (First Time)")
    print_info("User clicks 'Generate Summary' in frontend")
    print_info("Friend's backend calls PrepGen /summarize")
    print_info("This is the FIRST request - NO CACHE available")
    
    try:
        payload = {
            "session_id": session_id,
            "max_length": 150,
            "min_length": 50
        }
        
        print_info("Generating summary (this may take 10-15 seconds)...")
        start_time = time.time()
        response = requests.post(
            f"{PREPGEN_URL}/summarize", 
            json=payload, 
            timeout=120
        )
        first_duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            summary = data.get('summary', '')
            print_success(f"Summary generated in {first_duration:.2f}s")
            print_result("Summary Length", f"{len(summary)} characters")
            print("\n📝 Summary:")
            print("-" * 70)
            print(summary)
            print("-" * 70)
        else:
            print(f"❌ Summarization failed: {response.status_code}")
            print(response.text)
            return
    except Exception as e:
        print(f"❌ Summarization error: {e}")
        return
    
    # ========================================
    # STEP 4: User Requests Summary Again (Cache Test)
    # ========================================
    print_step(4, "User: Request Summary Again (Testing Cache)")
    print_info("User clicks 'Generate Summary' again (or refreshes)")
    print_info("Friend's backend calls PrepGen /summarize with same params")
    print_info("This is the SECOND request - CACHE should be used! 🚀")
    
    try:
        print_info("Generating summary (should be MUCH faster)...")
        start_time = time.time()
        response = requests.post(
            f"{PREPGEN_URL}/summarize", 
            json=payload, 
            timeout=120
        )
        second_duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Summary generated in {second_duration:.2f}s")
            
            # Calculate speedup
            speedup = first_duration / second_duration if second_duration > 0 else 0
            print_result("First request", f"{first_duration:.2f}s")
            print_result("Second request (cached)", f"{second_duration:.2f}s")
            print_result("🚀 SPEEDUP", f"{speedup:.1f}x FASTER!")
            
            if speedup > 10:
                print_success("✨ CACHE IS WORKING PERFECTLY!")
            elif speedup > 2:
                print_success("Cache is working (moderate speedup)")
            else:
                print("⚠️  Cache might not be active (similar times)")
        else:
            print(f"❌ Second summarization failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # ========================================
    # STEP 5: Check Session Stats
    # ========================================
    print_step(5, "Admin: Check Session Statistics")
    print_info("Friend's admin panel checks session stats")
    
    try:
        response = requests.get(f"{PREPGEN_URL}/sessions/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print_success("Session stats retrieved")
            print_result("Total Sessions", stats.get('total_sessions'))
            print_result("Active Sessions", stats.get('active_sessions'))
            
            if stats.get('sessions'):
                print("\n📋 Active Sessions:")
                for session in stats['sessions'][:3]:  # Show first 3
                    print(f"   • {session.get('session_id', 'N/A')[:8]}... - "
                          f"{session.get('filename', 'N/A')} - "
                          f"{session.get('status', 'N/A')}")
    except Exception as e:
        print(f"⚠️  Stats error: {e}")
    
    # ========================================
    # STEP 6: User Asks Question (RAG)
    # ========================================
    print_step(6, "User: Ask Question About Document")
    print_info("User types question in frontend chat")
    print_info("Friend's backend calls PrepGen /ask")
    print_info("This tests RAG (Retrieval-Augmented Generation)")
    
    try:
        question_payload = {
            "session_id": session_id,
            "question": "What is machine learning and what are its applications?"
        }
        
        print_info(f"Question: {question_payload['question']}")
        print_info("Generating answer (this may take 30-60 seconds)...")
        
        start_time = time.time()
        response = requests.post(
            f"{PREPGEN_URL}/ask",
            json=question_payload,
            timeout=120
        )
        rag_duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get('answer', '')
            print_success(f"Answer generated in {rag_duration:.2f}s")
            print("\n💬 Answer:")
            print("-" * 70)
            print(answer)
            print("-" * 70)
        else:
            print(f"❌ RAG failed: {response.status_code}")
            print(response.text)
    except requests.exceptions.Timeout:
        print("⚠️  RAG timeout (this can take up to 60 seconds)")
    except Exception as e:
        print(f"❌ RAG error: {e}")
    
    # ========================================
    # STEP 7: Generate Quiz
    # ========================================
    print_step(7, "User: Generate Quiz Questions")
    print_info("User clicks 'Generate Quiz' in frontend")
    print_info("Friend's backend calls PrepGen /quiz")
    
    try:
        quiz_payload = {
            "session_id": session_id,
            "num_questions": 3,
            "difficulty": "medium"
        }
        
        print_info("Generating quiz (this may take 30-60 seconds)...")
        
        start_time = time.time()
        response = requests.post(
            f"{PREPGEN_URL}/quiz",
            json=quiz_payload,
            timeout=120
        )
        quiz_duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            questions = data.get('questions', [])
            print_success(f"Quiz generated in {quiz_duration:.2f}s")
            print_result("Number of Questions", len(questions))
            
            print("\n📝 Quiz Questions:")
            print("-" * 70)
            for i, q in enumerate(questions[:3], 1):  # Show first 3
                print(f"\n{i}. {q.get('question', 'N/A')}")
                for j, option in enumerate(q.get('options', []), 1):
                    print(f"   {chr(64+j)}. {option}")
                print(f"   ✓ Answer: {q.get('correct_answer', 'N/A')}")
            print("-" * 70)
        else:
            print(f"❌ Quiz generation failed: {response.status_code}")
            print(response.text)
    except requests.exceptions.Timeout:
        print("⚠️  Quiz timeout (this can take up to 60 seconds)")
    except Exception as e:
        print(f"❌ Quiz error: {e}")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*70)
    print(" "*20 + "✅ INTEGRATION TEST COMPLETE!")
    print("="*70)
    
    print("\n📊 Performance Summary:")
    print(f"  • Upload:           {upload_duration:.2f}s")
    print(f"  • First Summary:    {first_duration:.2f}s")
    print(f"  • Cached Summary:   {second_duration:.2f}s (🚀 {speedup:.1f}x faster)")
    
    print("\n✅ What This Simulated:")
    print("  1. Friend's frontend receives user input")
    print("  2. Friend's backend calls your PrepGen API")
    print("  3. PrepGen processes requests (with caching!)")
    print("  4. Results flow back to frontend")
    print("  5. User sees results in their browser")
    
    print("\n🎯 Integration Points Tested:")
    print("  ✅ Health check endpoint")
    print("  ✅ File upload endpoint")
    print("  ✅ Summarization endpoint (with cache)")
    print("  ✅ Session stats endpoint")
    print("  ✅ RAG (Ask question) endpoint")
    print("  ✅ Quiz generation endpoint")
    
    print("\n💡 Next Steps:")
    print("  1. If this works: Ready for real friend integration!")
    print("  2. Start ngrok: ngrok http 8000")
    print("  3. Share ngrok URL with friend")
    print("  4. Friend updates their backend config")
    print("  5. Test together with real frontend!")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    simulate_full_integration()
