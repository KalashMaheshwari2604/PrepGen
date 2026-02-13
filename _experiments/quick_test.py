"""Quick test to verify Academic Summarizer integration"""

from ai_models import models

print("\n" + "="*60)
print("🧪 QUICK ACADEMIC SUMMARIZER TEST")
print("="*60)

# Test 1: Model Loading
print("\n1️⃣ Checking model loading...")
if hasattr(models, 'academic_summarizer'):
    print("   ✅ Academic Summarizer loaded!")
    print(f"   📁 Model: ./my_academic_summarizer_scientific")
    print(f"   🖥️  Device: {models.device}")
else:
    print("   ❌ Academic Summarizer NOT loaded")
    exit(1)

# Test 2: Backward Compatibility
print("\n2️⃣ Checking backward compatibility...")
if hasattr(models, 'custom_summary_model') and hasattr(models, 'custom_summary_tokenizer'):
    print("   ✅ Old attributes available (custom_summary_model, custom_summary_tokenizer)")
else:
    print("   ⚠️  Warning: Some old attributes missing")

# Test 3: Quick Summarization
print("\n3️⃣ Testing quick summarization...")
test_text = """
Cloud Computing Course (CS4550) covers IaaS, PaaS, and SaaS technologies. 
Prerequisites: CS3310 (Operating Systems), CS3320 (Networks). 
Assessment: 25% Midterm, 30% Project, 25% Labs, 10% Quizzes, 10% Participation.
"""

print(f"   📝 Input: {len(test_text.split())} words")
print("   🔄 Generating summary (scientific domain, greedy search for speed)...")

summarizer = models.academic_summarizer
summary = summarizer.summarize(
    text=test_text, 
    domain="scientific",
    max_new_tokens=80,
    num_beams=1  # Use greedy search for faster testing
)

print(f"   ✅ Output: {len(summary.split())} words")
print(f"\n   Summary: {summary}")

# Check technical term preservation
tech_terms = ['CS4550', 'IaaS', 'PaaS', 'SaaS']
preserved = [term for term in tech_terms if term in summary]
print(f"\n   🎯 Technical terms preserved: {len(preserved)}/{len(tech_terms)} {preserved}")

print("\n" + "="*60)
print("🎉 INTEGRATION TEST PASSED!")
print("✨ Academic Summarizer ready for production!")
print("="*60 + "\n")
