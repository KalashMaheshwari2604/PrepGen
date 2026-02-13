# SIMPLER QUIZ GENERATION FOR LLAMA 3.2 3B
# This approach uses a format the smaller model can handle reliably

def generate_quiz_simple_format(context: str, num_questions: int, llm) -> list:
    """
    Uses a simpler text format that Llama 3.2 3B can generate reliably,
    then parses it into the required JSON structure.
    """
    
    prompt = f"""[INST]
Generate {num_questions} multiple choice questions based on this content.

Format each question EXACTLY like this:

Q: [question text]
A) [option 1]
B) [option 2]  
C) [option 3]
D) [option 4]
ANSWER: [letter]
EXPLANATION: [why it's correct]

Content:
{context[:5000]}

Generate {num_questions} questions now:
[/INST]"""

    output = llm(prompt, max_tokens=2000, temperature=0.5)
    text = output['choices'][0]['text']
    
    # Parse the simple format into JSON
    questions = []
    current_q = {}
    options = []
    
    for line in text.split('\n'):
        line = line.strip()
        
        if line.startswith('Q:'):
            if current_q and 'question' in current_q:
                questions.append(current_q)
            current_q = {'question': line[2:].strip()}
            options = []
            
        elif line.startswith(('A)', 'B)', 'C)', 'D)')):
            options.append(line[2:].strip())
            if len(options) == 4:
                current_q['options'] = options.copy()
                
        elif line.startswith('ANSWER:'):
            answer_letter = line[7:].strip().upper()[0]
            if answer_letter in 'ABCD' and 'options' in current_q:
                idx = ord(answer_letter) - ord('A')
                current_q['correct_answer'] = current_q['options'][idx]
                
        elif line.startswith('EXPLANATION:'):
            current_q['explanation'] = line[12:].strip()
            current_q['difficulty'] = 'easy'
    
    if current_q and 'question' in current_q:
        questions.append(current_q)
    
    return questions

# This format is MUCH easier for smaller models to generate consistently
