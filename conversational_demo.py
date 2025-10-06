from qa_engine import QAEngine

def run_conversational_qa():
    """
    Simulates a conversational flow over a fixed context, demonstrating
    the model's contextual understanding and robustness to unanswerable queries.
    """
    print("--- Contextual & Robust Question Answering Demo ---")

    # 1. Initialize the QA Engine
    # Set the threshold slightly lower than the max score (e.g., 0.8) to distinguish 
    # between "no answer" and a low-confidence extracted answer.
    engine = QAEngine(score_threshold=0.8) 

    if not engine.qa_pipeline:
        print("\nDemo failed to start due to model loading error.")
        return

    # 2. Define the main context (The "document" the user is querying)
    context = (
        "The James Webb Space Telescope (JWST) is a large infrared observatory launched in December 2021. "
        "It was developed through an international collaboration between NASA, ESA, and the Canadian Space Agency (CSA). "
        "JWST is designed to study the formation of stars and galaxies, and to detect light from the first galaxies formed after the Big Bang."
    )
    
    print("\n--- New Context Set ---")
    print(context)
    print("----------------------")
    
    # Set the context for the engine
    engine.set_context(context)

    # 3. Simulate a conversation flow
    
    # Turn 1: Direct question
    print("\n[Turn 1] Direct Question")
    q1 = "When was the James Webb Space Telescope launched?"
    a1 = engine.answer_question(q1)
    print(a1)
    
    # Turn 2: Contextual Follow-up (Relies on the first context set)
    print("\n[Turn 2] Contextual Follow-up")
    q2 = "Who were the agencies involved in its development?"
    a2 = engine.answer_question(q2)
    print(a2)

    # Turn 3: Robustness test (Unanswerable Question - SQuAD 2.0 Style)
    # The answer is not in the context. The low confidence score should trigger the rejection logic.
    print("\n[Turn 3] Unanswerable Question Test")
    q3 = "How many astronauts operate it in space?"
    a3 = engine.answer_question(q3)
    print(a3)

    # Turn 4: Contextual Deepening
    print("\n[Turn 4] Contextual Deepening")
    q4 = "What is the telescope primarily designed to study?"
    a4 = engine.answer_question(q4)
    print(a4)
    
    print("\n--- Demo Complete ---")


if __name__ == '__main__':
    # Ensure all model/pipeline loading happens inside the function 
    # to maintain control over the output timing.
    run_conversational_qa()
