import os
import sys
from transformers import pipeline

class QAEngine:
    """
    A Question Answering engine wrapper using a Hugging Face pre-trained model
    fine-tuned for SQuAD 2.0 (handling unanswerable questions).

    This engine maintains a context and applies a confidence threshold for robustness.
    """
    def __init__(self, model_name="deepset/roberta-base-squad2", score_threshold=0.8):
        # A low score threshold (e.g., < 0.8) indicates the model is uncertain, 
        # which for SQuAD 2.0 models usually means the question is unanswerable.
        self.model_name = model_name
        self.score_threshold = score_threshold
        self.current_context = ""
        self.qa_pipeline = None
        self._load_model()
        
    def _load_model(self):
        """Initializes the Hugging Face Question Answering pipeline."""
        try:
            print(f"Loading QA model: {self.model_name}...")
            # Initialize the pipeline for question-answering
            self.qa_pipeline = pipeline("question-answering", model=self.model_name, tokenizer=self.model_name)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            self.qa_pipeline = None

    def set_context(self, context: str):
        """Sets the current context/document for the QA engine."""
        self.current_context = context

    def answer_question(self, question: str) -> str:
        """
        Processes a single question against the current context.
        
        Args:
            question: The question string.

        Returns:
            A formatted string containing the answer or a "not found" message.
        """
        if not self.qa_pipeline:
            return f"Q: {question}\nA: Error: Model not initialized."
        
        if not self.current_context:
            return f"Q: {question}\nA: Error: No context has been set."

        # Run the question through the pipeline
        result = self.qa_pipeline(
            question=question,
            context=self.current_context
        )
        
        answer = result['answer']
        score = result['score']

        # Determine if the answer is reliable based on the threshold
        if score >= self.score_threshold:
            # Format confidence to 2 decimal places
            confidence = f"{score * 100:.2f}%"
            return f"Q: {question}\nA: {answer} (Confidence: {confidence})"
        else:
            # If confidence is low, the model is signaling that the question is unanswerable
            confidence = f"{score * 100:.2f}%"
            return f"Q: {question}\nA: I cannot find a relevant answer in the provided context (Low confidence)."
