from llama_index.llms import Groq
from llama_index.prompts import PromptTemplate
from llama_index import ServiceContext
from typing import Dict, Any
import os

class ASKEvaluator:
    def __init__(self, groq_api_key: str):
        """Initialize the ASK Evaluator with Groq credentials."""
        # Initialize Groq LLM
        self.llm = Groq(
            api_key=groq_api_key,
            model_name="mixtral-8x7b-32768",  # Using Mixtral for its strong reasoning capabilities
            temperature=0.1  # Low temperature for more focused responses
        )
        
        # Create service context
        self.service_context = ServiceContext.from_defaults(llm=self.llm)
        
        # System prompt template
        self.system_prompt = """You are an AI assistant designed to help early-stage entrepreneurs refine and improve their
asks for collaborating with diverse stakeholders. Your role is to guide users in making effectual, clear, and persuasive
asks that maximize potential partnerships and value creation.

When evaluating an ask, consider these key principles:

1. Bird in Hand (Existing Assets):
- What assets/resources are being leveraged?
- How do current strengths benefit both parties?

2. Affordable Loss:
- Does it minimize risk for the stakeholder?
- Is there a low-commitment entry point?

3. Crazy Quilt (Partnership):
- Is it framed as mutual benefit?
- Does it invite collaboration?

4. Lemonade Principle:
- Is there openness to unexpected outcomes?
- Does it allow for flexibility?

5. Pilot-in-the-Plane:
- Does it show active ownership?
- Is there commitment to act on feedback?

Rate each aspect from 1-5 and provide specific suggestions for improvement.
"""

        # Evaluation template
        self.eval_template = PromptTemplate(
            template=(
                "{system_prompt}\n\n"
                "About the entrepreneur: {entrepreneur_context}\n"
                "About the stakeholder: {stakeholder_context}\n"
                "The ask: {ask}\n\n"
                "Please provide:\n"
                "1. Ratings for each principle (1-5)\n"
                "2. Specific improvements needed\n"
                "3. A rewritten version of the ask\n"
            )
        )

    def evaluate_ask(
        self,
        ask: str,
        entrepreneur_context: str,
        stakeholder_context: str
    ) -> Dict[str, Any]:
        """
        Evaluate an ask using the ASK framework and provide detailed feedback.
        
        Args:
            ask: The actual ask/request being made
            entrepreneur_context: Background information about the entrepreneur
            stakeholder_context: Information about the stakeholder being approached
            
        Returns:
            Dictionary containing evaluation results and suggestions
        """
        # Format the prompt
        formatted_prompt = self.eval_template.format(
            system_prompt=self.system_prompt,
            entrepreneur_context=entrepreneur_context,
            stakeholder_context=stakeholder_context,
            ask=ask
        )
        
        # Get evaluation from LLM
        response = self.llm.complete(formatted_prompt)
        
        return {
            "original_ask": ask,
            "evaluation": response.text,
        }

    def batch_evaluate_asks(self, asks: list[Dict[str, str]]) -> list[Dict[str, Any]]:
        """
        Evaluate multiple asks in batch.
        
        Args:
            asks: List of dictionaries containing ask details
                 Each dict should have 'ask', 'entrepreneur_context', 'stakeholder_context'
                 
        Returns:
            List of evaluation results
        """
        results = []
        for ask_data in asks:
            result = self.evaluate_ask(
                ask=ask_data["ask"],
                entrepreneur_context=ask_data["entrepreneur_context"],
                stakeholder_context=ask_data["stakeholder_context"]
            )
            results.append(result)
        return results

# Example usage function
def example_usage():
    # Initialize evaluator
    evaluator = ASKEvaluator(groq_api_key="your-api-key")
    
    # Single ask evaluation
    result = evaluator.evaluate_ask(
        ask="Can you introduce me to potential customers for my software?",
        entrepreneur_context="I'm the founder of a SaaS startup that provides workflow automation tools for small businesses.",
        stakeholder_context="The person is the head of a community of tech innovators with a strong background in scaling startups."
    )
    
    # Batch evaluation
    asks = [
        {
            "ask": "Can you introduce me to potential customers for my software?",
            "entrepreneur_context": "I'm the founder of a SaaS startup...",
            "stakeholder_context": "The person is the head of a community..."
        },
        {
            "ask": "Would you be interested in being an advisor?",
            "entrepreneur_context": "I'm building an AI-powered analytics platform...",
            "stakeholder_context": "They're a successful entrepreneur with multiple exits..."
        }
    ]
    
    batch_results = evaluator.batch_evaluate_asks(asks)
    return result, batch_results

if __name__ == "__main__":
    example_usage()