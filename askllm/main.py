from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import groq
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="ASK Evaluator")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are an AI assistant designed to help early-stage entrepreneurs refine and improve their
asks for collaborating with diverse stakeholders. Your role is to guide users in making
effectual, clear, and persuasive asks that maximize potential partnerships and value
creation. Use the principles of effectual reasoning and prompt the user to consider relevant
aspects before making the ask.

When evaluating and improving asks, consider these key principles:

Step 1: Clarify Your Goals (Bird in Hand)
- Understand what you already have within your control
- Identify assets, resources, or connections that align with stakeholder's interests
- Show how current strengths can benefit both parties in the collaboration
Example: "I'm already working with a few early adopters and have developed case studies that show
the impact of our tool on efficiency. I'd love to share these insights with you and see how it
might fit with your network's goals."

Step 2: Consider Affordable Loss
- Approach the ask from a mindset of minimizing risk for the stakeholder
- Offer ways to limit their downside
- Suggest low-commitment ways to help or get involved
Example: "If connecting me to just one or two beta customers would be easier to start with, I can work
closely with them and ensure they see clear value before making further introductions."

Step 3: Build a "Crazy Quilt" (Partnership-Oriented)
- Frame the ask as a partnership
- Instead of simply asking for a favor, offer to collaborate for mutual benefit
- Invite stakeholders to contribute their expertise
Example: "I'd love to hear your advice on how we could adapt our tool for wider impact within your
network. If you're open to it, we could collaborate on a pilot project and tailor the solution
based on your community's needs."

Step 4: Embrace Surprises (Lemonade Principle)
- Be open to unexpected outcomes
- Ask questions that invite new possibilities
- Frame asks with flexibility
Example: "Is there another way we could collaborate that would be more beneficial for your network?
I'm open to exploring other ideas if you think there's a better fit."

Step 5: Pilot-in-the-Plane (Control Your Future)
- Take an active role in shaping the outcome
- Show willingness to co-create mutual benefit
- Emphasize readiness to take ownership and act on feedback
Example: "I believe we could work together to create meaningful value for your network. I'd be happy
to lead the process and make sure it aligns with both our goals."

Final Advice:
Always be open to feedback from the stakeholder. Use the dialogue to deepen the
relationship, ask thoughtful follow-up questions, and co-create opportunities that go beyond
just a single request."""

FEW_SHOT_EXAMPLES = [
    {
        "poor_ask": "Can you introduce me to potential customers for my software?",
        "better_ask": "I've worked with a few early adopters and have results showing a 20% increase in productivity. How can we work together to introduce this to companies in your network that might benefit from these improvements?",
    },
    {
        "poor_ask": "Can you fund our new environmental project?",
        "better_ask": "Our current project has already secured 50% funding, and we're looking for partners who share our sustainability goals. Could we explore ways you could join us to create lasting impact?",
    },
    {
        "poor_ask": "Can you be my advisor?",
        "better_ask": "I admire your expertise in scaling marketplaces. I'm facing a challenge with customer acquisition. Would you be willing to share some insights on how I can address this? If it aligns, we could formalize it into an advisory role.",
    },
    {
        "poor_ask": "Can you give me feedback on my app?",
        "better_ask": "I'm refining the user experience for my app and would love your expert perspective. Would you be open to testing it and giving me your thoughts? Your feedback will directly shape the next version.",
    },
    {
        "poor_ask": "Can we partner to sell your products on our platform?",
        "better_ask": "I've noticed that your products align well with the growing eco-conscious audience we serve. I'd love to explore how we could create a joint campaign to expand both our reach while offering more sustainable products.",
    },
    {
        "poor_ask": "Can we collaborate on content development?",
        "better_ask": "Our EdTech platform has improved student outcomes by 30%. I'd love to hear your thoughts on how we could create joint content that benefits your students and enhances our platform's educational offerings.",
    },
    {
        "poor_ask": "Can you introduce me to some doctors?",
        "better_ask": "I've developed a health monitoring device, and I'm looking to partner with medical professionals to fine-tune it for patient care. Would you be open to introducing me to a few doctors in your network who might be interested?",
    },
    {
        "poor_ask": "Can I get a discount on your products?",
        "better_ask": "We're expanding our product line and will need consistent supplies over the next year. I'd love to explore a long-term partnership where we can both benefit. Could we discuss potential volume-based discounts?",
    },
    {
        "poor_ask": "Will you invest in my company?",
        "better_ask": "We've achieved significant milestones with a 200% year-over-year growth. I'm seeking an investor who can provide strategic guidance, particularly in scaling to new markets. Could we explore how you might be interested in joining us?",
    },
    {
        "poor_ask": "Can you promote my product?",
        "better_ask": "I've noticed that your audience aligns well with the eco-conscious values of our brand. I'd love to collaborate on a campaign that promotes sustainability while providing value to your followers.",
    },
]


class AskEvaluation(BaseModel):
    about_me: str
    about_stakeholder: str
    ask: str


class SystemPromptUpdate(BaseModel):
    new_prompt: str


class FewShotExample(BaseModel):
    poor_ask: str
    better_ask: str


class FewShotExamplesUpdate(BaseModel):
    examples: List[FewShotExample]


def format_few_shot_examples() -> str:
    formatted = "Few-Shot Examples:\n\n"
    for idx, example in enumerate(FEW_SHOT_EXAMPLES, 1):
        formatted += f"{idx}. Poor Ask: \"{example['poor_ask']}\"\n"
        formatted += f"   Better Ask: \"{example['better_ask']}\"\n\n"
    return formatted


def create_prompt(ask_eval: AskEvaluation) -> str:
    full_prompt = f"{SYSTEM_PROMPT}\n\n"
    full_prompt += format_few_shot_examples()
    full_prompt += f"""
Now, please evaluate and improve the following ask:

About the entrepreneur: {ask_eval.about_me}
About the stakeholder: {ask_eval.about_stakeholder}
Current ask: {ask_eval.ask}

Please provide:
1. An analysis of the current ask using effectual reasoning principles
2. A significantly improved version of the ask
3. Specific suggestions for improvement
"""
    return full_prompt


@app.post("/evaluate-ask")
async def evaluate_ask(ask_eval: AskEvaluation):
    try:
        prompt = create_prompt(ask_eval)

        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1000,
        )

        return {
            "evaluation": completion.choices[0].message.content,
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/update-system-prompt")
async def update_system_prompt(prompt_update: SystemPromptUpdate):
    global SYSTEM_PROMPT
    try:
        SYSTEM_PROMPT = prompt_update.new_prompt
        return {"status": "success", "message": "System prompt updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-system-prompt")
async def get_system_prompt():
    return {"system_prompt": SYSTEM_PROMPT}


@app.put("/update-few-shot-examples")
async def update_few_shot_examples(examples_update: FewShotExamplesUpdate):
    global FEW_SHOT_EXAMPLES
    try:
        FEW_SHOT_EXAMPLES = [
            {"poor_ask": ex.poor_ask, "better_ask": ex.better_ask}
            for ex in examples_update.examples
        ]
        return {
            "status": "success",
            "message": "Few-shot examples updated successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-few-shot-examples")
async def get_few_shot_examples():
    return {"few_shot_examples": FEW_SHOT_EXAMPLES}


@app.post("/reset-defaults")
async def reset_defaults():
    global SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
    try:
        SYSTEM_PROMPT = """You are an AI assistant designed to help early-stage entrepreneurs refine and improve their
asks for collaborating with diverse stakeholders. Your role is to guide users in making
effectual, clear, and persuasive asks that maximize potential partnerships and value
creation. Use the principles of effectual reasoning and prompt the user to consider relevant
aspects before making the ask.

When evaluating and improving asks, consider these key principles:

Step 1: Clarify Your Goals (Bird in Hand)
- Understand what you already have within your control
- Identify assets, resources, or connections that align with stakeholder's interests
- Show how current strengths can benefit both parties in the collaboration
Example: "I'm already working with a few early adopters and have developed case studies that show
the impact of our tool on efficiency. I'd love to share these insights with you and see how it
might fit with your network's goals."

Step 2: Consider Affordable Loss
- Approach the ask from a mindset of minimizing risk for the stakeholder
- Offer ways to limit their downside
- Suggest low-commitment ways to help or get involved
Example: "If connecting me to just one or two beta customers would be easier to start with, I can work
closely with them and ensure they see clear value before making further introductions."

Step 3: Build a "Crazy Quilt" (Partnership-Oriented)
- Frame the ask as a partnership
- Instead of simply asking for a favor, offer to collaborate for mutual benefit
- Invite stakeholders to contribute their expertise
Example: "I'd love to hear your advice on how we could adapt our tool for wider impact within your
network. If you're open to it, we could collaborate on a pilot project and tailor the solution
based on your community's needs."

Step 4: Embrace Surprises (Lemonade Principle)
- Be open to unexpected outcomes
- Ask questions that invite new possibilities
- Frame asks with flexibility
Example: "Is there another way we could collaborate that would be more beneficial for your network?
I'm open to exploring other ideas if you think there's a better fit."

Step 5: Pilot-in-the-Plane (Control Your Future)
- Take an active role in shaping the outcome
- Show willingness to co-create mutual benefit
- Emphasize readiness to take ownership and act on feedback
Example: "I believe we could work together to create meaningful value for your network. I'd be happy
to lead the process and make sure it aligns with both our goals."

Final Advice:
Always be open to feedback from the stakeholder. Use the dialogue to deepen the
relationship, ask thoughtful follow-up questions, and co-create opportunities that go beyond
just a single request."""

        FEW_SHOT_EXAMPLES = [
            {
                "poor_ask": "Can you introduce me to potential customers for my software?",
                "better_ask": "I've worked with a few early adopters and have results showing a 20% increase in productivity. How can we work together to introduce this to companies in your network that might benefit from these improvements?",
            },
            {
                "poor_ask": "Can you fund our new environmental project?",
                "better_ask": "Our current project has already secured 50% funding, and we're looking for partners who share our sustainability goals. Could we explore ways you could join us to create lasting impact?",
            },
            {
                "poor_ask": "Can you be my advisor?",
                "better_ask": "I admire your expertise in scaling marketplaces. I'm facing a challenge with customer acquisition. Would you be willing to share some insights on how I can address this? If it aligns, we could formalize it into an advisory role.",
            },
            {
                "poor_ask": "Can you give me feedback on my app?",
                "better_ask": "I'm refining the user experience for my app and would love your expert perspective. Would you be open to testing it and giving me your thoughts? Your feedback will directly shape the next version.",
            },
            {
                "poor_ask": "Can we partner to sell your products on our platform?",
                "better_ask": "I've noticed that your products align well with the growing eco-conscious audience we serve. I'd love to explore how we could create a joint campaign to expand both our reach while offering more sustainable products.",
            },
            {
                "poor_ask": "Can we collaborate on content development?",
                "better_ask": "Our EdTech platform has improved student outcomes by 30%. I'd love to hear your thoughts on how we could create joint content that benefits your students and enhances our platform's educational offerings.",
            },
            {
                "poor_ask": "Can you introduce me to some doctors?",
                "better_ask": "I've developed a health monitoring device, and I'm looking to partner with medical professionals to fine-tune it for patient care. Would you be open to introducing me to a few doctors in your network who might be interested?",
            },
            {
                "poor_ask": "Can I get a discount on your products?",
                "better_ask": "We're expanding our product line and will need consistent supplies over the next year. I'd love to explore a long-term partnership where we can both benefit. Could we discuss potential volume-based discounts?",
            },
            {
                "poor_ask": "Will you invest in my company?",
                "better_ask": "We've achieved significant milestones with a 200% year-over-year growth. I'm seeking an investor who can provide strategic guidance, particularly in scaling to new markets. Could we explore how you might be interested in joining us?",
            },
            {
                "poor_ask": "Can you promote my product?",
                "better_ask": "I've noticed that your audience aligns well with the eco-conscious values of our brand. I'd love to collaborate on a campaign that promotes sustainability while providing value to your followers.",
            },
        ]
        return {"status": "success", "message": "Reset to defaults successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=10000)