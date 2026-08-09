import os
import json
from google import genai
from dotenv import load_dotenv

from rag.retriever import retrieve_documents


# ==========================================
# Load environment variables
# ==========================================

load_dotenv()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)


# ==========================================
# Gemini Client
# ==========================================

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# ==========================================
# Generate troubleshooting response
# ==========================================

def generate_troubleshooting_response(
    sensor_data,
    predicted_fault,
    failure_probability,
    fault_probabilities
):

    query = f"""
    Industrial centrifugal pump troubleshooting.

    Predicted fault: {predicted_fault}
    Failure probability: {failure_probability:.2%}

    Temperature: {sensor_data["Temperature"]} °C
    Vibration: {sensor_data["Vibration"]} mm/s
    Pressure: {sensor_data["Pressure"]} bar
    RPM: {sensor_data["RPM"]}
    Operating hours: {sensor_data["Operating_Hours"]}
    Flow rate: {sensor_data["Flow_Rate"]} L/min

    Find relevant causes, symptoms, inspections,
    and recommended maintenance actions.
    """

    documents = retrieve_documents(query)

    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source')}\n"
            f"{doc.page_content}"
            for doc in documents
        ]
    )

    prompt = f"""
You are an industrial predictive maintenance assistant.

Analyze the machine using the ML prediction and retrieved
maintenance knowledge.

IMPORTANT:
- Use retrieved knowledge as the primary source.
- Do not invent maintenance procedures.
- Do not claim the machine will definitely fail.
- Keep recommendations practical.
- If information is insufficient, say so.

MACHINE DATA:

Temperature: {sensor_data["Temperature"]} °C
Vibration: {sensor_data["Vibration"]} mm/s
Pressure: {sensor_data["Pressure"]} bar
RPM: {sensor_data["RPM"]}
Operating Hours: {sensor_data["Operating_Hours"]}
Flow Rate: {sensor_data["Flow_Rate"]} L/min

ML PREDICTION:

Predicted Fault:
{predicted_fault}

Failure Probability:
{failure_probability:.2%}

FAULT CLASSIFICATION PROBABILITIES:

{fault_probabilities}

RETRIEVED KNOWLEDGE:

{context}

Return ONLY valid JSON using exactly this structure:

{{
    "machine_risk": "Low/Medium/High",
    "summary": "Short explanation of the machine condition",
    "why_prediction": [
        "reason 1",
        "reason 2",
        "reason 3"
    ],
    "possible_causes": [
        "cause 1",
        "cause 2",
        "cause 3"
    ],
    "inspection_steps": [
        "step 1",
        "step 2",
        "step 3"
    ],
    "corrective_actions": [
        "action 1",
        "action 2",
        "action 3"
    ],
    "safety_note": "Safety recommendation"
}}

Do not include markdown.
Do not include code fences.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    text = response.text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    try:
        result = json.loads(text)

    except json.JSONDecodeError:

        result = {
            "machine_risk": "Unknown",
            "summary": text,
            "why_prediction": [],
            "possible_causes": [],
            "inspection_steps": [],
            "corrective_actions": [],
            "safety_note": "Follow applicable equipment safety procedures."
        }

    return result, documents