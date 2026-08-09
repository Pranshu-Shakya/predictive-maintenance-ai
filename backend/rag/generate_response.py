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


def build_rag_query(
    sensor_data,
    predicted_fault,
    failure_probability,
):
    """
    Build a fault-specific semantic search query
    for the RAG system.
    """

    fault_context = {

        "Bearing Failure": [
            "bearing failure",
            "bearing wear",
            "bearing overheating",
            "bearing vibration",
            "bearing lubrication",
            "bearing inspection",
            "bearing maintenance",
        ],

        "Cavitation": [
            "pump cavitation",
            "low suction pressure",
            "low flow rate",
            "NPSH",
            "cavitation symptoms",
            "cavitation inspection",
            "pump maintenance",
        ],

        "Shaft Misalignment": [
            "shaft misalignment",
            "pump shaft alignment",
            "excessive vibration",
            "coupling alignment",
            "shaft inspection",
            "alignment maintenance",
        ],

        "Normal": [
            "normal pump operation",
            "preventive maintenance",
            "routine pump inspection",
        ],

        "General Mechanical Failure": [
            "pump mechanical failure",
            "pump troubleshooting",
            "mechanical inspection",
            "preventive maintenance",
        ],
    }


    keywords = fault_context.get(
        predicted_fault,
        [
            "industrial pump troubleshooting",
            "machine maintenance",
        ],
    )


    query = f"""
    Industrial predictive maintenance troubleshooting.

    Predicted fault:
    {predicted_fault}

    Failure probability:
    {failure_probability:.2%}

    Machine sensor conditions:

    Temperature:
    {sensor_data["Temperature"]} °C

    Vibration:
    {sensor_data["Vibration"]} mm/s

    Pressure:
    {sensor_data["Pressure"]} bar

    RPM:
    {sensor_data["RPM"]}

    Operating hours:
    {sensor_data["Operating_Hours"]}

    Flow rate:
    {sensor_data["Flow_Rate"]} L/min

    Relevant maintenance topics:
    {", ".join(keywords)}

    Find maintenance knowledge related to the
    predicted fault, symptoms, causes, inspection,
    troubleshooting and corrective actions.
    """

    return query


# ==========================================
# Generate troubleshooting response
# ==========================================

def generate_troubleshooting_response(
    sensor_data,
    predicted_fault,
    failure_probability,
    fault_probabilities
):

    query = build_rag_query(
        sensor_data,
        predicted_fault,
        failure_probability
    )

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

Your task is to interpret the ML prediction using
retrieved maintenance knowledge.

IMPORTANT RULES:

1. Use the retrieved maintenance knowledge as the
   primary technical reference.

2. Do not invent maintenance procedures, specifications,
   thresholds, or safety requirements.

3. The ML prediction represents a probability, not a
   guaranteed machine failure.

4. Clearly distinguish between:
   - ML prediction
   - Retrieved technical information
   - Recommended actions

5. If the retrieved information is insufficient,
   explicitly say that additional inspection or
   technical documentation is required.

6. Provide practical but conservative recommendations.

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