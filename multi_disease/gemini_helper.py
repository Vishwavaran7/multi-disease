import os
try:
    from .config import Config
except (ImportError, ValueError):
    from config import Config

# Try to use the new google-genai package if available, otherwise fall back to the
# older google.generativeai package. This keeps the app working while preferring
# the supported library.
new_client = None
old_model = None
new_available = False
old_available = False

try:
    from google import genai as genai_new
    new_available = True
    try:
        # Prefer passing API key to the client if accepted, otherwise set env var
        try:
            new_client = genai_new.Client(api_key=Config.GEMINI_API_KEY)
        except TypeError:
            os.environ['GENAI_API_KEY'] = Config.GEMINI_API_KEY
            new_client = genai_new.Client()
    except Exception:
        new_client = None
except Exception:
    genai_new = None

try:
    import google.generativeai as genai_old
    old_available = True
    genai_old.configure(api_key=Config.GEMINI_API_KEY)
    old_model = genai_old.GenerativeModel(Config.GEMINI_MODEL)
except Exception:
    genai_old = None


def _generate_with_new(prompt):
    if new_client is None:
        raise RuntimeError('New GenAI client not available')
    resp = new_client.generate_text(model=Config.GEMINI_MODEL, input=prompt)
    # Attempt common response attributes
    if hasattr(resp, 'text') and resp.text:
        return resp.text
    out = getattr(resp, 'output', None)
    if out:
        # output may be list-like
        first = out[0]
        if isinstance(first, dict):
            return first.get('content') or str(resp)
        return getattr(first, 'content', str(resp))
    return str(resp)


def _generate_with_old(prompt):
    if old_model is None:
        raise RuntimeError('Old GenerativeAI model not available')
    resp = old_model.generate_content(prompt)
    return getattr(resp, 'text', str(resp))


class GeminiHelper:
    @staticmethod
    def get_chatbot_response(user_message, chat_history=[]):
        try:
            context = (
                "You are a helpful medical assistant chatbot for Medisense AI. "
                "You provide health advice, answer questions about diseases, exercise, mental health, diet, and wellness. "
                "Be conversational, empathetic, and provide accurate information. If the user greets you, respond naturally. "
                "If they ask about health issues, provide helpful guidance."
            )

            conversation = context + "\n\n"
            for msg in chat_history:
                conversation += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n"
            conversation += f"User: {user_message}\nAssistant:"

            if new_available:
                return _generate_with_new(conversation)
            elif old_available:
                return _generate_with_old(conversation)
            else:
                return "Error: No GenAI client available"
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    def get_treatment_recommendation(disease_type, prediction_result, risk_score, input_features):
        try:
            prompt = f"""
You are a medical advisor for Medisense AI. A patient has been diagnosed with the following:

Disease Type: {disease_type}
Prediction Result: {prediction_result}
Risk Score: {risk_score}%
Patient Data: {input_features}

IMPORTANT: Format your response as clean, well-structured paragraphs with proper sections. Do NOT use markdown symbols like ###, **, *, or ---. Instead, use clear section headings followed by detailed explanations in paragraph form.

Provide a detailed treatment recommendation with these sections:

IMMEDIATE ACTIONS TO TAKE:
[Write detailed paragraph here]

LIFESTYLE MODIFICATIONS:
Diet: [Write paragraph]
Exercise: [Write paragraph]
Sleep: [Write paragraph]

WHEN TO CONSULT A DOCTOR:
[Write detailed paragraph here]

PREVENTIVE MEASURES:
[Write detailed paragraph here]

IMPORTANT WARNINGS:
[Write detailed paragraph here]

Be professional, empathetic, and actionable. Write in clear, flowing paragraphs without any markdown formatting.
"""
            if new_available:
                return _generate_with_new(prompt)
            elif old_available:
                return _generate_with_old(prompt)
            else:
                return "Error: No GenAI client available"

        except Exception as e:
            return f"Error generating recommendation: {str(e)}"