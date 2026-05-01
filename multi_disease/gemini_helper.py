import google.generativeai as genai
try:
    from .config import Config
except (ImportError, ValueError):
    from config import Config

genai.configure(api_key=Config.GEMINI_API_KEY)
model = genai.GenerativeModel(Config.GEMINI_MODEL)

class GeminiHelper:
    
    @staticmethod
    def get_chatbot_response(user_message, chat_history=[]):
        try:
            context = "You are a helpful medical assistant chatbot for Medisense AI. You provide health advice, answer questions about diseases, exercise, mental health, diet, and wellness. Be conversational, empathetic, and provide accurate information. If the user greets you, respond naturally. If they ask about health issues, provide helpful guidance."
            
            conversation = context + "\n\n"
            for msg in chat_history:
                conversation += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n"
            conversation += f"User: {user_message}\nAssistant:"
            
            response = model.generate_content(conversation)
            return response.text
        
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
            response = model.generate_content(prompt)
            return response.text
        
        except Exception as e:
            return f"Error generating recommendation: {str(e)}"