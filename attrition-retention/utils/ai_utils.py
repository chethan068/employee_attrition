import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Warning: GEMINI_API_KEY not found in environment variables")
else:
    genai.configure(api_key=api_key)

def generate_retention_strategies(employee_data: dict) -> list:
    """
    Generate personalized retention strategies based on employee data using Gemini AI.
    
    Args:
        employee_data (dict): Dictionary containing employee attributes
    
    Returns:
        list: List of retention strategy recommendations
    """
    # Convert input data to a more readable format
    formatted_data = "\n".join([f"{k}: {v}" for k, v in employee_data.items()])
    
    prompt = f"""
    Based on the following employee details, suggest the best retention strategies to reduce attrition.
    Employee Data:
    {formatted_data}
    Consider factors such as salary, work-life balance, job satisfaction, performance, and career growth.
    Provide actionable strategies in bullet points.
    """
    
    try:
        if not api_key:
            return ["API key not configured. Unable to generate retention strategies."]
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        
        # Parse the response into a list of strategies
        strategies = [line.strip() for line in response.text.split("\n") if line.strip()]
        return strategies
    except Exception as e:
        return [f"Error generating retention strategies: {str(e)}"]

