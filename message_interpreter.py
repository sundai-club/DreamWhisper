from anthropic import Anthropic
import os
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from datetime import datetime
import json

# Load environment variables
load_dotenv()

# Initialize Anthropic client
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

class MessageInterpreter:
    def __init__(self):
        # Initialize Stable Diffusion
        print("Loading Stable Diffusion model (this might take a minute)...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
        print("Model loaded successfully!")
        self.qa_file = "qa_history.json"

    def interpret_message(self, message):
        """Generate interpretation using Claude"""
        try:
            response = anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": f"Please provide a thoughtful interpretation of this message, analyzing its main points and underlying meaning: {message}"
                }]
            )
            return response.content
        except Exception as e:
            return f"Error generating interpretation: {str(e)}"

    def generate_image_prompt(self, interpretation):
        """Generate detailed image prompt using Claude"""
        try:
            response = anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": f"Based on this interpretation: '{interpretation}', create a detailed visual description that could be used as a prompt for Stable Diffusion image generation. Make it vivid and specific, but keep it under 75 words. Focus on visual elements that would make a compelling image."
                }]
            )
            # Extract the text content from the response
            return str(response.content)  # Convert TextBlock to string
        except Exception as e:
            return f"Error generating image prompt: {str(e)}"

    def generate_image(self, prompt):
        """Generate image using Stable Diffusion"""
        try:
            # Check if prompt is an error message
            if prompt.startswith("Error generating image prompt:"):
                raise Exception(prompt)
            
            # Generate the image
            image = self.pipe(prompt).images[0]
            
            # Create 'generated_images' directory if it doesn't exist
            os.makedirs("generated_images", exist_ok=True)
            
            # Save the image with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"generated_images/image_{timestamp}.png"
            image.save(image_path)
            
            return image_path
        except Exception as e:
            return f"Error generating image: {str(e)}"

    def generate_questions(self, message):
        """Generate 5 simple questions using Claude"""
        try:
            response = anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": f"Based on this message, generate 5 thought-provoking but simple questions that would help understand the message better. Format each question on a new line with a number: {message}"
                }]
            )
            return response.content
        except Exception as e:
            return f"Error generating questions: {str(e)}"

    def process_questions_and_answers(self, questions):
        """Process questions one by one and store answers"""
        qa_pairs = []
        
        # Handle both string and list responses from Claude
        if isinstance(questions, list):
            # If questions is a list, extract the text content
            question_text = questions[0].text if hasattr(questions[0], 'text') else str(questions[0])
        else:
            question_text = str(questions)
            
        # Split the questions string into individual questions
        # Remove any leading numbers and whitespace
        question_list = []
        for line in question_text.split('\n'):
            line = line.strip()
            if line and any(char.isdigit() for char in line[:2]):  # Check if line starts with a number
                # Split on first period after the number
                _, question = line.split('. ', 1)
                question_list.append(question.strip())
        
        print("\nPlease answer each question:")
        for i, question in enumerate(question_list, 1):
            print(f"\nQuestion {i}: {question}")
            answer = input("Your answer: ").strip()
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().isoformat()
            })
        
        # Load existing QA pairs if file exists
        try:
            with open(self.qa_file, 'r') as f:
                existing_qa = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_qa = []
        
        # Append new QA pairs and save
        existing_qa.extend(qa_pairs)
        with open(self.qa_file, 'w') as f:
            json.dump(existing_qa, f, indent=2)
        
        return qa_pairs

    def generate_updated_image_prompt(self, interpretation, qa_pairs):
        """Generate a new image prompt based on interpretation and Q&A pairs"""
        try:
            # Create a combined context from interpretation and Q&A
            qa_context = "\n".join([
                f"Q: {qa['question']}\nA: {qa['answer']}" 
                for qa in qa_pairs
            ])
            
            response = anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": f"""Based on this interpretation and the user's answers to follow-up questions, create a detailed visual description for image generation.
                    
                    Original interpretation: {interpretation}
                    
                    Q&A Context:
                    {qa_context}
                    
                    Create a vivid and specific visual description under 75 words that incorporates both the interpretation and the personal insights from the Q&A. Focus on visual elements that would make a compelling image."""
                }]
            )
            return str(response.content)
        except Exception as e:
            return f"Error generating updated image prompt: {str(e)}"

def main():
    interpreter = MessageInterpreter()
    
    # Get input message from user
    print("Please enter your message (press Enter twice to finish):")
    message_lines = []
    while True:
        line = input()
        if line:
            message_lines.append(line)
        else:
            break
    message = '\n'.join(message_lines)
    
    # Generate interpretation
    print("\nGenerating interpretation...")
    interpretation = interpreter.interpret_message(message)
    print("\nInterpretation:")
    print(interpretation)
    
    # Generate image prompt
    print("\nGenerating image prompt...")
    image_prompt = interpreter.generate_image_prompt(interpretation)
    print("\nImage Prompt:")
    print(image_prompt)
    
    # Generate image
    print("\nGenerating image using Stable Diffusion...")
    image_path = interpreter.generate_image(image_prompt)
    print(f"\nImage saved to: {image_path}")
    
    # Try to display the image if running in a compatible environment
    try:
        image = Image.open(image_path)
        image.show()
    except Exception as e:
        print(f"Could not display image directly: {str(e)}")
    
    # Generate questions
    print("\nGenerating questions...")
    questions = interpreter.generate_questions(message)
    print("\nQuestions:")
    print(questions)
    
    # Process questions and get answers
    qa_pairs = interpreter.process_questions_and_answers(questions)
    
    # Display stored QA pairs
    print("\nStored Question-Answer Pairs:")
    for i, qa in enumerate(qa_pairs, 1):
        print(f"\n{i}. Question: {qa['question']}")
        print(f"   Answer: {qa['answer']}")
    
    # Generate updated image prompt based on interpretation and Q&A
    print("\nGenerating updated image prompt based on interpretation and your answers...")
    updated_image_prompt = interpreter.generate_updated_image_prompt(interpretation, qa_pairs)
    print("\nUpdated Image Prompt:")
    print(updated_image_prompt)

    # Generate image
    print("\nReGenerating image using Stable Diffusion...")
    updated_image_path = interpreter.generate_image(updated_image_prompt)
    print(f"\nImage saved to: {updated_image_path}")
    
    # Try to display the image if running in a compatible environment
    try:
        image = Image.open(updated_image_path)
        image.show()
    except Exception as e:
        print(f"Could not display image directly: {str(e)}")

if __name__ == "__main__":
    main() 