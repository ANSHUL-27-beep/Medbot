from medical_bot import MedicalBot

def main():
    # Initialize the medical bot with the PDF file
    pdf_path = "Gale Encyclopedia of Medicine. Vol. 2. 2nd ed.pdf"
    bot = MedicalBot(pdf_path)
    
    print("Medical Bot initialized successfully!")
    print("You can now ask medical questions. Type 'exit' to quit.")
    
    # Simple command-line interface
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'exit':
            break
        
        # Get answer from the bot
        answer = bot.query(question)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
