from medical_bot import MedicalBot

def main():
    # Initialize the medical bot with both PDF files
    pdf_paths = [
        "Gale Encyclopedia of Medicine. Vol. 2. 2nd ed.pdf",
        "Gale Encyclopedia of Medicine. Vol. 1. 2nd ed (1).pdf"
    ]
    bot = MedicalBot(pdf_paths)
    
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
