from training.train import run_training
from inference.generate_text import load_model_and_generate

if __name__ == "__main__":
    # Step 1: Train Model
    run_training()

    # Step 2: Generate Text
    seed = "Who is Amaninder"
    result = load_model_and_generate(seed)
    print("\n📝 Generated Text:")
    print(result)
