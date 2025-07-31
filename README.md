# Medical Symptom Classification: DistilBERT vs GPT-4.1

A comparison study showing that smaller, specialized models can outperform large language models on focused classification tasks.

## Project Overview

This project compares two different approaches for medical symptom classification:
- **DistilBERT** (fine-tuned encoder model)
- **GPT-4.1** (large language model with zero-shot prompting)

The task is simple: given a description of symptoms, predict which medical condition it represents from a list of 40+ possible diagnoses.

## Key Results

| Model | Macro F1 Score | Weighted F1 Score | Performance |
|-------|----------------|-------------------|-------------|
| **DistilBERT (fine-tuned)** | **0.94** | **0.96** | ✅ Excellent |
| **GPT-4.1 (zero-shot)** | 0.54 | 0.58 | ❌ Poor |

**Winner: DistilBERT** 🏆

## What This Project Does

1. **Data Processing**: Loads a medical symptom dataset and cleans it
2. **Model Training**: Fine-tunes DistilBERT on symptom-to-condition mapping
3. **LLM Testing**: Tests GPT-4.1 using carefully crafted prompts
4. **Comparison**: Evaluates both approaches using standard metrics

## Project Structure

```
├── Dataset loading and preprocessing
├── DistilBERT fine-tuning pipeline
├── GPT-4.1 zero-shot evaluation
├── Results comparison and analysis
└── Data visualization
```

## How to Run

### Prerequisites
- Python 3.8+
- Required packages: `transformers`, `torch`, `sklearn`, `pandas`, `openai`
- OpenAI API key (for GPT-4.1 comparison)

### Steps
1. **Install dependencies**:
   ```bash
   pip install transformers torch scikit-learn pandas openai python-dotenv datasets
   ```

2. **Set up environment**:
   - Create a `.env` file
   - Add your OpenAI API key: `OPENAI_KEY=your_key_here`

3. **Run the notebook**:
   - Execute cells sequentially
   - The code will automatically download the medical dataset

## The Medical Classification Task

**Input**: Text describing symptoms
> "I have been experiencing severe headaches with nausea and sensitivity to light"

**Output**: One medical condition from the list
> "migraine"

**Available Conditions**: 40+ medical conditions including:
- Drug reaction, Allergy, Bronchial asthma
- Malaria, Hepatitis variants, Diabetes
- Heart attack, Migraine, Tuberculosis
- And many more...

## Why DistilBERT Won

### DistilBERT Advantages:
- **Purpose-built**: Encoder models excel at classification tasks
- **Fine-tuned**: Trained specifically on this medical data
- **Consistent**: Always outputs from the correct label set
- **Efficient**: Smaller, faster, cheaper to run
- **Reliable**: Predictable performance on structured tasks

### GPT-4.1 Challenges:
- **Generalist**: Not optimized for this specific task
- **Inconsistent**: Sometimes generated labels not in the target list
- **Expensive**: Higher API costs per prediction. It cost almost $9 dollars to run this task.
- **Overkill**: Too powerful for this simple classification task

## Performance Breakdown

### DistilBERT Results:
- **Training**: Converged quickly with early stopping
- **Accuracy**: 96% weighted F1 score
- **Consistency**: Always predicted valid conditions
- **Speed**: Fast inference time

### GPT-4.1 Results:
- **Accuracy**: 58% weighted F1 score
- **Issues**: Inconsistent label formatting
- **Cost**: Significantly more expensive per prediction
- **Reliability**: Variable performance across conditions

## Key Takeaways

1. **Match the tool to the task**: Use encoder models for classification
2. **Bigger isn't always better**: DistilBERT (66M parameters) beat GPT-4.1 (1.7T+ parameters)
3. **Fine-tuning matters**: Specialized training beats general prompting
4. **Cost efficiency**: Smaller models can be much more economical

## Technical Details

- **Model**: DistilBERT-base-uncased
- **Training**: 3 epochs with early stopping
- **Batch size**: 16
- **Learning rate**: 2e-5
- **Max sequence length**: 512 tokens
- **Data split**: 80% training, 20% testing

## When to Use Each Approach

### Use Encoder Models (BERT, DistilBERT, RoBERTa) for:
- Text classification
- Sentiment analysis
- Spam detection
- Entity recognition
- Semantic similarity

### Use Decoder Models (GPT, Claude, Gemini) for:
- Text generation
- Summarisation
- Conversational AI
- Creative writing
- Open-ended reasoning

## Files Generated (and shared here)

- `gpt4_1_prediction_train.csv`: GPT-4.1 training predictions
- `gpt4_1_prediction_test.csv`: GPT-4.1 test predictions


## Contributing

Feel free to:
- Try different encoder models (RoBERTa, BERT-large)
- Experiment with different prompting strategies for GPT
- Add more evaluation metrics
- Test on other medical datasets
