from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

app = Flask(__name__)

# Load the tokenizer and model from Hugging Face
model_name = "Amna100/Fold-4"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Initialize the pipeline
token_classifier = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="first")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        results = token_classifier(text)

        # Sort the entities by their starting position to handle overlapping replacements
        sorted_results = sorted(results, key=lambda x: x['start'])

        # Create a list to build the final result text
        result_text_parts = []
        last_idx = 0

        for entity in sorted_results:
            # Add text before the current entity
            result_text_parts.append(text[last_idx:entity['start']])
            # Highlight the entity in the text with appropriate class
            if entity['entity_group'] == 'Drug':
                result_text_parts.append(f"<span class='highlight-drug'>{entity['word']}<span class='label'>Drug</span></span>")
            elif entity['entity_group'] == 'ADE':
                result_text_parts.append(f"<span class='highlight-ade'>{entity['word']}<span class='label'>ADE</span></span>")
            # Update the last index to the end of the current entity
            last_idx = entity['end']

        # Add any remaining text after the last entity
        result_text_parts.append(text[last_idx:])

        # Join all parts into the final result text
        result_text = ''.join(result_text_parts)

        return jsonify({'result': result_text})

if __name__ == "__main__":
    app.run(debug=True)
