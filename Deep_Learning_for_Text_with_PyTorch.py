# Import the necessary functions
from torchtext.data.utils import get_tokenizer
from nltk.probability import FreqDist

text = "In the city of Dataville, a data analyst named Alex explores hidden insights within vast data. With determination, Alex uncovers patterns, cleanses the data, and unlocks innovation. Join this adventure to unleash the power of data-driven decisions."

# Initialize the tokenizer and tokenize the text
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer(text)

threshold = 1
# Remove rare words and print common tokens
freq_dist = FreqDist(tokens)
common_tokens = [token for token in tokens if freq_dist[token] > threshold]
print(common_tokens)

# Initialize and tokenize the text
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer(text)

# Remove any stopwords
stop_words = set(stopwords.words("english"))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

# Perform stemming on the filtered tokens
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
print(stemmed_tokens)

genres = ['Fiction','Non-fiction','Biography', 'Children','Mystery']

# Define the size of the vocabulary
vocab_size = len(genres)

# Create one-hot vectors
one_hot_vectors = torch.eye(vocab_size)

# Create a dictionary mapping genres to their one-hot vectors
one_hot_dict = {genre: one_hot_vectors[i] for i, genre in enumerate(genres)}

for genre, vector in one_hot_dict.items():
    print(f'{genre}: {vector.numpy()}')

# Import from sklearn
from sklearn.feature_extraction.text import CountVectorizer

titles = ['The Great Gatsby','To Kill a Mockingbird','1984','The Catcher in the Rye','The Hobbit', 'Great Expectations']

# Initialize Bag-of-words with the list of book titles
vectorizer = CountVectorizer()
bow_encoded_titles = vectorizer.fit_transform(titles)

# Extract and print the first five features
print(vectorizer.get_feature_names_out()[:5])
print(bow_encoded_titles.toarray()[0, :5])

# Importing TF-IDF from sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF encoding vectorizer
vectorizer = TfidfVectorizer() 
tfidf_encoded_descriptions = vectorizer.fit_transform(descriptions)

# Extract and print the first five features
print(vectorizer.get_feature_names_out()[:5])
print(tfidf_encoded_descriptions.toarray()[0, :5])

# Create a list of stopwords
stop_words = set(stopwords.words("english"))

# Initialize the tokenizer and stemmer
tokenizer = get_tokenizer("basic_english")
stemmer = PorterStemmer() 

# Complete the function to preprocess sentences
def preprocess_sentences(sentences):
    processed_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        tokens = tokenizer(sentence)
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [stemmer.stem(token) for token in tokens]
        processed_sentences.append(' '.join(tokens))
    return processed_sentences

processed_shakespeare = preprocess_sentences(shakespeare)
print(processed_shakespeare[:5]) 

# Define your Dataset class
class ShakespeareDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# Complete the encoding function
def encode_sentences(sentences):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    return X.toarray(), vectorizer
    
# Complete the text processing pipeline
def text_processing_pipeline(sentences):
    processed_sentences = preprocess_sentences(sentences)
    encoded_sentences, vectorizer = encode_sentences(processed_sentences)
    dataset = ShakespeareDataset(encoded_sentences)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    return dataloader, vectorizer

dataloader, vectorizer = text_processing_pipeline(processed_shakespeare)

# Print the vectorizer's feature names and the first 10 components of the first item
print(vectorizer.get_feature_names_out()[:10]) 
print(next(iter(dataloader))[0, :10])

# Map a unique index to each word
words = ["This", "book", "was", "fantastic", "I", "really", "love", "science", "fiction", "but", "the", "protagonist", "was", "rude", "sometimes"]
word_to_idx = {word: i for i, word in enumerate(words)}

# Convert word_to_idx to a tensor
inputs = torch.LongTensor([word_to_idx[w] for w in words])

# Initialize embedding layer with ten dimensions
embedding = nn.Embedding(num_embeddings=len(words), embedding_dim=10)

# Pass the tensor to the embedding layer
output = embedding(inputs)
print(output)

class TextClassificationCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(TextClassificationCNN, self).__init__()
        # Initialize the embedding layer 
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(embed_dim, 2)
    def forward(self, text):
        embedded = self.embedding(text).permute(0, 2, 1)
        # Pass the embedded text through the convolutional layer and apply a ReLU
        conved = F.relu(self.conv(embedded))
        conved = conved.mean(dim=2) 
        return self.fc(conved)

# Define the loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(10):
    for sentence, label in data:     
        # Clear the gradients
        model.zero_grad()
        sentence = torch.LongTensor([word_to_ix.get(w, 0) for w in sentence]).unsqueeze(0) 
        label = torch.LongTensor([int(label)])
        outputs = model(sentence)
        loss = criterion(outputs, label)
        loss.backward()
        # Update the parameters
        optimizer.step()
print('Training complete!')

book_reviews = [
    "I love this book".split(),
    "I do not like this book".split()
]
for review in book_reviews:
    # Convert the review words into tensor form
    input_tensor = torch.tensor([word_to_ix[w] for w in review], dtype=torch.long).unsqueeze(0) 
    # Get the model's output
    outputs = model(input_tensor)
    # Find the index of the most likely sentiment category
    _, predicted_label = torch.max(outputs.data, 1)
    # Convert the predicted label into a sentiment string
    sentiment = "Positive" if predicted_label.item() else "Negative"
    print(f"Book Review: {' '.join(review)}")
    print(f"Sentiment: {sentiment}\n")

# Complete the RNN class
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

# Initialize the model
rnn_model = RNNModel(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn_model.parameters(), lr=0.01)

# Train the model for ten epochs and zero the gradients
for epoch in range(10): 
    optimizer.zero_grad()
    outputs = rnn_model(X_train_seq)
    loss = criterion(outputs, y_train_seq)
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

# Initialize the LSTM and the output layer with parameters
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)       
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

# Initialize model with required parameters
lstm_model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.01)

# Train the model by passing the correct parameters and zeroing the gradient
for epoch in range(10): 
    optimizer.zero_grad()
    outputs = lstm_model(X_train_seq)
    loss = criterion(outputs, y_train_seq)
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

# Complete the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

# Initialize the model
gru_model = GRUModel(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(gru_model.parameters(), lr=0.01)

# Train the model and backpropagate the loss after initialization
for epoch in range(15):
    optimizer.zero_grad()
    outputs = gru_model(X_train_seq)
    loss = criterion(outputs, y_train_seq)
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

# Create an instance of the metrics
accuracy = Accuracy(task="multiclass", num_classes=3)
precision = Precision(task="multiclass", num_classes=3)
recall = Recall(task="multiclass", num_classes=3)
f1 = F1Score(task="multiclass", num_classes=3)

# Generate the predictions
outputs = rnn_model(X_test_seq)
_, predicted = torch.max(outputs, 1)

# Calculate the metrics
accuracy_score = accuracy(predicted, y_test_seq)
precision_score = precision(predicted, y_test_seq)
recall_score = recall(predicted, y_test_seq)
f1_score = f1(predicted, y_test_seq)
print("RNN Model - Accuracy: {}, Precision: {}, Recall: {}, F1 Score: {}".format(accuracy_score, precision_score, recall_score, f1_score))

# Create an instance of the metrics
accuracy = Accuracy(task="multiclass", num_classes=3)
precision = Precision(task="multiclass", num_classes=3)
recall = Recall(task="multiclass", num_classes=3)
f1 = F1Score(task="multiclass", num_classes=3)

# Calculate metrics for the LSTM model
accuracy_1 = accuracy(y_pred_lstm, y_test)
precision_1 = precision(y_pred_lstm, y_test)
recall_1 = recall(y_pred_lstm, y_test)
f1_1 = f1(y_pred_lstm, y_test)
print("LSTM Model - Accuracy: {}, Precision: {}, Recall: {}, F1 Score: {}".format(accuracy_1, precision_1, recall_1, f1_1))

# Calculate metrics for the GRU model
accuracy_2 = accuracy(y_pred_lstm, y_test)
precision_2 = precision(y_pred_lstm, y_test)
recall_2 = recall(y_pred_lstm, y_test)
f1_2 = f1(y_pred_lstm, y_test)
print("GRU Model - Accuracy: {}, Precision: {}, Recall: {}, F1 Score: {}".format(accuracy_2, precision_2, recall_2, f1_2))

# Include an RNN layer and linear layer in RNNmodel class
class RNNmodel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNmodel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
      h0 = torch.zeros(1, x.size(0), self.hidden_size)
      out, _ = self.rnn(x, h0)  
      out = self.fc(out[:, -1, :])  
      return out

# Instantiate the RNN model
model = RNNmodel(1, 16, 1)

# Instantiate the loss function
criterion = nn.CrossEntropyLoss()
# Instantiate the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(100):
    model.train()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/100, Loss: {loss.item()}')

# Test the model
model.eval()
test_input = char_to_ix['r']
test_input = nn.functional.one_hot(torch.tensor(test_input).view(-1, 1), num_classes=len(chars)).float()
predicted_output = model(test_input)
predicted_char_ix = torch.argmax(predicted_output, 1).item()
print(f"Test Input: 'r', Predicted Output: '{ix_to_char[predicted_char_ix]}'")

# Define the generator class
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(seq_length, seq_length), nn.Sigmoid())
    def forward(self, x):
        return self.model(x)

# Define the discriminator networks
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(seq_length, 1), nn.Sigmoid())
    def forward(self, x):
        return self.model(x)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer_gen = torch.optim.Adam(generator.parameters(), lr=0.001)
optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for real_data in data:
      	# Unsqueezing real_data and prevent gradient recalculations
        real_data = real_data.unsqueeze(0)
        noise = torch.rand((1, seq_length))
        fake_data = generator(noise)
        disc_real = discriminator(real_data)
        disc_fake = discriminator(fake_data.detach())
        loss_disc = criterion(disc_real, torch.ones_like(disc_real)) + criterion(disc_fake, torch.zeros_like(disc_fake))
        optimizer_disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Initialize the pre-trained model
model = GPT2LMHeadModel.from_pretrained('gpt2')

seed_text = "Once upon a time"

# Encode the seed text to get input tensors
input_ids = tokenizer.encode(seed_text, return_tensors='pt')

# Generate text from the model
output = model.generate(input_ids, max_length=100, temperature=0.7, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id) 

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)

# Initalize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

input_prompt = "translate English to French: 'Hello, how are you?'"

# Encode the input prompt using the tokenizer
input_ids = tokenizer.encode(input_prompt, return_tensors="pt")

# Generate the translated ouput
output = model.generate(input_ids, max_length=50)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:",generated_text)

reference_text = "Once upon a time, there was a little girl who lived in a village near the forest."
generated_text = "Once upon a time, the world was a place of great beauty and great danger. The world of the gods was the place where the great gods were born, and where they were to live."

# Initialize BLEU and ROUGE scorers
bleu = BLEUScore()
rouge = ROUGEScore()

# Calculate the BLEU and ROUGE scores
bleu_score = bleu([generated_text], [[reference_text]])
rouge_score = rouge([generated_text], [[reference_text]])

# Print the BLEU and ROUGE scores
print("BLEU Score:", bleu_score.item())
print("ROUGE Score:", rouge_score)

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize your data and return PyTorch tensors
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=32)
inputs["labels"] = torch.tensor(labels)

# Setup the optimizer using model parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
model.train()
for epoch in range(2):
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

text = "I had an awesome day!"

# Tokenize the text and return PyTorch tensors
input_eval = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=32)
outputs_eval = model(**input_eval)

# Convert the output logits to probabilities
predictions = torch.nn.functional.softmax(outputs_eval.logits, dim=-1)

# Display the sentiments
predicted_label = "positive" if torch.argmax(predictions) > 0 else "negative"
print(f"Text: {text}\nSentiment: {predicted_label}")

class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, dropout):
        super(TransformerEncoder, self).__init__()
        # Initialize the encoder 
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=heads),
            num_layers=num_layers)
        # Define the fully connected layer
        self.fc = nn.Linear(embed_size, 2)

    def forward(self, x):
        # Pass the input through the transformer encoder 
        x = self.encoder(x)
        x = x.mean(dim=1) 
        return self.fc(x)

model = TransformerEncoder(embed_size=512, heads=8, num_layers=3, dropout=0.5)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):  
    for sentence, label in zip(train_sentences, train_labels):
        # Split the sentences into tokens and stack the embeddings
        tokens = sentence.split()
        data = torch.stack([token_embeddings[token] for token in tokens], dim=1)
        output = model(data)
        loss = criterion(output, torch.tensor([label]))
        # Zero the gradients and perform a backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

def predict(sentence):
    model.eval()
    # Deactivate the gradient computations and get the sentiment prediction.
    with torch.no_grad():
        tokens = sentence.split()
        data = torch.stack([token_embeddings.get(token, torch.rand((1, 512))) for token in tokens], dim=1)
        output = model(data)
        predicted = torch.argmax(output, dim=1)
        return "Positive" if predicted.item() == 1 else "Negative"

sample_sentence = "This product can be better"
print(f"'{sample_sentence}' is {predict(sample_sentence)}")

class RNNWithAttentionModel(nn.Module):
    def __init__(self):
        super(RNNWithAttentionModel, self).__init__()
        # Create an embedding layer for the vocabulary
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        # Apply a linear transformation to get the attention scores
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        x = self.embeddings(x)
        out, _ = self.rnn(x)
        #  Get the attention weights
        attn_weights = torch.nn.functional.softmax(self.attention(out).squeeze(2), dim=1)
        # Compute the context vector 
        context = torch.sum(attn_weights.unsqueeze(2) * out, dim=1)
        out = self.fc(context)
        return out
      
attention_model = RNNWithAttentionModel()
optimizer = torch.optim.Adam(attention_model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
print("Model Instantiated")

for epoch in range(epochs):
    attention_model.train()
    optimizer.zero_grad()
    padded_inputs = pad_sequences(inputs)
    outputs = attention_model(padded_inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

for input_seq, target in zip(input_data, target_data):
    input_test = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0)
   	
    #  Set the RNN model to evaluation mode
    rnn_model.eval()
    # Get the RNN output by passing the appropriate input 
    rnn_output = rnn_model(input_test)
    # Extract the word with the highest prediction score 
    rnn_prediction = ix_to_word[torch.argmax(rnn_output).item()]

    attention_model.eval()
    attention_output = attention_model(input_test)
    # Extract the word with the highest prediction score
    attention_prediction = ix_to_word[torch.argmax(attention_output).item()]

    print(f"\nInput: {' '.join([ix_to_word[ix] for ix in input_seq])}")
    print(f"Target: {ix_to_word[target]}")
    print(f"RNN prediction: {rnn_prediction}")
    print(f"RNN with Attention prediction: {attention_prediction}")

