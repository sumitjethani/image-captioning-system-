import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle

IMAGE_SIZE = 224
ENCODER_DIM = 2048
EMBED_DIM = 256
DECODER_DIM = 512
ATTENTION_DIM = 256
DROPOUT = 0.5
MAX_LEN = 80

device = torch.device("cpu")

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def tokenizer_eng(self, text):
        return text.split()

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"] 
            for token in tokenized_text
        ]

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att_score = self.full_att(self.relu(att1 + att2.unsqueeze(1)))
        alpha = self.softmax(att_score)
        attention_weighted_encoding = (encoder_out * alpha).sum(dim=1)
        return attention_weighted_encoding, alpha

class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=ENCODER_DIM, dropout=DROPOUT):
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

def greedy_search(model, image_features, vocab, max_len=MAX_LEN):
    model.eval()
    caption = []
    with torch.no_grad():
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)
            
        image_features = image_features.to(device)
        h, c = model.init_hidden_state(image_features)
        
        start_word = '<start>' if '<start>' in vocab.stoi else '<SOS>'
        end_word = '<end>' if '<end>' in vocab.stoi else '<EOS>'
        
        word_idx = vocab.stoi[start_word]
        
        for _ in range(max_len):
            embed = model.embedding(torch.tensor([word_idx]).to(device))
            att_weight, alpha = model.attention(image_features, h)
            gate = torch.sigmoid(model.f_beta(h))
            att_weight = gate * att_weight
            
            lstm_input = torch.cat((embed, att_weight), dim=1)
            h, c = model.lstm_cell(lstm_input, (h, c))
            
            output = model.fc(h)
            predicted_word_idx = output.argmax(dim=1).item()
            
            word_idx = predicted_word_idx
            word = vocab.itos[word_idx]
            
            if word == end_word:
                break
            if word not in [start_word, '<pad>', '<PAD>', '<unk>', '<UNK>']:
                caption.append(word)
    return caption

def beam_search(model, image_features, vocab, beam_width=5, max_len=MAX_LEN):
    model.eval()
    with torch.no_grad():
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)
        
        image_features = image_features.to(device)
        h, c = model.init_hidden_state(image_features)
        
        start_word = '<start>' if '<start>' in vocab.stoi else '<SOS>'
        end_word = '<end>' if '<end>' in vocab.stoi else '<EOS>'
        
        start_token = vocab.stoi[start_word]
        end_token = vocab.stoi[end_word]
        
        beams = [([start_token], 0.0, h, c)]
        completed_sequences = []
        
        for step in range(max_len):
            all_candidates = []
            for sequence, score, h_state, c_state in beams:
                if sequence[-1] == end_token:
                    completed_sequences.append((sequence, score))
                    continue
                
                last_word = sequence[-1]
                embed = model.embedding(torch.tensor([last_word]).to(device))
                att_weight, alpha = model.attention(image_features, h_state)
                gate = torch.sigmoid(model.f_beta(h_state))
                att_weight = gate * att_weight
                
                lstm_input = torch.cat((embed, att_weight), dim=1)
                h_new, c_new = model.lstm_cell(lstm_input, (h_state, c_state))
                output = model.fc(h_new)
                
                log_probs = torch.log_softmax(output, dim=1)
                top_log_probs, top_indices = log_probs.topk(beam_width, dim=1)
                
                for i in range(beam_width):
                    next_word = top_indices[0, i].item()
                    next_log_prob = top_log_probs[0, i].item()
                    new_sequence = sequence + [next_word]
                    new_score = score + next_log_prob
                    all_candidates.append((new_sequence, new_score, h_new, c_new))
            
            if not all_candidates:
                break
            
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]
            
            if all(seq[-1] == end_token for seq, _, _, _ in beams):
                completed_sequences.extend([(seq, score) for seq, score, _, _ in beams])
                break
        
        for seq, score, _, _ in beams:
            if (seq, score) not in completed_sequences:
                completed_sequences.append((seq, score))
        
        if completed_sequences:
            completed_sequences.sort(key=lambda x: x[1], reverse=True)
            best_sequence, _ = completed_sequences[0]
        else:
            best_sequence = beams[0][0]
        
        caption = []
        for word_idx in best_sequence:
            word = vocab.itos[word_idx]
            if word not in [start_word, end_word, '<pad>', '<PAD>', '<unk>', '<UNK>']:
                caption.append(word)
        return caption

def load_models():
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    modules = list(resnet.children())[:-1]
    feature_extractor = nn.Sequential(*modules).to(device)
    feature_extractor.eval()

    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # FIX 1: Set weights_only=False to allow loading the custom Vocabulary class from the .pth file
    checkpoint = torch.load('best_model_val_loss.pth', map_location=device, weights_only=False)
    
    decoder = DecoderWithAttention(
        attention_dim=ATTENTION_DIM,
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        vocab_size=len(vocab),
        encoder_dim=ENCODER_DIM,
        dropout=DROPOUT
    ).to(device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        decoder.load_state_dict(checkpoint['model_state_dict'])
    else:
        decoder.load_state_dict(checkpoint)
        
    decoder.eval()
    return feature_extractor, decoder, vocab

try:
    feature_extractor, decoder, vocab = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    error_message = f"Error loading models: {e}. Make sure vocab.pkl and best_model_val_loss.pth are uploaded."

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def generate_caption(image, decoding_method, beam_width):
    if not models_loaded:
        return error_message
    if image is None:
        return "Please upload an image first."
        
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = feature_extractor(img_tensor)
        features = features.view(features.size(0), -1)
    
    if decoding_method == "Greedy Search":
        caption_words = greedy_search(decoder, features, vocab)
    else:
        caption_words = beam_search(decoder, features, vocab, beam_width=int(beam_width))
        
    return " ".join(caption_words).capitalize()

with gr.Blocks(title="Image Caption Generator") as demo:
    gr.Markdown("# 📝 Seq2Seq Image Caption Generator")
    gr.Markdown("Upload an image below.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image")
            
            with gr.Accordion("Model Settings", open=True):
                decoding_method = gr.Radio(
                    choices=["Greedy Search", "Beam Search"], 
                    value="Beam Search", 
                    label="Decoding Strategy"
                )
                beam_width = gr.Slider(
                    minimum=2, maximum=10, step=1, value=5, 
                    label="Beam Width (Only applies to Beam Search)"
                )
                
            submit_btn = gr.Button("Generate Caption", variant="primary")
            
        with gr.Column(scale=1):
            caption_output = gr.Textbox(label="Generated Caption", lines=4)

    submit_btn.click(
        fn=generate_caption, 
        inputs=[image_input, decoding_method, beam_width], 
        outputs=caption_output
    )

if __name__ == "__main__":
    
    demo.launch(theme=gr.themes.Soft())