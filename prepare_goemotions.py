import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm

# 0: 'грусть' (sadness)
# 1: 'радость' (joy)
# 2: 'любовь' (love)
# 3: 'злость' (anger)
# 4: 'страх' (fear)
# 5: 'удивление' (surprise)

emotion_mapping = {
    'sadness': 0,         
    'disappointment': 0,    
    'grief': 0,            
    
    'joy': 1,               
    'amusement': 1,         
    'approval': 1,         
    'excitement': 1,        
    'gratitude': 1,         
    'optimism': 1,          
    'relief': 1,            
    'pride': 1,             
    'admiration': 1,        
    
    'love': 2,              
    'caring': 2,            
    'desire': 2,           
    
    'anger': 3,             
    'annoyance': 3,         
    'disgust': 3,           
    'disapproval': 3,      
    
    'fear': 4,              
    'nervousness': 4,       
    'remorse': 4,           
    'embarrassment': 4,     
    
    'surprise': 5,          
    'confusion': 5,         
    'curiosity': 5,         
    'realization': 5,       
}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def map_to_six_emotions(row):
    if 'neutral' in row and row['neutral'] == 1:
        return None
    
    emotion_votes = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    for emotion, mapping in emotion_mapping.items():
        if emotion in row and row[emotion] == 1:
            emotion_votes[mapping] += 1
    
    if sum(emotion_votes.values()) == 0:
        return None
    
    max_votes = max(emotion_votes.values())
    

    max_emotions = [e for e, v in emotion_votes.items() if v == max_votes]
    if len(max_emotions) == 1:
        return max_emotions[0]
    else:
        priority_order = [1, 3, 4, 0, 2, 5]
        for emotion in priority_order:
            if emotion in max_emotions:
                return emotion
        return max_emotions[0]

def process_goemotions_files():
    data_frames = []
    
    for file in os.listdir('datasets/goemotions'):
        if file.endswith('.csv') and file.startswith('goemotions_'):
            filepath = os.path.join('datasets/goemotions', file)
            
            chunks = pd.read_csv(filepath, chunksize=10000)
            for chunk in chunks:
                data_frames.append(chunk)
    
    combined_df = pd.concat(data_frames, ignore_index=True)

    combined_df['text'] = combined_df['text'].apply(clean_text)
    
    combined_df = combined_df[combined_df['text'].str.len() > 0]
    
    results = []
    for _, row in tqdm(combined_df.iterrows(), total=len(combined_df), desc="Отображение эмоций"):
        emotion = map_to_six_emotions(row)
        if emotion is not None:
            results.append({
                'text': row['text'],
                'label': emotion
            })
    
    new_df = pd.DataFrame(results)
    
    class_distribution = new_df['label'].value_counts()
    for class_id, count in class_distribution.items():
        emotion_name = {
            0: 'sad',
            1: 'joi',
            2: 'love',
            3: 'angry',
            4: 'fear',
            5: 'surprise'
        }[class_id]
        percentage = count / len(new_df) * 100
        print(f"{class_id} ({emotion_name}): {count} ({percentage:.2f}%)")
    
    min_class_count = min(class_distribution)
    target_count = max(min_class_count, 1000)
    balanced_df = pd.DataFrame()
    
    for class_id in range(6):
        class_samples = new_df[new_df['label'] == class_id]
        if len(class_samples) > target_count:
            class_samples = class_samples.sample(target_count, random_state=42)
        balanced_df = pd.concat([balanced_df, class_samples])
    
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    balanced_df.to_csv('datasets/goemotions_processed.csv', index=False)
    
    return balanced_df, new_df

if __name__ == "__main__":
    balanced_df, full_df = process_goemotions_files()
    