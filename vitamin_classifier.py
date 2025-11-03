import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import json
import os
import numpy as np
import timm
import torch.nn as nn

# Define the exact same model architecture
class ConvNeXtV2Classifier(nn.Module):
    def __init__(self, num_classes=5, model_name='convnextv2_base', pretrained=False):
        super(ConvNeXtV2Classifier, self).__init__()
        
        # Load pre-trained ConvNeXt V2
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get the number of features from the backbone
        num_features = self.backbone.num_features
        
        # Add custom classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(num_features),
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone.forward_features(x)
        output = self.classifier(features)
        return output

class VitaminDeficiencyClassifier:
    def __init__(self, model_weights_path, class_info_path, preprocessing_info_path, model_config_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model config
        with open(model_config_path, 'r') as f:
            config = json.load(f)
        
        # Recreate the exact same model architecture
        self.model = ConvNeXtV2Classifier(
            num_classes=config['num_classes'],
            model_name=config['model_name']
        )
        
        # Load only the weights
        try:
            state_dict = torch.load(model_weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("✅ Model weights loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model weights: {e}")
            raise e
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load class information
        with open(class_info_path, 'r') as f:
            class_info = json.load(f)
        self.classes = class_info['classes']
        self.decoding_classes = class_info['decoding_classes']
        
        # Load preprocessing information
        with open(preprocessing_info_path, 'r') as f:
            preprocessing_info = json.load(f)
        self.mean = preprocessing_info['mean']
        self.std = preprocessing_info['std']
        self.input_size = tuple(preprocessing_info['input_size'])
        
        # Create transform
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # Vitamin deficiency information mapping
        self.vitamin_info = {
            'Light Diseases and Disorders of Pigmentation': {
                'deficiencies': ['Vitamin D', 'Vitamin B12', 'Vitamin C', 'Vitamin A', 'Vitamin E'],
                'foods': {
                    'Vitamin D': ['Salmon', 'Mushrooms', 'Egg yolks', 'Fortified milk', 'Tuna'],
                    'Vitamin B12': ['Clams', 'Beef liver', 'Fortified cereals', 'Sardines', 'Tuna'],
                    'Vitamin C': ['Oranges', 'Strawberries', 'Bell peppers', 'Broccoli', 'Kiwi'],
                    'Vitamin A': ['Sweet potatoes', 'Carrots', 'Spinach', 'Kale', 'Liver'],
                    'Vitamin E': ['Almonds', 'Sunflower seeds', 'Avocado', 'Spinach', 'Olive oil']
                },
                'description': 'Conditions affecting skin pigmentation and light response'
            },
            'Acne and Rosacea Photos': {
                'deficiencies': ['Vitamin D', 'Vitamin A', 'Zinc', 'Vitamin B2'],
                'foods': {
                    'Vitamin D': ['Salmon', 'Sardines', 'Fortified milk', 'Egg yolks', 'Mushrooms'],
                    'Vitamin A': ['Sweet potatoes', 'Carrots', 'Spinach', 'Red bell peppers', 'Liver'],
                    'Zinc': ['Oysters', 'Beef', 'Pumpkin seeds', 'Chickpeas', 'Cashews'],
                    'Vitamin B2': ['Almonds', 'Yogurt', 'Mushrooms', 'Spinach', 'Eggs']
                },
                'description': 'Inflammatory skin conditions including acne and rosacea'
            },
            'Poison Ivy Photos and other Contact Dermatitis': {
                'deficiencies': ['Vitamin B2', 'Vitamin B3', 'Vitamin B7', 'Vitamin C', 'Vitamin D'],
                'foods': {
                    'Vitamin B2': ['Almonds', 'Yogurt', 'Mushrooms', 'Spinach', 'Eggs'],
                    'Vitamin B3': ['Tuna', 'Chicken breast', 'Turkey', 'Peanuts', 'Mushrooms'],
                    'Vitamin B7': ['Eggs', 'Almonds', 'Sweet potatoes', 'Spinach', 'Salmon'],
                    'Vitamin C': ['Oranges', 'Strawberries', 'Bell peppers', 'Broccoli', 'Kiwi'],
                    'Vitamin D': ['Salmon', 'Mushrooms', 'Egg yolks', 'Fortified milk', 'Tuna']
                },
                'description': 'Inflammatory skin reactions from contact with irritants'
            },
            'Atopic Dermatitis Photos': {
                'deficiencies': ['Vitamin D', 'Vitamin B2', 'Vitamin B3', 'Vitamin B7', 'Vitamin C'],
                'foods': {
                    'Vitamin D': ['Salmon', 'Sardines', 'Fortified milk', 'Egg yolks', 'Mushrooms'],
                    'Vitamin B2': ['Almonds', 'Yogurt', 'Mushrooms', 'Spinach', 'Eggs'],
                    'Vitamin B3': ['Tuna', 'Chicken breast', 'Turkey', 'Peanuts', 'Mushrooms'],
                    'Vitamin B7': ['Eggs', 'Almonds', 'Sweet potatoes', 'Spinach', 'Salmon'],
                    'Vitamin C': ['Oranges', 'Strawberries', 'Bell peppers', 'Broccoli', 'Kiwi']
                },
                'description': 'Chronic inflammatory skin condition (eczema)'
            },
            'Hair Loss Photos Alopecia and other Hair Diseases': {
                'deficiencies': ['Vitamin D', 'Vitamin B12', 'Vitamin B7', 'Vitamin A', 'Vitamin E'],
                'foods': {
                    'Vitamin D': ['Salmon', 'Mushrooms', 'Egg yolks', 'Fortified milk', 'Tuna'],
                    'Vitamin B12': ['Clams', 'Beef liver', 'Fortified cereals', 'Sardines', 'Tuna'],
                    'Vitamin B7': ['Eggs', 'Almonds', 'Sweet potatoes', 'Spinach', 'Salmon'],
                    'Vitamin A': ['Sweet potatoes', 'Carrots', 'Spinach', 'Kale', 'Liver'],
                    'Vitamin E': ['Almonds', 'Sunflower seeds', 'Avocado', 'Spinach', 'Olive oil']
                },
                'description': 'Hair thinning, balding, and other hair-related conditions'
            }
        }
        
    def predict(self, image_path):
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Get results
            predicted_class_idx = predicted.item()
            confidence_score = confidence.item()
            predicted_class_name = self.decoding_classes[str(predicted_class_idx)]
            
            # Get vitamin information
            vitamin_data = self.vitamin_info.get(predicted_class_name, {
                'deficiencies': [],
                'foods': {},
                'description': 'No specific information available'
            })
            
            # Get all probabilities
            all_probabilities = {}
            for i, prob in enumerate(probabilities[0]):
                all_probabilities[self.decoding_classes[str(i)]] = prob.item()
            
            return {
                'predicted_class': predicted_class_name,
                'confidence': confidence_score,
                'class_index': predicted_class_idx,
                'all_probabilities': all_probabilities,
                'vitamin_deficiencies': vitamin_data['deficiencies'],
                'vitamin_foods': vitamin_data['foods'],
                'condition_description': vitamin_data['description']
            }
        except Exception as e:
            print(f"Error in prediction: {e}")
            raise e