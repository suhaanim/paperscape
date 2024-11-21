import os
import json
import spacy
import PyPDF2
import numpy as np
from gensim.summarization import summarize
from transformers import pipeline
from pdfminer.high_level import extract_text

class PaperProcessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.key_phrases = []
        self.summary = ""
        self.concepts = []

    def process_paper(self, file_path):
        """Process the uploaded research paper"""
        text = self._extract_text(file_path)
        self.summary = self._generate_summary(text)
        self.key_phrases = self._extract_key_phrases(text)
        self.concepts = self._identify_concepts(text)
        return self._prepare_game_data()

    def _extract_text(self, file_path):
        """Extract text from PDF or text file"""
        if file_path.endswith('.pdf'):
            return extract_text(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()

    def _generate_summary(self, text):
        """Generate a concise summary of the paper"""
        try:
            chunks = [text[i:i + 1024] for i in range(0, len(text), 1024)]
            summaries = []
            for chunk in chunks:
                summary = self.summarizer(chunk, max_length=130, min_length=30, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            return ' '.join(summaries)
        except Exception as e:
            print(f"Error generating summary: {e}")
            return text[:1000]  # Fallback to first 1000 characters

    def _extract_key_phrases(self, text):
        """Extract key phrases from the text"""
        doc = self.nlp(text)
        key_phrases = []
        
        for sent in doc.sents:
            # Extract noun phrases
            for chunk in sent.noun_chunks:
                if len(chunk.text.split()) >= 2:  # Only phrases with 2+ words
                    key_phrases.append(chunk.text)
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']:
                key_phrases.append(ent.text)
        
        return list(set(key_phrases))  # Remove duplicates

    def _identify_concepts(self, text):
        """Identify main concepts from the paper"""
        doc = self.nlp(text)
        concepts = []
        
        # Extract concepts based on specific patterns
        for sent in doc.sents:
            # Look for definition patterns
            if any(token.text.lower() in ['is', 'are', 'refers', 'defines'] for token in sent):
                concepts.append({
                    'type': 'definition',
                    'content': sent.text,
                    'keywords': [token.text for token in sent if token.pos_ in ['NOUN', 'PROPN']]
                })
            
            # Look for methodology patterns
            if any(token.text.lower() in ['method', 'approach', 'technique', 'algorithm'] for token in sent):
                concepts.append({
                    'type': 'methodology',
                    'content': sent.text,
                    'keywords': [token.text for token in sent if token.pos_ in ['NOUN', 'PROPN']]
                })
            
            # Look for result patterns
            if any(token.text.lower() in ['result', 'conclusion', 'finding', 'shows'] for token in sent):
                concepts.append({
                    'type': 'result',
                    'content': sent.text,
                    'keywords': [token.text for token in sent if token.pos_ in ['NOUN', 'PROPN']]
                })
        
        return concepts

    def _prepare_game_data(self):
        """Prepare data structure for game generation"""
        return {
            'summary': self.summary,
            'key_phrases': self.key_phrases,
            'concepts': self.concepts,
            'game_elements': {
                'quiz': self._generate_quiz_questions(),
                'simulation': self._prepare_simulation_data(),
                'puzzle': self._prepare_puzzle_data()
            }
        }

    def _generate_quiz_questions(self):
        """Generate quiz questions from the paper content"""
        questions = []
        for concept in self.concepts:
            if concept['type'] == 'definition':
                questions.append({
                    'type': 'multiple_choice',
                    'question': f"What is the correct definition of {concept['keywords'][0]}?",
                    'correct_answer': concept['content'],
                    'options': [concept['content']] + [c['content'] for c in self.concepts if c != concept][:3]
                })
            elif concept['type'] == 'methodology':
                questions.append({
                    'type': 'true_false',
                    'question': f"Is the following statement about the methodology correct? {concept['content']}",
                    'correct_answer': True,
                    'explanation': concept['content']
                })
        return questions

    def _prepare_simulation_data(self):
        """Prepare data for interactive simulations"""
        simulation_data = {
            'type': 'particle_system',
            'elements': [],
            'interactions': []
        }
        
        # Create simulation elements from concepts
        for i, concept in enumerate(self.concepts):
            element = {
                'id': f"element_{i}",
                'type': concept['type'],
                'label': concept['keywords'][0] if concept['keywords'] else f"Concept {i}",
                'description': concept['content'],
                'properties': {
                    'mass': 1 + i,
                    'charge': (-1 if i % 2 == 0 else 1),
                    'size': len(concept['keywords']) + 5
                }
            }
            simulation_data['elements'].append(element)
        
        # Create interactions between related concepts
        for i, concept1 in enumerate(self.concepts):
            for j, concept2 in enumerate(self.concepts[i+1:], i+1):
                if any(keyword in concept2['keywords'] for keyword in concept1['keywords']):
                    interaction = {
                        'source': f"element_{i}",
                        'target': f"element_{j}",
                        'strength': 0.5,
                        'type': 'attraction'
                    }
                    simulation_data['interactions'].append(interaction)
        
        return simulation_data

    def _prepare_puzzle_data(self):
        """Prepare data for drag-and-drop puzzles"""
        puzzle_data = {
            'type': 'concept_map',
            'nodes': [],
            'connections': []
        }
        
        # Create puzzle pieces from concepts
        for i, concept in enumerate(self.concepts):
            node = {
                'id': f"node_{i}",
                'type': concept['type'],
                'content': concept['content'],
                'keywords': concept['keywords'],
                'position': {
                    'x': np.random.randint(100, 700),
                    'y': np.random.randint(100, 500)
                }
            }
            puzzle_data['nodes'].append(node)
        
        # Create connections between related concepts
        for i, concept1 in enumerate(self.concepts):
            for j, concept2 in enumerate(self.concepts[i+1:], i+1):
                shared_keywords = set(concept1['keywords']) & set(concept2['keywords'])
                if shared_keywords:
                    connection = {
                        'source': f"node_{i}",
                        'target': f"node_{j}",
                        'label': list(shared_keywords)[0],
                        'strength': len(shared_keywords)
                    }
                    puzzle_data['connections'].append(connection)
        
        return puzzle_data
