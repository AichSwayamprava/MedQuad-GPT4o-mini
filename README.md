# Enhancing Medical Question Answering with Fine-Tuned GPT-4o-Mini Model on MedQuad Dataset

Code and resources for enhancing medical question answering (QA) using a fine-tuned GPT-4o-mini model. 

## Dependency Installation

Before running the project, ensure that all the required dependencies are installed. Follow the steps below to set up your environment:

`This project uses OpenAI's API key to finetune the GPT-4o-mini model. So, create an API key and keep it saved somewhere safe and accesible.` 

### Install Python Libraries
Run the following command to install the required Python libraries:
`pip install openai datasets jsonlines sentence-transformers nltk scikit-learn`

## Project Description

This project investigates the application of fine-tuning the GPT-4o-mini language model for medical QA tasks using the MedQuad dataset. The primary objective is to compare the effectiveness of fine-tuning against prompt-based learning approaches (zero-shot, one-shot, few-shot) in improving medical QA performance.

---

## Contents

1. [Introduction](#introduction)  
2. [Problems and Solutions](#problems-and-solutions)  
3. [Methodology](#methodology)  
   - [3.1 Data Collection](#31-data-collection)  
   - [3.2 Data Cleaning and Preprocessing](#32-data-cleaning-and-preprocessing)  
   - [3.3 Experiments](#33-experiments)  
   - [3.4 Evaluation](#34-evaluation)  
4. [Results](#results)  

---

## 1. Introduction

This project focuses on developing a medical chatbot designed to empower patients with immediate access to medical information and services. Traditional methods for obtaining health information—such as scheduling appointments or online searches—can be time-consuming and overwhelming. The chatbot leverages the power of artificial intelligence (AI) and natural language processing (NLP) techniques, utilizing OpenAI's GPT-4o-mini model, to address this challenge. By offering an interactive and personalized platform, it aims to enhance patient awareness and promote timely healthcare interventions.

---

## 2. Problems and Solutions

### Challenges in Healthcare  
- **Limited Access to Healthcare Professionals**: Patients often face long wait times, geographical barriers, or financial constraints, leading to delays in diagnosis and treatment.  
  - **Solution**: The chatbot provides immediate access to medical information, reducing dependency on healthcare professionals for routine inquiries.  

- **Delayed Medical Attention**: Lack of awareness about symptom severity can delay critical interventions.  
  - **Solution**: The chatbot educates patients about symptoms and potential risks, encouraging earlier medical attention, especially in cases like cancer, where early detection is crucial.

---

## 3. Methodology

### 3.1 Data Collection  
- **Dataset**: The MedQuad dataset, curated for medical QA tasks, was used in this project.  
- **Source**: [MedQuad Dataset on Hugging Face](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset)  
- **Data Format**: The dataset contains 16,407 rows of data with the following columns:
  - `qtype`: Type of medical question.
  - `Question`: Medical query in text form.
  - `Answer`: Corresponding evidence-based answer.

---

### 3.2 Data Cleaning and Preprocessing  

- **Cleaning**: 
  - Removed missing and duplicate values.
- **Data Sampling**: A random subset of 1,000 records was selected for experiments.  
- **Formatting**: Data was preprocessed to align with the GPT-4o-mini model's input format.  
- **Splitting**: The dataset was split into:
  - Training Set (80%)  
  - Validation Set (10%)  
  - Test Set (10%)  

---

### 3.3 Experiments

- **Prompt Engineering**: Baseline experiments included:
  - **Zero-Shot Learning**: Generating answers without examples.  
  - **One-Shot Learning**: Providing one example along with the query.  
  - **Few-Shot Learning**: Using multiple examples to guide responses.  

- **Fine-Tuning**: The GPT-4o-mini model was fine-tuned on the training subset of the MedQuad dataset, optimizing model parameters to enhance performance.

---

### 3.4 Evaluation

- **Metrics**: The model's performance was assessed using:  
  - **Exact Match Accuracy**: Measures exact matches with the ground truth.  
  - **BLEU Score**: Evaluates n-gram overlap between generated and expected answers.  
  - **Semantic Similarity**: Assesses the semantic alignment between responses using embeddings.  
  - **Cosine Similarity**: Quantifies similarity between response vectors and ground truth vectors.  
  - **Token-Level Accuracy**: Measures token overlap between answers.  
  - **Accuracy and F1 Score**: Evaluates overall token classification performance.  

- **Comparison**: The fine-tuned model's performance was compared with zero-shot, one-shot, and few-shot results to evaluate the impact of fine-tuning.  

- **Human Evaluation**: Evaluators assessed the chatbot on:
  - Medical Accuracy
  - Guideline Adherence
  - Clarity
  - Empathy
  - Response Relevance

---

## 4. Results

- **Performance Analysis**:  
  Fine-tuning GPT-4o-mini resulted in significant improvements across most evaluation metrics, particularly in semantic similarity and response relevance. The model demonstrated enhanced understanding of medical terminology, relationships, and nuanced language, leading to more comprehensive answers.  

- **Key Findings**:  
  Fine-tuning outperformed prompt-based methods, but challenges in exact match accuracy and handling ambiguous queries remain areas for improvement.  

---  

-**NOTE**: **Complete code and Research paper has been uploaded in this repository for complete work.**
