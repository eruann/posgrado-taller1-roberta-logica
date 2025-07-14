#!/usr/bin/env python3
"""
Script para analizar la estructura de los datasets SNLI y FOLIO
"""

from datasets import load_dataset
import pandas as pd

def analyze_dataset_structure():
    print("=== ANÁLISIS DE ESTRUCTURA DE DATASETS ===\n")
    
    # Cargar datasets
    print("Cargando SNLI...")
    snli = load_dataset('arrow', data_files={'train': 'data/snli/dataset/data-00000-of-00001.arrow'})
    
    print("Cargando FOLIO...")
    folio = load_dataset('arrow', data_files={'train': 'data/folio/dataset/data-00000-of-00001.arrow'})
    
    print("\n" + "="*50)
    print("SNLI DATASET")
    print("="*50)
    print(f"Splits disponibles: {list(snli.keys())}")
    print(f"Tamaño del split 'train': {len(snli['train'])}")
    print(f"Columnas: {snli['train'].column_names}")
    print(f"Tipo de dataset: {type(snli['train'])}")
    
    # Mostrar ejemplo
    print("\nEjemplo del primer registro:")
    example = snli['train'][0]
    for key, value in example.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*50)
    print("FOLIO DATASET")
    print("="*50)
    print(f"Splits disponibles: {list(folio.keys())}")
    print(f"Tamaño del split 'train': {len(folio['train'])}")
    print(f"Columnas: {folio['train'].column_names}")
    print(f"Tipo de dataset: {type(folio['train'])}")
    
    # Mostrar ejemplo
    print("\nEjemplo del primer registro:")
    example = folio['train'][0]
    for key, value in example.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*50)
    print("ANÁLISIS DE ETIQUETAS")
    print("="*50)
    
    # SNLI labels
    snli_labels = set(snli['train']['label'])
    print(f"SNLI - Etiquetas únicas: {snli_labels}")
    print(f"SNLI - Distribución de etiquetas:")
    label_counts = pd.Series(snli['train']['label']).value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(snli['train'])*100:.1f}%)")
    
    # FOLIO labels
    folio_labels = set(folio['train']['label'])
    print(f"\nFOLIO - Etiquetas únicas: {folio_labels}")
    print(f"FOLIO - Distribución de etiquetas:")
    label_counts = pd.Series(folio['train']['label']).value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(folio['train'])*100:.1f}%)")

if __name__ == "__main__":
    analyze_dataset_structure() 