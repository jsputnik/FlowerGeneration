# FlowerGeneration
Uruchomienie projektu
1. Stworzenie i aktywacja wirtualnego srodowiska. Do tego celu mozna wykorzystać
narzędzie virtualenv:
    - stworzenie środowiska: virtualenv [nazwa środkowiska]
    - aktywacja środowiska: [nazwa środowiska]/Scripts/activate
2. Pobranie zależności zawartych w pliku requirements.txt:
    - pip install -r requirements.txt
3. Uruchomienie z poziomu konsoli wybranego pliku:
    - aplikacja webowa:

        python webapp.py
    - trenowanie sieci neuronowej odpowiedzialnej za segmentację kwiatów:
    
        python train_general_nn.py
    - testowanie sieci neuronowej odpowiedzialnej za segmentację kwiatów na zbiorze danych:
  
        python test_general_nn.py
    - testowanie sieci neuronowej odpowiedzialnej za segmentację kwiatów na pojedynczym zdjęciu:
    
        python single_segment_flower.py
        
    W celu poprawnego uruchomienia skryptów związanych z sieciami neuronowymi,
    nalezy w nich podmienić następujące parametry:
    - save_model_path - ściezka pod którą zostanie zapisany wytrenowany model
    (train_general_nn.py)
    - model_path - ścieżka do testowanego modelu sieci
    (test_general_nn.py oraz single_segment_flower.py)
    - model - architektura sieci, która musi odpowiadać architekturze modelu zawartemu pod zmienną model_path
    (train_general_nn.py, test_general_nn.py oraz single_segment_flower.py)
    - image_path - ściezka do zdjęcia testowego
    (single_segment_flower.py)
    - data_path - ściezka do zbioru z oryginalnymi zdjęciami
    (train_general_nn.py oraz test_general_nn.py)
    - masks_path - ściezka do zbioru z maskami
    (train_general_nn.py oraz test_general_nn.py)
