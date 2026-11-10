===========================================================
BEYİN TÜMÖRÜ SINIFLANDIRMA VE ANALİZ SİSTEMİ (XAI DESTEKLİ)
===========================================================

PROJE TANIMI:
Bu proje, MRI görüntülerinden 4 farklı beyin tümörü tipini 
(Glioma, Meningioma, No Tumor, Pituitary) %98.86 doğrulukla 
sınıflandıran ve Grad-CAM++ ile görsel açıklama sunan bir 
derin öğrenme sistemidir.

VERİ SETİ (DATASET):
Projede kullanılan veri seti Kaggle üzerinden temin edilmiştir.
Dosya boyutu nedeniyle zip içerisine eklenmemiştir.
Link: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

DOSYA YAPISI:
- 01_ResNet50_Final_Egitim.ipynb: Ana modelin eğitim kodları.
- 02_Model_Karsilastirmalari.ipynb: Diğer mimarilerle kıyaslama kodları.
- app.py: Kullanıcı arayüzü (Streamlit) kodu.
- ResNet50_best.pth: %98.86 başarı oranına sahip model ağırlıkları.
- requirements.txt: Gerekli Python kütüphaneleri listesi.

ÇALIŞTIRMA TALİMATLARI:
1. Gerekli kütüphaneleri kurun:
torch
torchvision
streamlit
numpy
pandas
Pillow
opencv-python
matplotlib

2. Veri setini yukarıdaki linkten indirin ve "Testing" ile 
   "Training" klasörlerini proje dizinine yerleştirin.

3. Arayüzü başlatın:
   streamlit run app5.py

4. Tarayıcıda açılan sayfada MRI görüntüsü yükleyerek 
   analiz sonuçlarını görüntüleyebilirsiniz.

NOTLAR:
- Eğitim süreci ResNet50 mimarisi üzerine Transfer Learning uygulanarak yapılmıştır.
- Tahminleme sırasında kullanılan ısı haritaları, modelin hangi anatomik
  bölgelere odaklandığını göstermektedir.
===========================================================