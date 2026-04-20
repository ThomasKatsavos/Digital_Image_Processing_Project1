# Εισάγουμε τη βιβλιοθήκη numpy και τις απαιτούμενες συναρτήσεις από τα άλλα αρχεία
import numpy as np
from dft2_function import DFT2, iDFT2
from assist_functions import spectrum_mag

#Συνάρτηση για σχεδιασμό Βαθυπερατού Φίλτρου
def designLowFilter(N1: int, N2: int, param : tuple)-> np.ndarray:
   #Η μεταβλητή D0-συχνότητα αποκοπής- από παράμετρο εισόδου
   D0 = param[1]
   # Η τάξη για Butt. φίλτρο
   n = param[2]
   # Οι 'συντεταγμένες' για κάθε σημείο της εικόνας
   u, v = np.meshgrid(np.arange(N1), np.arange(N2) , indexing='ij')

   #Η ευκλείδια απόσταση κάθε σημείου 
   D = np.sqrt((u-N1/2)**2 + (v-N2/2)**2)

   # Η περίπτωση εφαρμογής Ιδανικού Φίλτρου 
   if param[0] == 0:
      H = (D<=D0).astype(float)
   
   # Η περίπτωση εφαρμογής Butterworth Φίλτρου 
   elif param[0] == 1: 
      H = 1 / (1 + (D/D0)**(2*n))
   
   # Η περίπτωση εφαρμογής Gaussian Φίλτρου 
   elif param[0] == 2:
      H = np.exp(-(D**2)/(2 * (D0**2)))

   # Αν η παράμετρος δεν ακολουθεί τη σύμβαση της εκφώνησης, επιστρέφεται κενός πίνακας φίλτρου 
   else:
      H = None 
   
   return H

#Συνάρτηση για σχεδιασμό Υψιπερατού Φίλτρου
def designHighFilter(N1: int, N2: int, param : tuple)-> np.ndarray:
   #Ορισμός του Υψιπερατού μέσω της συμπληρωματικότητάς του με ένα βαθυπερατό ίδιας συχνότητας αποκοπής
   H_lp = designLowFilter(N1,N2,param)
   H = 1-H_lp
   
   return H

#Συνάρτηση για εφαρμογή δεδομένου φίλτρου
def freqFilter(f: np.ndarray, H: np.ndarray, N1: int, N2: int, title) -> np.ndarray:
   F = DFT2(f, N1,N2, 1) # FFT της αρχικής εικόνας
   G = F * H #Γινόμενο στις συχνότητς <-> συνέλιξη στο χρόνο - για εφαρμογή φίλτρου
   
   spectrum_mag(G, title)
   
   g = iDFT2(G, N1, N2, 1) # επιστρεφόμενη εικόνα εξόδου
   
   return g
