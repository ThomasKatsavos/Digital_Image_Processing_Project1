# Εισαγωγή της numpy για χρήση των fft συναρτήσεων κυρίως
import numpy as np

# Ορισμός συνάρτησης για υπολογισμό του 2-D DFT με FFT αλγόριθμο 
def DFT2(f: np.ndarray, N1: int, N2: int, fftshift: int) ->  np.ndarray:
   #Αφαιρούμε γραμμή ή/και στήλη για εξασφάλιση ακέραιου αποτελέσματος διάιρεσης με το 2
   if N1 % 2 == 1:
      N1 = N1 - 1
   
   if N2 % 2 == 1:
      N2 = N2 - 1
   #Ενημέρωση της εικόνας προς μετασχηματισμό DFT με τις πιθανώς νέες τιμές των Ν1,Ν2
   f_ready = f[:N1, :N2]

   #Υπολογισμός του Μ/Σ Fourier με την 'fft2()'
   F = np.fft.fft2(f_ready, s=(N1,N2))

   #Mεταφορά του συντελεστή DC στο κέντρο αν  η τιμή του ορίσματος fftshift() είναι 1
   if fftshift:
      F = np.fft.fftshift(F)

   return F

#Ακριβώς η αντίστοιχη λογική αλλά αντίστροφα, για τον IDFT
def iDFT2(F: np.ndarray, N1: int, N2: int, fftshift: int) -> np.ndarray:
   if N1 % 2 == 1:
      N1 = N1 - 1
   
   if N2 % 2 == 1:
      N2 = N2 - 1
      
   if fftshift:
      F = np.fft.ifftshift(F)
   
   #Εδώ η επιστρεφόμενη εικόνα είναι μιγαδικών τιμών λόγω των αριθμητικών μεθόδων και στρογγυλοποιήσεων
   f_complex = np.fft.ifft2(F, s=(N1,N2))
   #Κρατάμε μόνο το πραγματικό μέρος, αγνοώντας τις μικρές φανταστικές τιμές
   f = np.real(f_complex)
   
   return f
