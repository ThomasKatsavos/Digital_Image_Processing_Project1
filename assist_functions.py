import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#Ορίζουμε μια συνάρτηση που τυπώνει το FFT διάγραμμα πλάτους μιας εικόνας
def spectrum_mag(F:np.ndarray, title):
   # Επιλέγουμε το 'Qt5Agg' graphics backend της matplotlib
   matplotlib.use('Qt5Agg') 
   
   #Λογαριθμική αναπαράσταση του φάσματος για κάθε θέση του πίνακα DFT
   magnitude_spectrum = np.log(1 + np.abs(F)) 
   
   # Aναπαράσταση του διαγράμματος/γραφικής παράστασης
   plt.figure(figsize=(10, 5))

   plt.subplot(1, 2, 2)
   plt.imshow(magnitude_spectrum, cmap='magma') 
   plt.title(title) #Ορισμός τίτλου από όρισμα της συνάρτησης
   plt.axis('off')

   plt.show()

#Ορίζουμε μια συνάρτηση που να εκτυπώνει την εικόνα μετά από την επεξεργασία
def print_out(f_reconstructed : np.ndarray, title):
   #Eξαγωγή της εικόνας προς εκτύπωση, με ανάθεση σε συγκεκριμένο πλήθος/εύρος τιμών(grayscale) και τύπου δεδομένων
   f_viewable = np.clip(f_reconstructed, 0, 255).astype(np.uint8)

   #Εκτύπωση της εικόνας 
   plt.figure(figsize=(6, 6))
   plt.imshow(f_viewable, cmap='gray')
   plt.title(title)
   plt.axis('off')
   plt.show()