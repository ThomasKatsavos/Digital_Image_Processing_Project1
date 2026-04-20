# Εισάγουμε τη βιβλιοθήκη numpy και τις απαιτούμενες συναρτήσεις από τα άλλα αρχεία
import numpy as np


# Ορίζεται η συνάρτηση για τα ιδανικά κατευθυντικά φίλτρα
def designDirectionalFilter(N1: int, N2: int, D0: float, q: int)-> np.ndarray:
   H = np.zeros((N1, N2,q+1))  # Αρχικά δημιουργούμε ως μηδενικό πίνακα τη συνάρτηση μεταφοράς για όλα τα φίλτρα
   
   #Εξάγουμε τις μεταβλητές για τα σημεία της εικόνας
   n1,n2 = np.meshgrid(np.arange(N1),np.arange(N2), indexing='ij')

   # Εφαρμόζουμε κανονικοποίηση όπως στην εκφώνηση
   u1 = n1-N1/2
   u2 = n2 - N2/2
   
   #Υπολογίζουμε την ευκλείδια απόσταση D των κανονικ. σημείων
   D = np.sqrt(u1**2+u2**2)
   fi = np.arctan2(u2,u1)

   #Όρίζουμε το πρώτο φίλτρο το οποίο είναι Ιδανικό LP
   H[:,:, 0] = (D<=D0).astype(float)

   #Δημιουργία της συστοιχίας(μπάνκας) q κατευθυντικών φίλτρων
   for i in range(1,q+1):
      #Ορίζουμε τις γωνίες θi
       th_i = (i-1) * np.pi/q
       #Ορίζουμε τη διαφορά arctan('u2/'u1 - θi) της εκφώνησης
       angle_diff = fi - th_i

      #Kανονικοποίηση των γωνιών(και των συμμετρικών/αντιδιαμετρικών τους) στο εύρος από -π έως π
       angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
       angle_diff_symm = (fi - (th_i + np.pi) + np.pi) % (2 * np.pi) - np.pi
       
       # Βοηθητική μεταβλητή d- χρηση μιας πολύ μικρής τιμής για να αποφευχθεί λάθος αριθμητική στρογγυλοποίηση
       # που θα οδηγούσε σε μηδενισμό του φίλτρου σε μια ευθεία εντός του εύρους διέλευσης(παρατηρήθηκε στα π/4)
       d = 1e-7
       
       #Χρήση μάσκας, δηλαδή προσδιορισμός όλων των σημείων που ανήκουν στο 'pass band'
       mask = (D>D0) & ((np.abs(angle_diff) <= np.pi/(2*q)+d) | (np.abs(angle_diff_symm) <= np.pi/(2*q)+d))
       H[mask, i] = 1
   return H

# Ακολουθεί η ίδια ακριβώς διαδικασία για συστοιχία Εξομαλυμένων Κατευθυντικών Φίλτρων
def designSmoothDirectionalFilter(N1: int, N2: int, D0: float, q: int, K: int)->np.ndarray:
   H = np.zeros((N1, N2,q+1))

   n1,n2 = np.meshgrid(np.arange(N1),np.arange(N2), indexing='ij')

   u1 = n1-N1/2
   u2 = n2 - N2/2

   D = np.sqrt(u1**2+u2**2)
   fi = np.arctan2(u2,u1)

   H[:,:, 0] = (D<=D0).astype(float)
   
   sum = np.zeros((N1,N2))#Πίνακας που αποθηκεύει ένα φίλτρο που χρησιμεύει για το άθροισμα φίλτρων

   for i in range(1,q+1):
      th_i = (i-1) * np.pi/q

      angle_diff = fi - th_i

      angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
      angle_diff_symm = (fi - (th_i + np.pi) + np.pi) % (2 * np.pi) - np.pi
      
      d = 1e-7

      mask = (D>D0) & ((np.abs(angle_diff) <= np.pi/2 +d) | (np.abs(angle_diff_symm) <= np.pi/2+d)) 
      #Εδώ σε αντίθεση με την προηγούμενη συνάρτηση κάνουμε χρήση εξομάλυνσης, με ημιτονοειδή συνάρτηση στην Κ δύναμη 
      #Παρακάτω προκύπτει η μη κανονικοποιημένη μορφή
      H[mask, i] = np.cos(angle_diff[mask])**K

      #Eδω επαναληπτικά προσθέτουμε τις μη κανονικοποιημένες συναρτήσεις για την τελική διαίρεση/κανονικοποίηση
      sum += H[:,:,i]


   mask_norm = D>D0 #Ορισμός νέας μάσκας για το άθροισμα συναρτησεων προς κανονικοποίηση
   #Επαναληπτική διαίρεση για κανονικοποίηση των τελικών κατευθυντικών φίλτρων
   for i in range(1,q+1):
      H[mask_norm, i] /= sum[mask_norm]

   return H