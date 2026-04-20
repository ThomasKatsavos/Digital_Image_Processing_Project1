import numpy as np
from PIL import Image
from dft2_function import DFT2, iDFT2
from lp_hp_filters import designLowFilter, designHighFilter,freqFilter
from directional_filters import designDirectionalFilter, designSmoothDirectionalFilter
from assist_functions import print_out, spectrum_mag


# Άνοιγμα/'Διάβασμα' της grayscale εικόνας. Αλλάζοντας το όνομα 'cameraman.png' επιλέγουμε την εικόνα που θέλουμε από το ίδιο directory
img = Image.open('cameraman.png').convert('L')
f = np.array(img) #ανάθεση της εικόνας σε ένα πίνακα

# Τυπώνονται οι διαστάσεις και ο τύπος δεδομένων της grayscale εικόνας
print("Διαστάσεις της εικόνας: ", f.shape)
print("Τύπος των Grayscale δεδομένων: ", f.dtype)

print_out(f, 'Original Image')
#Οι μεταβλητές Ν1,2 παίρνουν τις τιμές των διαστάσεων της εικόνας
N1,N2 = f.shape



# Παρακάτω περιλαμβάνονται με τη σειρά όπως στο κείμενο της εκφώνησης τα ζητούμενα Μέρη της
# εργασίας ως συναρτήσεις μεταφοράς και εξόδου στο πεδίο των συχνοτήτων. Επίσης, για κάθε μέρος, με την εκτέλεση 
# του κώδικα θα εκτυπώνεται το συχνοτικό διάγραμμα πλάτους του 2-D DFT καθώς και η επεξεργασμένη εικόνα-έξοδος του κάθε φίλτρου

#Aρχικά βρίσκουμε τον DFT της εικόνας
F = DFT2(f,N1,N2,1)
spectrum_mag(F, "Original Image(Spatial Frequency Domain)")

#Eπιπλέον έλεγχος για τις διαστάσεις της εικόνας, για χρήση με τις συναρτήσεις φίλτρων
if N1 % 2 == 1:
      N1 = N1 - 1
   
if N2 % 2 == 1:
   N2 = N2 - 1

################################
# Μέρος 1: 2 Ιδανικά Βαθυπερατά Φίλτρα
H1 = designLowFilter(N1,N2,(0,min(N1,N2)/5,None))
H2 = designLowFilter(N1,N2,(0,min(N1,N2)/2,None))

#Eκτύπωση του φάσματος χωρ. συχνοτήτων και την τελικής εικόνας
g_lp1 = freqFilter(f,H1, N1, N2, 'Ideal Low Pass Filter (min[N1,N2]/5)')
print_out(g_lp1, 'Ideal Low Pass Filter (min[N1,N2]/5)')

g_lp2 = freqFilter(f,H2, N1, N2, 'Ideal Low Pass Filter (min[N1,N2]/2)')
print_out(g_lp2, 'Ideal Low Pass Filter (min[N1,N2]/2)')

#################################
# Μέρος 2: 2 Ιδανικά Υψιπερατά(συμπληρωματικά) Φίλτρα
H_hp1 = designHighFilter(N1,N2,(0,min(N1,N2)/5,None))
H_hp2 = designHighFilter(N1,N2,(0,min(N1,N2)/2,None))

#Eκτύπωση του φάσματος χωρ. συχνοτήτων και την τελικής εικόνας
g_hp1 = freqFilter(f,H_hp1, N1, N2, 'Ideal High Pass Filter (min[N1,N2]/5)')
print_out(g_hp1, 'Ideal High Pass Filter (min[N1,N2]/5)')

g_hp2 = freqFilter(f,H_hp2, N1, N2, 'Ideal High Pass Filter (min[N1,N2]/2)')
print_out(g_hp2, 'Ideal High Pass Filter (min[N1,N2]/2)')

################################
# Μέρος 3: 2 Gaussian Bαθυπερατά Φίλτρα
H1g = designLowFilter(N1,N2,(2,min(N1,N2)/5,None))
H2g = designLowFilter(N1,N2,(2,min(N1,N2)/2,None))

#Eκτύπωση του φάσματος χωρ. συχνοτήτων και την τελικής εικόνας
g_lp1g = freqFilter(f,H1g, N1, N2, 'Gaussian Low Pass Filter (min[N1,N2]/5)')
print_out(g_lp1g, 'Gaussian Low Pass Filter (min[N1,N2]/5)')

g_lp2g = freqFilter(f,H2g, N1, N2, 'Gaussian Low Pass Filter (min[N1,N2]/2)')
print_out(g_lp2g, 'Gaussian Low Pass Filter (min[N1,N2]/2)')

###################################
# Μέρος 4: 2 Gaussian Υψιπερατά(συμπληρωματικά) Φίλτρα
H_hp1g = designHighFilter(N1,N2,(2,min(N1,N2)/5,None))
H_hp2g = designHighFilter(N1,N2,(2,min(N1,N2)/2,None))

#Eκτύπωση του φάσματος χωρ. συχνοτήτων και την τελικής εικόνας
g_hp1g = freqFilter(f,H_hp1g, N1, N2, 'Gaussian High Pass Filter (min[N1,N2]/5)')
print_out(g_hp1g, 'Gaussian High Pass Filter (min[N1,N2]/5)')

g_hp2g = freqFilter(f,H_hp2g, N1, N2, 'Gaussian High Pass Filter (min[N1,N2]/2)')
print_out(g_hp2g, 'Gaussian High Pass Filter (min[N1,N2]/2)')

############################################
#Μέρος 5: Συστοιχία 1 Βαθυπερατού και 8 Ιδανικών  Κατευθυντικών Φίλτρων
H_dir = designDirectionalFilter(N1,N2, min(N1,N2)/8, 8)

#Εφαρμογή μόνο ενός κατευθυντικού φίλτρου(τυχαία)
g_dir1 =  freqFilter(f, H_dir[:,:,7], N1,N2, 'Single Ideal Directional Filter')
print_out(g_dir1, 'Single Ideal Directional Filter')

#Εφαρμογή όλης της συστοιχίας
F_dir = np.zeros_like(F)
for i in range(H_dir.shape[2]):
   F_dir += F  * H_dir[:,:,i] 

spectrum_mag(F_dir, 'Array of directional Filters and a LP filter')
g_dir = iDFT2(F_dir, N1,N2, 1)
print_out(g_dir, 'Array of directional Filters and a LP filter')


###########################################
#Μέρος 6: Συστοιχία 1 Βαθυπερατού και 8 Εξομαλυμένων  Κατευθυντικών Φίλτρων
K = 2 #Eπιλογή παραμέτρου για εκθέτη Κ
H_dirs = designSmoothDirectionalFilter(N1,N2, min(N1,N2)/8, 8, K)

#Εφαρμογή μόνο ενός κατευθυντικού φίλτρου(τυχαία)
g_dirs1 =  freqFilter(f, H_dirs[:,:,5], N1,N2, f"Single Smoothing Directional Filter - K={K}")
print_out(g_dirs1, f"Single Smoothing Directional Filter - K={K}")

#Εφαρμογή όλης της συστοιχίας
F_dirs = np.zeros_like(F)
for i in range(H_dirs.shape[2]):
   F_dirs += F  * H_dirs[:,:,i] 

spectrum_mag(F_dirs, f"Array of directional Smoothing Filters and a LP filter - Κ={K}")
g_dirs = iDFT2(F_dirs, N1,N2, 1)
print_out(g_dirs, f"Array of directional Smoothing Filters and a LP filter - Κ={K}")

#TEΛΟΣ