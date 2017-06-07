#Calssification Autism brains and Normal brains
#Import packages
import numpy
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from scipy import interp
from bct.algorithms.clustering import clustering_coef_wu
from  bct.algorithms.distance import charpath
from bct.algorithms.modularity import community_louvain
from bct.algorithms.distance import efficiency_wei 
##########################################################################
#function to caculate(measure) the four matrices
def caculate_matrices(A,n):
	b_modularity=numpy.zeros(n)
	b_charpath=numpy.zeros(n)
	b_clustering_coef=numpy.zeros(n)
	b_efficiency_wei=numpy.zeros(n)
	All_case_brains=numpy.zeros((n,4))
	for i in range(n):
		b_modularity[i]=community_louvain(A[:,:,i])[1]
		b_charpath[i]=charpath (A[:,:,i])[0]
		b_clustering_coef[i]=numpy.mean(clustering_coef_wu(A[:,:,i]))
		b_efficiency_wei[i]=efficiency_wei(A[:,:,i])
		All_case_brains[i]=[b_modularity[i],b_charpath[i],b_clustering_coef[i],b_efficiency_wei[i]]
	return All_case_brains
################################# Main_function ##############################################
def main():
######################### Defining_Autism_brains #########################
	n=31
	A= numpy.zeros((90,90,n))   
	A[:,:,0]= numpy.loadtxt(open("28853.csv", "rb"), delimiter=",")
	A[:,:,1]= numpy.loadtxt(open("28855.csv", "rb"), delimiter=",")
	A[:,:,2]= numpy.loadtxt(open("28856.csv", "rb"), delimiter=",")
	A[:,:,3]= numpy.loadtxt(open("28857.csv", "rb"), delimiter=",")
	A[:,:,4]= numpy.loadtxt(open("28859.csv", "rb"), delimiter=",")
	A[:,:,5]= numpy.loadtxt(open("28860.csv", "rb"), delimiter=",")
	A[:,:,6]= numpy.loadtxt(open("28861.csv", "rb"), delimiter=",")
	A[:,:,7]= numpy.loadtxt(open("28864.csv", "rb"), delimiter=",")
	A[:,:,8]=numpy.loadtxt(open("28865.csv", "rb"), delimiter=",")
	A[:,:,9]= numpy.loadtxt(open("28866.csv", "rb"), delimiter=",")
	A[:,:,10]= numpy.loadtxt(open("28871.csv", "rb"), delimiter=",")
	A[:,:,11]= numpy.loadtxt(open("28872.csv", "rb"), delimiter=",")
	A[:,:,12]= numpy.loadtxt(open("28873.csv", "rb"), delimiter=",")
	A[:,:,13]= numpy.loadtxt(open("28874.csv", "rb"), delimiter=",")
	A[:,:,14]= numpy.loadtxt(open("28875.csv", "rb"), delimiter=",")
	A[:,:,15]= numpy.loadtxt(open("28876.csv", "rb"), delimiter=",")
	A[:,:,16]= numpy.loadtxt(open("28879.csv", "rb"), delimiter=",")
	A[:,:,17]= numpy.loadtxt(open("28885.csv", "rb"), delimiter=",")
	A[:,:,18]= numpy.loadtxt(open("28887.csv", "rb"), delimiter=",")
	A[:,:,19]= numpy.loadtxt(open("28890.csv", "rb"), delimiter=",")
	A[:,:,20]= numpy.loadtxt(open("28896.csv", "rb"), delimiter=",")
	A[:,:,21]= numpy.loadtxt(open("28897.csv", "rb"), delimiter=",")
	A[:,:,22]= numpy.loadtxt(open("28898.csv", "rb"), delimiter=",")
	A[:,:,23]= numpy.loadtxt(open("28899.csv", "rb"), delimiter=",")
	A[:,:,24]= numpy.loadtxt(open("28901.csv", "rb"), delimiter=",")
	A[:,:,25]= numpy.loadtxt(open("28903.csv", "rb"), delimiter=",")
	A[:,:,26]= numpy.loadtxt(open("28905.csv", "rb"), delimiter=",")
	A[:,:,27]= numpy.loadtxt(open("28906.csv", "rb"), delimiter=",")
	A[:,:,28]= numpy.loadtxt(open("28907.csv", "rb"), delimiter=",")
	A[:,:,29]= numpy.loadtxt(open("28908.csv", "rb"), delimiter=",")
	A[:,:,30]= numpy.loadtxt(open("28909.csv", "rb"), delimiter=",")
############################Defining_Normal_brains#################################
	m=24
	B=numpy.zeros((90,90,m)) 
	B[:,:,0]= numpy.loadtxt(open("28854.csv", "rb"), delimiter=",")
	B[:,:,1]= numpy.loadtxt(open("28858.csv", "rb"), delimiter=",")
	B[:,:,2]= numpy.loadtxt(open("28862.csv", "rb"), delimiter=",")
	B[:,:,3]= numpy.loadtxt(open("28863.csv", "rb"), delimiter=",")
	B[:,:,4]= numpy.loadtxt(open("28867.csv", "rb"), delimiter=",")
	B[:,:,5]= numpy.loadtxt(open("28868.csv", "rb"), delimiter=",")
	B[:,:,6]= numpy.loadtxt(open("28870.csv", "rb"), delimiter=",")
	B[:,:,7]= numpy.loadtxt(open("28877.csv", "rb"), delimiter=",")
	B[:,:,8]= numpy.loadtxt(open("28878.csv", "rb"), delimiter=",")
	B[:,:,9]= numpy.loadtxt(open("28880.csv", "rb"), delimiter=",")
	B[:,:,10]= numpy.loadtxt(open("28881.csv", "rb"), delimiter=",")
	B[:,:,11]= numpy.loadtxt(open("28882.csv", "rb"), delimiter=",")
	B[:,:,12]= numpy.loadtxt(open("28883.csv", "rb"), delimiter=",")
	B[:,:,13]= numpy.loadtxt(open("28886.csv", "rb"), delimiter=",")
	B[:,:,14]= numpy.loadtxt(open("28888.csv", "rb"), delimiter=",")
	B[:,:,15]= numpy.loadtxt(open("28889.csv", "rb"), delimiter=",")
	B[:,:,16]= numpy.loadtxt(open("28891.csv", "rb"), delimiter=",")
	B[:,:,17]= numpy.loadtxt(open("28892.csv", "rb"), delimiter=",")
	B[:,:,18]= numpy.loadtxt(open("28893.csv", "rb"), delimiter=",")
	B[:,:,19]= numpy.loadtxt(open("28894.csv", "rb"), delimiter=",")
	B[:,:,20]= numpy.loadtxt(open("28895.csv", "rb"), delimiter=",")
	B[:,:,21]= numpy.loadtxt(open("28900.csv", "rb"), delimiter=",")
	B[:,:,22]= numpy.loadtxt(open("28902.csv", "rb"), delimiter=",")
	B[:,:,23]= numpy.loadtxt(open("28904.csv", "rb"), delimiter=",")
###################################################################################
# Defining Austim[] and Normal[] brains after measuring the four matrices
	Autism=numpy.zeros((31,4))
	Normal=numpy.zeros((24,4))
	All_brains_matrices=numpy.zeros((55,4))
	Autism=caculate_matrices(A,n) # calculate the four matrices for 31 brains
	Normal=caculate_matrices(B,m) # calculate the four matrices for 24 brains
#Combine the two matrices into one matrix (All_brains_matrices
	All_brains_matrices=numpy.concatenate((Autism,Normal),axis=0)
	print("All_Brain_Matrices is ",All_brains_matrices)
	print("Dim_All_Brain_Matrices is",All_brains_matrices.shape)
#############################################################################
	X = All_brains_matrices
	y=numpy.zeros(55)
	y[0:31]=1
	y[31:55]=0
################################ Leave-One-Out ####################################################
	loo = LeaveOneOut()
	score = numpy.zeros(55)
	count = 0
	train_X=numpy.zeros((54,4))
	train_y=numpy.zeros(54)
	test_X=[0]
	test_y=[0]
	for train_index, test_index in loo.split(X):
		#print(train_index.shape,test_index.shape)
		for i in range(len(train_index)):
			train_X[i,:]=X[train_index[i]]
			train_y[i]=y[train_index[i]]
		test_X=X[test_index[0]]
		test_y=y[test_index[0]]
		clf = svm.SVC(kernel='linear', C=1, probability=True).fit(train_X, train_y)
		probs = clf.predict_proba(test_X)
                score[count] = probs[:,0]
                count+=1


############ Alessandro's ROC#############
        roc_x = []
        roc_y = []
        min_score = min(score)
        max_score = max(score)
        thr = numpy.linspace(min_score, max_score, 30)
        FP=0
        TP=0
        P = sum(y)
        N = len(y) - P

        for (i, T) in enumerate(thr):
            for i in range(0, len(score)):
               if (score[i] > T):
                 if (y[i]==1):
                   TP = TP + 1
                 if (y[i]==0):
                   FP = FP + 1
            roc_x.append(FP/float(N))
            roc_y.append(TP/float(P))
            FP=0
            TP=0
 
	roc_auc= auc(roc_x, roc_y)
##############################################################################
	#Plot of a ROC curve for a specific class
	lw = 2
	plt.plot(roc_x, roc_y, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.show()
################################# Call Main function ############################################
main()
