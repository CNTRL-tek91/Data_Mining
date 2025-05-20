

# Importing some Python libraries
import csv
import math
from sklearn.metrics.pairwise import cosine_similarity

documents = []

#reading the documents in a csv file
with open('cleaned_documents.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         documents.append (row)
         #print(row)



#Building the document-term matrix by using binary encoding.
#You must identify each distinct word in the collection without applying any transformations, using
# the spaces as your character delimiter.


#gets the text of the document and split it into its terms
def getDocText(docText):
    docTerms = docText.split()
    return docTerms

#creates a dictionary and stores the frequency of each term
def termFreq(docTerms):
   termDict = {}

   for term in docTerms:
          if term in termDict:
             termDict[term] += 1
          else:
             termDict[term] = 1
   return termDict
   
docTermMatrix = []

for i in documents:
    '''Since the data in the csv file is structured with
    the id first followed by the text, we need to use i[1] 
    to only grab the text
    '''
    docText = i[1]
    #Calling the created functions
    termsList = getDocText(docText)
    wordFreq = termFreq(termsList)

    #creating a vector to store the frequency of each term
    textVector = []
    for term in sorted(wordFreq):
        textVector.append(wordFreq.get(term, 0))
    docTermMatrix.append(textVector)
   


# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors


#Function to calculate the cosine similarity of X and Y
def cosineSimilarity(x, y):
    
    dotProduct = sum(x*y for x, y in zip(x, y))
    magX = math.sqrt(sum(x*x for x in x))
    magY = math.sqrt(sum(y*y for y in y))

    if magX != 0 and magY != 0:
        return dotProduct / (magX * magY)


highSimVal = 0
doc1ID = 0
doc2ID = 0


for i in range(len(documents)):

    for j in range(i+1, len(documents)):
        #cosine_similarity([X], [Y])
        docSimilarity = cosineSimilarity(docTermMatrix[i], docTermMatrix[j])
        
        
        
        if docSimilarity > highSimVal:
            #Will update when the highest similarity is found
            highSimVal = docSimilarity
            doc1ID = i + 1
            doc2ID = j + 1

    
    




# Print the highest cosine similarity following the information below
# The most similar documents are document 10 and document 100 with cosine similarity = x

print(f"The most similar documents are document {doc1ID} and document {doc2ID} with cosine similarity = {highSimVal}.")