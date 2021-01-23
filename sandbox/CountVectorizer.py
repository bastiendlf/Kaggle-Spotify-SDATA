from sklearn.feature_extraction.text import CountVectorizer

text = ["hello world", "sdata on aime bien"]

vectorizer = CountVectorizer()

# tokenize and build vocab
vectorizer.fit(text)
print(vectorizer.vocabulary_)
# encode document

vector = vectorizer.transform(text)
# summarize encoded vector

print(vector.toarray())

def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return  listOfKeys


vocab = list()

for i in range(len(vectorizer.vocabulary_)):
    vocab.append(getKeysByValue(vectorizer.vocabulary_, i)[0])

print(vocab)