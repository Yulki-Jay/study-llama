def idTotext():
    id2text = {}
    text = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    for i in range(len(text)):
        id2text[i] = text[i]
    return id2text

def textToid():
    text2id = {}
    text = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    for i in range(len(text)):
        text2id[text[i]] = i
    return text2id    
    
    
